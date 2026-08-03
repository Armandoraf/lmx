use lmx_core::{
    CancellationToken, ImageRequest, ProviderRegistry, ResponseFrame, ResponseMachine,
    ResponseRequest, ToolOutput, VERSION, VideoRequest, build_message_item,
    execute_round_with_cancellation, execute_round_with_observer, generate_image, generate_video,
    load_request_context, normalize_tool_output, output_text_from_items, tool_failure_output,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::{Arc, Mutex, mpsc};

type FrameReceiver = Arc<Mutex<mpsc::Receiver<ResponseFrame>>>;

fn napi_error(error: impl std::fmt::Display) -> Error {
    Error::from_reason(error.to_string())
}

fn provider_registry() -> Result<ProviderRegistry> {
    ProviderRegistry::from_environment().map_err(napi_error)
}

#[napi]
pub fn version() -> String {
    VERSION.into()
}

#[napi]
pub fn provider_registry_json() -> Result<String> {
    Ok(provider_registry()?.as_json().to_string())
}

#[napi]
pub fn load_request_context_json(provider_json: String) -> Result<String> {
    let provider = serde_json::from_str(&provider_json).map_err(napi_error)?;
    serde_json::to_string(&load_request_context(provider).map_err(napi_error)?).map_err(napi_error)
}

#[napi]
pub fn build_message_item_json(role: String, text: String) -> Result<String> {
    serde_json::to_string(&build_message_item(&role, &text).map_err(napi_error)?)
        .map_err(napi_error)
}

#[napi]
pub fn output_text_from_items_json(items_json: String) -> Result<String> {
    let items: Vec<serde_json::Map<String, serde_json::Value>> =
        serde_json::from_str(&items_json).map_err(napi_error)?;
    Ok(output_text_from_items(&items))
}

#[napi]
pub fn normalize_tool_output_json(call_id: String, value_json: String) -> Result<String> {
    let value = serde_json::from_str(&value_json).map_err(napi_error)?;
    serde_json::to_string(&normalize_tool_output(call_id, value)).map_err(napi_error)
}

#[napi]
pub fn tool_failure_output_json(call_id: String, error: String) -> Result<String> {
    serde_json::to_string(&tool_failure_output(call_id, error)).map_err(napi_error)
}

#[napi]
pub fn build_wire_request_json(request_json: String) -> Result<String> {
    let request: ResponseRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
    let machine = ResponseMachine::new(&provider_registry()?, request).map_err(napi_error)?;
    serde_json::to_string(&machine.wire_request().map_err(napi_error)?).map_err(napi_error)
}

#[napi]
pub async fn generate_image_json(request_json: String) -> Result<String> {
    let request: ImageRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
    serde_json::to_string(&generate_image(request).await.map_err(napi_error)?).map_err(napi_error)
}

#[napi]
pub async fn generate_video_json(request_json: String) -> Result<String> {
    let request: VideoRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
    serde_json::to_string(&generate_video(request).await.map_err(napi_error)?).map_err(napi_error)
}

#[napi]
pub struct ResponseSession {
    cancellation: CancellationToken,
    machine: Arc<Mutex<Option<ResponseMachine>>>,
    frames: Arc<Mutex<Option<FrameReceiver>>>,
}

#[napi]
impl ResponseSession {
    #[napi(constructor)]
    pub fn new(request_json: String) -> Result<Self> {
        let request: ResponseRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
        Ok(Self {
            cancellation: CancellationToken::new(),
            machine: Arc::new(Mutex::new(Some(
                ResponseMachine::new(&provider_registry()?, request).map_err(napi_error)?,
            ))),
            frames: Arc::new(Mutex::new(None)),
        })
    }

    #[napi]
    pub async fn execute_round_json(&self) -> Result<String> {
        let mut machine = self
            .machine
            .lock()
            .map_err(napi_error)?
            .take()
            .ok_or_else(|| Error::from_reason("response session is already executing"))?;
        let outcome = execute_round_with_cancellation(&mut machine, &self.cancellation).await;
        *self.machine.lock().map_err(napi_error)? = Some(machine);
        serde_json::to_string(&outcome.map_err(napi_error)?).map_err(napi_error)
    }

    #[napi]
    pub fn submit_tool_outputs_json(&self, outputs_json: String) -> Result<String> {
        let outputs: Vec<ToolOutput> = serde_json::from_str(&outputs_json).map_err(napi_error)?;
        let mut guard = self.machine.lock().map_err(napi_error)?;
        let machine = guard
            .as_mut()
            .ok_or_else(|| Error::from_reason("response session is already executing"))?;
        serde_json::to_string(&machine.submit_tool_outputs(outputs).map_err(napi_error)?)
            .map_err(napi_error)
    }

    #[napi]
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    #[napi]
    pub fn start_round(&self) -> Result<()> {
        let machine = self
            .machine
            .lock()
            .map_err(napi_error)?
            .take()
            .ok_or_else(|| Error::from_reason("response session is already executing"))?;
        let mut frames = self.frames.lock().map_err(napi_error)?;
        if frames.is_some() {
            return Err(Error::from_reason(
                "response session already has a running stream",
            ));
        }
        let (sender, receiver) = mpsc::channel();
        *frames = Some(Arc::new(Mutex::new(receiver)));
        let machine_slot = Arc::clone(&self.machine);
        let cancellation = self.cancellation.clone();
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    if let Ok(mut slot) = machine_slot.lock() {
                        *slot = Some(machine);
                    }
                    let _ = sender.send(ResponseFrame::Failed {
                        error: error.to_string(),
                    });
                    return;
                }
            };
            let mut machine = machine;
            let result = runtime.block_on(execute_round_with_observer(
                &mut machine,
                &cancellation,
                |event| {
                    let _ = sender.send(ResponseFrame::Event { event });
                },
            ));
            if let Ok(mut slot) = machine_slot.lock() {
                *slot = Some(machine);
            }
            let frame = match result {
                Ok(next) => ResponseFrame::Ready { next },
                Err(error) => ResponseFrame::Failed {
                    error: error.to_string(),
                },
            };
            let _ = sender.send(frame);
        });
        Ok(())
    }

    #[napi]
    pub async fn next_frame_json(&self) -> Result<String> {
        let receiver = self.frames.lock().map_err(napi_error)?.take();
        let Some(receiver) = receiver else {
            return Err(Error::from_reason("response session has no running stream"));
        };
        let worker = Arc::clone(&receiver);
        let frame = tokio::task::spawn_blocking(move || {
            worker
                .lock()
                .map_err(|error| error.to_string())?
                .recv()
                .map_err(|error| error.to_string())
        })
        .await
        .map_err(napi_error)?
        .map_err(Error::from_reason)?;
        let terminal = matches!(
            frame,
            ResponseFrame::Ready { .. } | ResponseFrame::Failed { .. }
        );
        if !terminal {
            *self.frames.lock().map_err(napi_error)? = Some(receiver);
        }
        serde_json::to_string(&frame).map_err(napi_error)
    }
}

impl Drop for ResponseSession {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}
