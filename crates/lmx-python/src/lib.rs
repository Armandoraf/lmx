use lmx_core::{
    CancellationToken, ImageRequest, ImageStreamEvent, ImageStreamRequest, ProviderRegistry,
    ResponseFrame, ResponseMachine, ResponseRequest, ToolOutput, VERSION, VideoRequest,
    build_message_item, execute_round_with_cancellation, execute_round_with_observer,
    generate_image, generate_images, generate_video, load_request_context, normalize_tool_output,
    output_text_from_items, stream_image, tool_failure_output,
};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex, mpsc};

type FrameReceiver = Arc<Mutex<mpsc::Receiver<ResponseFrame>>>;
type ImageEventReceiver = mpsc::Receiver<std::result::Result<ImageStreamEvent, String>>;
type SharedImageEventReceiver = Arc<Mutex<ImageEventReceiver>>;

fn api_error(error: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

fn provider_registry() -> PyResult<ProviderRegistry> {
    ProviderRegistry::from_environment().map_err(api_error)
}

#[pyfunction]
fn version() -> &'static str {
    VERSION
}

#[pyfunction]
fn provider_registry_json() -> PyResult<String> {
    Ok(provider_registry()?.as_json().to_string())
}

#[pyfunction]
#[pyo3(signature = (context_json=None))]
fn discover_provider_registry_json(
    py: Python<'_>,
    context_json: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let context = context_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(api_error)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(api_error)?;
        Ok(runtime
            .block_on(ProviderRegistry::discover_with_context(context.as_ref()))
            .map_err(api_error)?
            .as_json()
            .to_string())
    })
}

#[pyfunction]
fn prepare_response_request_json(py: Python<'_>, request_json: String) -> PyResult<String> {
    py.detach(move || {
        let request = serde_json::from_str(&request_json).map_err(api_error)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(api_error)?;
        serde_json::to_string(
            &runtime
                .block_on(lmx_core::prepare_response_request(request))
                .map_err(api_error)?,
        )
        .map_err(api_error)
    })
}

#[pyfunction]
fn load_request_context_json(provider_json: &str) -> PyResult<String> {
    let provider = serde_json::from_str(provider_json).map_err(api_error)?;
    serde_json::to_string(&load_request_context(provider).map_err(api_error)?).map_err(api_error)
}

#[pyfunction]
fn build_message_item_json(role: &str, text: &str) -> PyResult<String> {
    serde_json::to_string(&build_message_item(role, text).map_err(api_error)?).map_err(api_error)
}

#[pyfunction]
fn output_text_from_items_json(items_json: &str) -> PyResult<String> {
    let items: Vec<serde_json::Map<String, serde_json::Value>> =
        serde_json::from_str(items_json).map_err(api_error)?;
    Ok(output_text_from_items(&items))
}

#[pyfunction]
fn normalize_tool_output_json(call_id: &str, value_json: &str) -> PyResult<String> {
    let value = serde_json::from_str(value_json).map_err(api_error)?;
    serde_json::to_string(&normalize_tool_output(call_id, value)).map_err(api_error)
}

#[pyfunction]
fn tool_failure_output_json(call_id: &str, error: &str) -> PyResult<String> {
    serde_json::to_string(&tool_failure_output(call_id, error)).map_err(api_error)
}

/// Build the exact request that LMX's Rust engine would send. The Python
/// facade uses this during the staged migration and it gives callers a stable
/// introspection seam without reimplementing provider logic.
#[pyfunction]
fn build_wire_request_json(request_json: &str) -> PyResult<String> {
    let request: ResponseRequest = serde_json::from_str(request_json).map_err(api_error)?;
    let machine = ResponseMachine::new(&provider_registry()?, request).map_err(api_error)?;
    serde_json::to_string(&machine.wire_request().map_err(api_error)?).map_err(api_error)
}

#[pyfunction]
fn generate_image_json(request_json: &str) -> PyResult<String> {
    let request: ImageRequest = serde_json::from_str(request_json).map_err(api_error)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(api_error)?;
    serde_json::to_string(
        &runtime
            .block_on(generate_image(request))
            .map_err(api_error)?,
    )
    .map_err(api_error)
}

#[pyfunction]
fn generate_images_json(request_json: &str) -> PyResult<String> {
    let request: ImageRequest = serde_json::from_str(request_json).map_err(api_error)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(api_error)?;
    serde_json::to_string(
        &runtime
            .block_on(generate_images(request))
            .map_err(api_error)?,
    )
    .map_err(api_error)
}

#[pyclass]
struct ImageStream {
    cancellation: CancellationToken,
    events: Arc<Mutex<Option<SharedImageEventReceiver>>>,
}

#[pymethods]
impl ImageStream {
    #[new]
    fn new(request_json: &str) -> PyResult<Self> {
        let request: ImageStreamRequest = serde_json::from_str(request_json).map_err(api_error)?;
        let cancellation = CancellationToken::new();
        let (sender, receiver) = mpsc::channel();
        let worker_cancellation = cancellation.clone();
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = sender.send(Err(error.to_string()));
                    return;
                }
            };
            let result = runtime.block_on(stream_image(request, &worker_cancellation, |event| {
                sender
                    .send(Ok(event))
                    .map_err(|error| lmx_core::Error::Event(error.to_string()))
            }));
            if let Err(error) = result {
                let _ = sender.send(Err(error.to_string()));
            }
        });
        Ok(Self {
            cancellation,
            events: Arc::new(Mutex::new(Some(Arc::new(Mutex::new(receiver))))),
        })
    }

    fn next_event_json(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let receiver = self.events.lock().map_err(api_error)?.take();
        let Some(receiver) = receiver else {
            return Ok(None);
        };
        let worker = Arc::clone(&receiver);
        let item = py.detach(move || worker.lock().ok().and_then(|receiver| receiver.recv().ok()));
        match item {
            Some(Ok(event)) => {
                *self.events.lock().map_err(api_error)? = Some(receiver);
                serde_json::to_string(&event).map(Some).map_err(api_error)
            }
            Some(Err(error)) => Err(api_error(error)),
            None => Ok(None),
        }
    }

    fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl Drop for ImageStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[pyfunction]
fn generate_video_json(request_json: &str) -> PyResult<String> {
    let request: VideoRequest = serde_json::from_str(request_json).map_err(api_error)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(api_error)?;
    serde_json::to_string(
        &runtime
            .block_on(generate_video(request))
            .map_err(api_error)?,
    )
    .map_err(api_error)
}

#[pyclass]
struct ResponseSession {
    cancellation: CancellationToken,
    machine: Arc<Mutex<Option<ResponseMachine>>>,
    frames: Mutex<Option<FrameReceiver>>,
}

#[pymethods]
impl ResponseSession {
    #[new]
    fn new(request_json: &str) -> PyResult<Self> {
        let request: ResponseRequest = serde_json::from_str(request_json).map_err(api_error)?;
        Ok(Self {
            cancellation: CancellationToken::new(),
            machine: Arc::new(Mutex::new(Some(
                ResponseMachine::new(&provider_registry()?, request).map_err(api_error)?,
            ))),
            frames: Mutex::new(None),
        })
    }

    fn execute_round_json(&self) -> PyResult<String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(api_error)?;
        let mut machine = self.machine.lock().map_err(api_error)?;
        let machine = machine
            .as_mut()
            .ok_or_else(|| api_error("response session is already executing"))?;
        serde_json::to_string(
            &runtime
                .block_on(execute_round_with_cancellation(machine, &self.cancellation))
                .map_err(api_error)?,
        )
        .map_err(api_error)
    }

    fn submit_tool_outputs_json(&self, outputs_json: &str) -> PyResult<String> {
        let outputs: Vec<ToolOutput> = serde_json::from_str(outputs_json).map_err(api_error)?;
        let mut machine = self.machine.lock().map_err(api_error)?;
        let machine = machine
            .as_mut()
            .ok_or_else(|| api_error("response session is already executing"))?;
        serde_json::to_string(&machine.submit_tool_outputs(outputs).map_err(api_error)?)
            .map_err(api_error)
    }

    fn cancel(&self) {
        self.cancellation.cancel();
    }

    fn start_round(&self) -> PyResult<()> {
        let machine = self
            .machine
            .lock()
            .map_err(api_error)?
            .take()
            .ok_or_else(|| api_error("response session is already executing"))?;
        let mut frames = self.frames.lock().map_err(api_error)?;
        if frames.is_some() {
            return Err(api_error("response session already has a running stream"));
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
                    Ok(())
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

    fn next_frame_json(&self, py: Python<'_>) -> PyResult<String> {
        let receiver = self.frames.lock().map_err(api_error)?.take();
        let Some(receiver) = receiver else {
            return Err(api_error("response session has no running stream"));
        };
        let worker = Arc::clone(&receiver);
        let frame = py
            .detach(move || {
                worker
                    .lock()
                    .map_err(|error| error.to_string())?
                    .recv()
                    .map_err(|error| error.to_string())
            })
            .map_err(api_error)?;
        let terminal = matches!(
            frame,
            ResponseFrame::Ready { .. } | ResponseFrame::Failed { .. }
        );
        if !terminal {
            *self.frames.lock().map_err(api_error)? = Some(receiver);
        }
        serde_json::to_string(&frame).map_err(api_error)
    }
}

impl Drop for ResponseSession {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(provider_registry_json, module)?)?;
    module.add_function(wrap_pyfunction!(discover_provider_registry_json, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_response_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(load_request_context_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_message_item_json, module)?)?;
    module.add_function(wrap_pyfunction!(output_text_from_items_json, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_tool_output_json, module)?)?;
    module.add_function(wrap_pyfunction!(tool_failure_output_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_wire_request_json, module)?)?;
    module.add_function(wrap_pyfunction!(generate_image_json, module)?)?;
    module.add_function(wrap_pyfunction!(generate_images_json, module)?)?;
    module.add_function(wrap_pyfunction!(generate_video_json, module)?)?;
    module.add_class::<ResponseSession>()?;
    module.add_class::<ImageStream>()?;
    Ok(())
}
