use lmx_core::{
    ImageRequest, ProviderRegistry, ResponseMachine, ResponseRequest, ToolOutput, VERSION,
    VideoRequest, build_message_item, execute_round, generate_image, generate_video,
    load_request_context, output_text_from_items,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::{Arc, Mutex};

fn napi_error(error: impl std::fmt::Display) -> Error {
    Error::from_reason(error.to_string())
}

#[napi]
pub fn version() -> String {
    VERSION.into()
}

#[napi]
pub fn provider_registry_json() -> String {
    ProviderRegistry::default().as_json().to_string()
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
pub fn build_wire_request_json(request_json: String) -> Result<String> {
    let request: ResponseRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
    let machine =
        ResponseMachine::new(&ProviderRegistry::default(), request).map_err(napi_error)?;
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
    machine: Arc<Mutex<Option<ResponseMachine>>>,
}

#[napi]
impl ResponseSession {
    #[napi(constructor)]
    pub fn new(request_json: String) -> Result<Self> {
        let request: ResponseRequest = serde_json::from_str(&request_json).map_err(napi_error)?;
        Ok(Self {
            machine: Arc::new(Mutex::new(Some(
                ResponseMachine::new(&ProviderRegistry::default(), request).map_err(napi_error)?,
            ))),
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
        let outcome = execute_round(&mut machine).await;
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
}
