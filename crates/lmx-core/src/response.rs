use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::{Error, ProviderRegistry, RequestContext, Result, endpoint_url};

pub type ResponseItem = Map<String, Value>;

pub fn build_message_item(role: &str, text: &str) -> Result<ResponseItem> {
    if !matches!(role, "user" | "assistant" | "system" | "developer") {
        return Err(Error::State(format!("unsupported message role: {role:?}")));
    }
    if text.trim().is_empty() {
        return Err(Error::State("message text must not be empty".into()));
    }
    Ok(Map::from_iter([
        ("type".into(), Value::String("message".into())),
        ("role".into(), Value::String(role.into())),
        ("content".into(), Value::String(text.into())),
    ]))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseRequest {
    pub input: Vec<ResponseItem>,
    pub context: RequestContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default = "default_verbosity")]
    pub text_verbosity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_format: Option<Value>,
}

fn default_verbosity() -> String {
    "low".into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireRequest {
    pub method: String,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoreEvent {
    TextDelta {
        delta: String,
    },
    OutputItem {
        output_index: usize,
        item: ResponseItem,
    },
    ImageGenerationPartial {
        partial_image_index: u8,
        partial_image_base64: String,
    },
    ToolCallStarted {
        name: String,
        call_id: String,
        arguments: Value,
    },
    ToolCallCompleted {
        name: String,
        call_id: String,
        result: Value,
    },
    Completed {
        output_items: Vec<ResponseItem>,
        output_text: String,
        tool_roundtrips: u8,
    },
    Failed {
        error: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseResult {
    pub provider: String,
    pub model: String,
    pub output_items: Vec<ResponseItem>,
    pub output_text: String,
    pub tool_roundtrips: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub name: String,
    pub call_id: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutput {
    pub call_id: String,
    pub result: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
}

/// Convert the language-neutral tool result envelope into the one Responses
/// input item the next provider round requires.  SDK adapters call this rather
/// than independently deciding how `json`, multimodal `content`, and ordinary
/// values are represented.
pub fn normalize_tool_output(call_id: impl Into<String>, value: Value) -> ToolOutput {
    let call_id = call_id.into();
    let (result, content) = match &value {
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("content") => (
            object.get("result").cloned().unwrap_or(Value::Null),
            object.get("content").cloned(),
        ),
        Value::Object(object) if object.get("type").and_then(Value::as_str) == Some("json") => {
            (object.get("result").cloned().unwrap_or(Value::Null), None)
        }
        _ => (value, None),
    };
    ToolOutput {
        call_id,
        result,
        content,
    }
}

pub fn tool_failure_output(call_id: impl Into<String>, error: impl Into<String>) -> ToolOutput {
    normalize_tool_output(call_id, json!({"ok": false, "error": error.into()}))
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NextAction {
    ToolCalls { calls: Vec<ToolCall> },
    Completed { result: ResponseResult },
}

/// A binding-friendly frame emitted by a response session.  Events are sent
/// immediately; `ready` marks the point where a host may run tools or return a
/// completed result.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFrame {
    Event { event: CoreEvent },
    Ready { next: NextAction },
    Failed { error: String },
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RoundResult {
    pub events: Vec<CoreEvent>,
    pub next: NextAction,
}

/// A provider-neutral state machine. It owns all request construction and
/// response normalization; bindings only execute user-supplied tools between rounds.
pub struct ResponseMachine {
    request: ResponseRequest,
    model: String,
    base_url: String,
    running_input: Vec<ResponseItem>,
    accumulated_items: Vec<ResponseItem>,
    round_items: BTreeMap<usize, ResponseItem>,
    round_completed: bool,
    tool_roundtrips: u8,
    completed: bool,
}

impl ResponseMachine {
    pub fn new(registry: &ProviderRegistry, request: ResponseRequest) -> Result<Self> {
        let spec = registry.get(&request.context.provider)?;
        validate_request_context(&request.context)?;
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| spec.default_model.clone());
        if !spec
            .available_models
            .iter()
            .any(|candidate| candidate == &model)
        {
            return Err(Error::UnsupportedModel {
                provider: spec.provider.as_str().into(),
                model,
            });
        }
        if !request.tools.is_empty() && !spec.capabilities.supports_tools {
            return Err(Error::UnsupportedCapability(
                spec.provider.as_str().into(),
                "tools",
            ));
        }
        if request.text_format.is_some() && !spec.capabilities.supports_structured_output {
            return Err(Error::UnsupportedCapability(
                spec.provider.as_str().into(),
                "native structured output",
            ));
        }
        let base_url = if request.context.provider == crate::Provider::Bedrock {
            String::new()
        } else {
            request.context.resolved_base_url(spec)?
        };
        Ok(Self {
            base_url,
            running_input: request.input.clone(),
            request,
            model,
            accumulated_items: Vec::new(),
            round_items: BTreeMap::new(),
            round_completed: false,
            tool_roundtrips: 0,
            completed: false,
        })
    }

    pub fn wire_request(&self) -> Result<WireRequest> {
        if self.completed {
            return Err(Error::State("response is already complete".into()));
        }
        if self.request.context.provider == crate::Provider::Bedrock {
            return Err(Error::State(
                "bedrock provider uses its own transport; wire_request is not applicable".into(),
            ));
        }
        let mut headers = self.request.context.headers.clone();
        headers.insert(
            "Authorization".into(),
            format!("Bearer {}", self.request.context.api_key),
        );
        headers
            .entry("Content-Type".into())
            .or_insert_with(|| "application/json".into());
        headers
            .entry("Accept".into())
            .or_insert_with(|| "text/event-stream".into());
        headers
            .entry("User-Agent".into())
            .or_insert_with(|| format!("lmx/{}", crate::VERSION));
        if self.uses_codex_responses_lite() {
            headers.insert(
                "X-OpenAI-Internal-Codex-Responses-Lite".into(),
                "true".into(),
            );
        }
        let text = match &self.request.text_format {
            Some(format) => json!({"verbosity": self.request.text_verbosity, "format": format}),
            None => json!({"verbosity": self.request.text_verbosity}),
        };
        let input = self
            .running_input
            .iter()
            .map(openai_input_item)
            .collect::<Vec<_>>();
        let mut body = json!({
            "model": self.model,
            "input": input,
            "tool_choice": self.request.tool_choice.clone().unwrap_or_else(|| json!("auto")),
            "parallel_tool_calls": false,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "text": text,
            "stream": true,
        });
        if self.uses_codex_responses_lite() {
            let mut input = vec![json!({
                "type": "additional_tools",
                "role": "developer",
                "tools": self.request.tools,
            })];
            if !self.request.instructions.is_empty() {
                input.push(json!({
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": self.request.instructions}],
                }));
            }
            input.extend(body["input"].as_array().cloned().unwrap_or_default());
            body["input"] = Value::Array(input);
            let mut reasoning = json!({"summary": null, "context": "all_turns"});
            if let Some(effort) = &self.request.reasoning_effort {
                reasoning["effort"] = json!(effort);
            }
            body["reasoning"] = reasoning;
        } else {
            body["instructions"] = json!(self.request.instructions);
            body["tools"] = Value::Array(self.request.tools.clone());
        }
        if !self.uses_codex_responses_lite()
            && let Some(effort) = &self.request.reasoning_effort
        {
            body["reasoning"] = json!({"effort": effort, "summary": null});
        }
        Ok(WireRequest {
            method: "POST".into(),
            url: endpoint_url(&self.base_url, "responses", &self.request.context.query)?,
            headers,
            body,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn uses_codex_responses_lite(&self) -> bool {
        self.request.context.provider == crate::Provider::Codex
            && self.model.starts_with("gpt-5.6-")
    }

    fn begin_round(&mut self) -> Result<()> {
        if self.completed {
            return Err(Error::State("response is already complete".into()));
        }
        if !self.round_items.is_empty() {
            return Err(Error::State(
                "response round has unprocessed output items".into(),
            ));
        }
        self.round_completed = false;
        Ok(())
    }

    /// Feed one decoded Responses API SSE event into the engine.
    pub fn ingest(&mut self, event: &Value) -> Result<Vec<CoreEvent>> {
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "response.output_text.delta" => match event.get("delta").and_then(Value::as_str) {
                Some(delta) if !delta.is_empty() => Ok(vec![CoreEvent::TextDelta {
                    delta: delta.into(),
                }]),
                _ => Ok(Vec::new()),
            },
            "response.output_item.done" => {
                let output_index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let item = event
                    .get("item")
                    .and_then(Value::as_object)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Event("response.output_item.done is missing an object item".into())
                    })?;
                self.round_items.insert(output_index, item.clone());
                Ok(vec![CoreEvent::OutputItem { output_index, item }])
            }
            "response.image_generation_call.partial_image" => {
                let partial_image_index = event
                    .get("partial_image_index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        Error::Event(
                            "image generation partial event did not include partial_image_index"
                                .into(),
                        )
                    })?
                    .try_into()
                    .map_err(|_| Error::Event("partial image index exceeds u8".into()))?;
                let partial_image_base64 = event
                    .get("partial_image_b64")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        Error::Event(
                            "image generation partial event did not include partial_image_b64"
                                .into(),
                        )
                    })?
                    .to_owned();
                Ok(vec![CoreEvent::ImageGenerationPartial {
                    partial_image_index,
                    partial_image_base64,
                }])
            }
            "response.completed" => {
                self.round_completed = true;
                Ok(Vec::new())
            }
            "response.incomplete" => {
                Err(Error::Event(format!("request ended incomplete: {event}")))
            }
            "response.failed" => Err(Error::Event(format!(
                "request failed during stream: {event}"
            ))),
            _ => Ok(Vec::new()),
        }
    }

    /// Call exactly once after a completed stream event.
    pub fn finish_round(&mut self) -> Result<NextAction> {
        if self.completed {
            return Err(Error::State("response is already complete".into()));
        }
        if !self.round_completed {
            return Err(Error::State(
                "request ended without a completed stream event".into(),
            ));
        }
        let items = std::mem::take(&mut self.round_items)
            .into_values()
            .collect::<Vec<_>>();
        self.accumulated_items.extend(items.clone());
        self.running_input.extend(items.clone());
        let calls = items
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .map(parse_tool_call)
            .collect::<Result<Vec<_>>>()?;
        if calls.is_empty() {
            self.completed = true;
            return Ok(NextAction::Completed {
                result: ResponseResult {
                    provider: self.request.context.provider.as_str().into(),
                    model: self.model.clone(),
                    output_text: output_text_from_items(&self.accumulated_items),
                    output_items: self.accumulated_items.clone(),
                    tool_roundtrips: self.tool_roundtrips,
                },
            });
        }
        self.tool_roundtrips += 1;
        Ok(NextAction::ToolCalls { calls })
    }

    pub fn submit_tool_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = ToolOutput>,
    ) -> Result<Vec<CoreEvent>> {
        if self.completed {
            return Err(Error::State("response is already complete".into()));
        }
        let mut events = Vec::new();
        for output in outputs {
            let item = function_call_output_item(&output);
            self.accumulated_items.push(item.clone());
            self.running_input.push(item);
            events.push(CoreEvent::ToolCallCompleted {
                name: String::new(),
                call_id: output.call_id,
                result: output.result,
            });
        }
        Ok(events)
    }
}

fn validate_request_context(context: &RequestContext) -> Result<()> {
    if context.provider == crate::Provider::Bedrock {
        if context.headers.get("x-bedrock-access-key").is_none_or(|k| k.is_empty()) {
            return Err(Error::State(
                "provider 'bedrock' requires AWS credentials in its request context".into(),
            ));
        }
        return Ok(());
    }
    if context.provider != crate::Provider::Codex {
        return Ok(());
    }
    if context.api_key.trim().is_empty() {
        return Err(Error::State(
            "provider 'codex' requires a non-empty request-scoped access token".into(),
        ));
    }
    if context
        .headers
        .get("ChatGPT-Account-ID")
        .is_none_or(|account_id| account_id.trim().is_empty())
    {
        return Err(Error::State(
            "provider 'codex' requires a non-empty ChatGPT-Account-ID header in its request-scoped context".into(),
        ));
    }
    if let Some(base_url) = &context.base_url
        && base_url.trim_end_matches('/') != "https://chatgpt.com/backend-api/codex"
    {
        return Err(Error::State(
            "provider 'codex' only permits the https://chatgpt.com/backend-api/codex base URL"
                .into(),
        ));
    }
    Ok(())
}

/// Execute one provider round. The caller may execute returned tool calls in
/// its host language, submit their outputs, and invoke this again; all request
/// construction and state transitions remain in this core.
pub async fn execute_round(machine: &mut ResponseMachine) -> Result<RoundResult> {
    let cancellation = CancellationToken::new();
    execute_round_with_cancellation(machine, &cancellation).await
}

/// Execute one provider round with cooperative cancellation.
pub async fn execute_round_with_cancellation(
    machine: &mut ResponseMachine,
    cancellation: &CancellationToken,
) -> Result<RoundResult> {
    let mut events = Vec::new();
    let next = execute_round_with_observer(machine, cancellation, |event| {
        events.push(event);
        Ok(())
    })
    .await?;
    Ok(RoundResult { events, next })
}

/// Execute one provider round and emit normalized events as the SSE response
/// arrives. Bindings may choose to buffer these events for their API boundary,
/// but transport and core state transition processing are never buffered.
pub async fn execute_round_with_observer<F>(
    machine: &mut ResponseMachine,
    cancellation: &CancellationToken,
    mut observer: F,
) -> Result<NextAction>
where
    F: FnMut(CoreEvent) -> Result<()>,
{
    machine.begin_round()?;
    if machine.request.context.provider == crate::Provider::Bedrock {
        bedrock_stream_and_observe(machine, cancellation, &mut observer).await?;
    } else {
        let transport = crate::OpenAiTransport::new();
        stream_and_observe(&transport, machine, cancellation, &mut observer).await?;
    }
    machine.finish_round()
}

async fn stream_and_observe<F>(
    transport: &crate::OpenAiTransport,
    machine: &mut ResponseMachine,
    cancellation: &CancellationToken,
    observer: &mut F,
) -> Result<()>
where
    F: FnMut(CoreEvent) -> Result<()>,
{
    let wire = machine.wire_request()?;
    transport
        .stream_round(wire, cancellation, |raw| {
            for event in machine.ingest(&raw)? {
                observer(event)?;
            }
            Ok(())
        })
        .await
}

async fn bedrock_stream_and_observe<F>(
    machine: &mut ResponseMachine,
    cancellation: &CancellationToken,
    observer: &mut F,
) -> Result<()>
where
    F: FnMut(CoreEvent) -> Result<()>,
{
    use crate::bedrock::{
        BedrockCredentials, BedrockRequest, BedrockTransport, build_converse_messages,
        convert_tools,
    };

    let credentials = BedrockCredentials::from_context(&machine.request.context)?;
    let (system, messages) =
        build_converse_messages(&machine.running_input, &machine.request.instructions);
    let tools = convert_tools(&machine.request.tools);
    let tool_choice = machine.request.tool_choice.as_ref().and_then(|tc| {
        match tc.as_str() {
            Some("auto") | None => Some(serde_json::json!({"auto": {}})),
            Some("required") => Some(serde_json::json!({"any": {}})),
            Some("none") => None,
            _ => Some(tc.clone()),
        }
    });
    let reasoning = machine.request.reasoning_effort.as_ref().map(|effort| {
        serde_json::json!({"enabled": true, "budget_tokens": effort_to_budget(effort)})
    });

    let bedrock_request = BedrockRequest {
        model: machine.model.clone(),
        system,
        messages,
        tools,
        tool_choice,
        reasoning,
    };

    let transport = BedrockTransport::new();
    let body = bedrock_request.body();
    let model = machine.model.clone();

    let mut text_buffer = String::new();

    let completed_tools = transport
        .converse_stream(&model, body, &credentials, cancellation, |core_event| {
            match &core_event {
                CoreEvent::TextDelta { delta } if !delta.is_empty() => {
                    text_buffer.push_str(delta);
                }
                _ => {}
            }
            observer(core_event)
        })
        .await?;

    let mut output_index: usize = 0;
    if !text_buffer.is_empty() {
        let item = Map::from_iter([
            ("type".into(), Value::String("message".into())),
            ("role".into(), Value::String("assistant".into())),
            ("content".into(), Value::String(text_buffer)),
        ]);
        machine.round_items.insert(output_index, item);
        output_index += 1;
    }
    for tc in &completed_tools {
        let item = Map::from_iter([
            ("type".into(), Value::String("function_call".into())),
            ("name".into(), Value::String(tc.name.clone())),
            ("call_id".into(), Value::String(tc.call_id.clone())),
            (
                "arguments".into(),
                Value::String(if tc.arguments_json.is_empty() {
                    "{}".into()
                } else {
                    tc.arguments_json.clone()
                }),
            ),
        ]);
        machine.round_items.insert(output_index, item);
        output_index += 1;
    }

    machine.round_completed = true;
    Ok(())
}

fn effort_to_budget(effort: &str) -> u32 {
    match effort {
        "low" => 1024,
        "medium" => 4096,
        "high" => 16384,
        _ => 8192,
    }
}

fn openai_input_item(item: &ResponseItem) -> Value {
    let allowed: &[&str] = match item.get("type").and_then(Value::as_str) {
        Some("message") => &["type", "role", "content"],
        Some("function_call") => &["type", "call_id", "name", "arguments"],
        Some("function_call_output") => &["type", "call_id", "output"],
        _ => {
            return Value::Object(
                item.iter()
                    .filter(|(key, _)| key.as_str() != "id" && key.as_str() != "status")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            );
        }
    };
    Value::Object(
        allowed
            .iter()
            .filter_map(|key| {
                item.get(*key)
                    .map(|value| ((*key).to_owned(), value.clone()))
            })
            .collect(),
    )
}

fn parse_tool_call(item: &ResponseItem) -> Result<ToolCall> {
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let arguments = match item.get("arguments") {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(raw)) if raw.is_empty() => json!({}),
        Some(Value::String(raw)) => serde_json::from_str(raw)
            .map_err(|error| Error::FunctionArguments(error.to_string()))?,
        _ => {
            return Err(Error::FunctionArguments(
                "function_call arguments must be JSON text".into(),
            ));
        }
    };
    if !arguments.is_object() {
        return Err(Error::FunctionArguments(
            "function_call arguments must decode to a JSON object".into(),
        ));
    }
    Ok(ToolCall {
        name,
        call_id,
        arguments,
    })
}

fn function_call_output_item(output: &ToolOutput) -> ResponseItem {
    let content = output
        .content
        .clone()
        .unwrap_or_else(|| Value::String(output.result.to_string()));
    Map::from_iter([
        (
            String::from("type"),
            Value::String("function_call_output".into()),
        ),
        (
            String::from("call_id"),
            Value::String(output.call_id.clone()),
        ),
        (String::from("output"), content),
    ])
}

pub fn output_text_from_items(items: &[ResponseItem]) -> String {
    items
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some("message")
                && item.get("role").and_then(Value::as_str) == Some("assistant")
        })
        .filter_map(|item| match item.get("content") {
            Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
            Some(Value::Array(parts)) => {
                let text = parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                (!text.is_empty()).then_some(text)
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderRegistry;
    use serde_json::json;

    fn request() -> ResponseRequest {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai-response-request.json"
        ))
        .unwrap()
    }

    #[test]
    fn builds_the_shared_wire_payload() {
        let machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        let wire = machine.wire_request().unwrap();
        assert_eq!(wire.url, "https://api.openai.com/v1/responses");
        assert_eq!(wire.body["input"][0]["content"], "Hello");
        assert_eq!(wire.body["instructions"], "Answer concisely.");
        assert!(wire.body["stream"].as_bool().unwrap());
    }

    #[test]
    fn builds_a_direct_codex_request_from_request_scoped_credentials() {
        let request = ResponseRequest {
            input: vec![build_message_item("user", "Hello").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-sol".into()),
            instructions: String::new(),
            tools: vec![],
            tool_choice: None,
            reasoning_effort: None,
            text_verbosity: "low".into(),
            text_format: None,
        };
        let wire = ResponseMachine::new(&ProviderRegistry::default(), request)
            .unwrap()
            .wire_request()
            .unwrap();
        assert_eq!(wire.url, "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(wire.headers["Authorization"], "Bearer request-token");
        assert_eq!(wire.headers["ChatGPT-Account-ID"], "account-123");
        assert_eq!(
            wire.headers["X-OpenAI-Internal-Codex-Responses-Lite"],
            "true"
        );
        assert_eq!(wire.headers["Accept"], "text/event-stream");
        assert!(wire.body.get("instructions").is_none());
        assert!(wire.body.get("tools").is_none());
        assert_eq!(wire.body["reasoning"]["context"], "all_turns");
        assert_eq!(wire.body["input"][0]["type"], "additional_tools");
        assert_eq!(wire.body["input"][0]["role"], "developer");
        assert_eq!(wire.body["input"][1]["role"], "user");
    }

    #[test]
    fn preserves_an_explicit_tool_choice() {
        let mut request = request();
        request.tools = vec![json!({"type": "image_generation", "action": "generate"})];
        request.tool_choice = Some(json!("required"));
        let wire = ResponseMachine::new(&ProviderRegistry::default(), request)
            .unwrap()
            .wire_request()
            .unwrap();
        assert_eq!(wire.body["tool_choice"], "required");
    }

    #[test]
    fn emits_image_generation_partials() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        let events = machine
            .ingest(&json!({
                "type": "response.image_generation_call.partial_image",
                "partial_image_index": 2,
                "partial_image_b64": "cHJldmlldw==",
            }))
            .unwrap();
        assert!(matches!(
            events.as_slice(),
            [CoreEvent::ImageGenerationPartial {
                partial_image_index: 2,
                partial_image_base64,
            }] if partial_image_base64 == "cHJldmlldw=="
        ));
    }

    #[test]
    fn rejects_codex_context_without_an_account_header() {
        let mut request = request();
        request.context.provider = crate::Provider::Codex;
        request.context.api_key = "request-token".into();
        request.context.headers.clear();
        assert!(matches!(
            ResponseMachine::new(&ProviderRegistry::default(), request),
            Err(Error::State(message)) if message.contains("ChatGPT-Account-ID")
        ));
    }

    #[test]
    fn normalizes_a_completed_round() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        let events = machine.ingest(&json!({"type":"response.output_item.done", "output_index":0, "item":{"type":"message", "role":"assistant", "content":"Hi"}})).unwrap();
        assert!(matches!(events[0], CoreEvent::OutputItem { .. }));
        machine
            .ingest(&json!({"type":"response.completed"}))
            .unwrap();
        let NextAction::Completed { result } = machine.finish_round().unwrap() else {
            panic!("expected completed")
        };
        assert_eq!(result.output_text, "Hi");
    }

    #[test]
    fn rejects_a_round_without_a_terminal_completed_event() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        machine
            .ingest(&json!({"type":"response.output_item.done", "output_index":0, "item":{"type":"message", "role":"assistant", "content":"partial"}}))
            .unwrap();
        assert!(
            matches!(machine.finish_round(), Err(Error::State(message)) if message.contains("completed stream event"))
        );
    }

    #[test]
    fn permits_more_than_the_previous_tool_roundtrip_limit() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        for index in 0..9 {
            machine
                .ingest(&json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "name": "continue",
                        "call_id": format!("call_{index}"),
                        "arguments": "{}"
                    }
                }))
                .unwrap();
            machine
                .ingest(&json!({"type": "response.completed"}))
                .unwrap();
            assert!(matches!(
                machine.finish_round().unwrap(),
                NextAction::ToolCalls { .. }
            ));
            machine
                .submit_tool_outputs([ToolOutput {
                    call_id: format!("call_{index}"),
                    result: json!({"ok": true}),
                    content: None,
                }])
                .unwrap();
        }
        assert_eq!(machine.tool_roundtrips, 9);
    }
}
