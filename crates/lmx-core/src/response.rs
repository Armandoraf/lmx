use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

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
    pub reasoning_effort: Option<String>,
    #[serde(default = "default_verbosity")]
    pub text_verbosity: String,
    #[serde(default = "default_max_roundtrips")]
    pub max_tool_roundtrips: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_format: Option<Value>,
}

fn default_verbosity() -> String {
    "low".into()
}
fn default_max_roundtrips() -> u8 {
    8
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
        Ok(Self {
            base_url: request.context.resolved_base_url(spec)?,
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
        let mut headers = self.request.context.headers.clone();
        headers.insert(
            "Authorization".into(),
            format!("Bearer {}", self.request.context.api_key),
        );
        headers
            .entry("Content-Type".into())
            .or_insert_with(|| "application/json".into());
        headers
            .entry("User-Agent".into())
            .or_insert_with(|| format!("lmx/{}", crate::VERSION));
        let text = match &self.request.text_format {
            Some(format) => json!({"verbosity": self.request.text_verbosity, "format": format}),
            None => json!({"verbosity": self.request.text_verbosity}),
        };
        let mut body = json!({
            "model": self.model,
            "instructions": self.request.instructions,
            "input": self.running_input.iter().map(openai_input_item).collect::<Vec<_>>(),
            "tools": self.request.tools,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "text": text,
            "stream": true,
        });
        if let Some(effort) = &self.request.reasoning_effort {
            body["reasoning"] = json!({"effort": effort, "summary": null});
        }
        Ok(WireRequest {
            method: "POST".into(),
            url: endpoint_url(&self.base_url, "responses", &self.request.context.query)?,
            headers,
            body,
        })
    }

    pub(crate) fn is_codex(&self) -> bool {
        self.request.context.provider == crate::Provider::Codex
    }

    pub(crate) fn replace_context(&mut self, context: RequestContext) {
        self.base_url = context
            .base_url
            .clone()
            .unwrap_or_else(|| self.base_url.clone());
        self.request.context = context;
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
        if self.tool_roundtrips >= self.request.max_tool_roundtrips {
            return Err(Error::State(format!(
                "model exceeded max tool round-trips ({}) in one response",
                self.request.max_tool_roundtrips
            )));
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

/// Execute one provider round. The caller may execute returned tool calls in
/// its host language, submit their outputs, and invoke this again; all request
/// construction and state transitions remain in this core.
pub async fn execute_round(machine: &mut ResponseMachine) -> Result<RoundResult> {
    let mut events = Vec::new();
    let next = execute_round_with_observer(machine, |event| events.push(event)).await?;
    Ok(RoundResult { events, next })
}

/// Execute one provider round and emit normalized events as the SSE response
/// arrives. Bindings may choose to buffer these events for their API boundary,
/// but transport and core state transition processing are never buffered.
pub async fn execute_round_with_observer<F>(
    machine: &mut ResponseMachine,
    mut observer: F,
) -> Result<NextAction>
where
    F: FnMut(CoreEvent),
{
    let transport = crate::OpenAiTransport::new();
    machine.begin_round()?;
    match stream_and_observe(&transport, machine, &mut observer).await {
        Ok(()) => {}
        Err(crate::Error::HttpStatus { status: 401, .. }) if machine.is_codex() => {
            machine.replace_context(crate::refresh_codex_auth().await?);
            machine.begin_round()?;
            stream_and_observe(&transport, machine, &mut observer).await?;
        }
        Err(error) => return Err(error),
    }
    machine.finish_round()
}

async fn stream_and_observe<F>(
    transport: &crate::OpenAiTransport,
    machine: &mut ResponseMachine,
    observer: &mut F,
) -> Result<()>
where
    F: FnMut(CoreEvent),
{
    let wire = machine.wire_request()?;
    transport
        .stream_round(wire, |raw| {
            for event in machine.ingest(&raw)? {
                observer(event);
            }
            Ok(())
        })
        .await
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
}
