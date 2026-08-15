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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_protocol: Option<CodexProtocol>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagement>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CodexProtocol {
    ResponsesLite,
    ResponsesStandard,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResponseUsage {
    pub input_tokens: u64,
    #[serde(default)]
    pub cached_input_tokens: u64,
    #[serde(default)]
    pub cache_write_input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "mode",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum ContextManagement {
    RemoteV2 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_usage: Option<ResponseUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        previous_usage_input_item_count: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        auto_compact_token_limit: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retained_message_token_budget: Option<u64>,
    },
    Server {
        compact_threshold: u64,
    },
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
        output_item: ResponseItem,
    },
    ContextCompacted {
        items: Vec<ResponseItem>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<ResponseUsage>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    pub history_update: InferenceHistoryUpdate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceHistoryUpdate {
    Append { items: Vec<ResponseItem> },
    Replace { items: Vec<ResponseItem> },
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
    ToolCalls {
        calls: Vec<ToolCall>,
    },
    Compacted {
        items: Vec<ResponseItem>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<ResponseUsage>,
    },
    Completed {
        result: ResponseResult,
    },
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
    response_items: Vec<ResponseItem>,
    round_items: BTreeMap<usize, ResponseItem>,
    round_completed: bool,
    round_kind: RoundKind,
    round_response_id: Option<String>,
    round_usage: Option<ResponseUsage>,
    round_server_compaction_index: Option<usize>,
    last_response_id: Option<String>,
    last_usage: Option<ResponseUsage>,
    last_usage_input_item_count: Option<usize>,
    history_replaced: bool,
    tool_roundtrips: u8,
    completed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RoundKind {
    #[default]
    Response,
    Compaction,
}

const CODEX_GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT: u64 = 244_800;
const CODEX_RETAINED_MESSAGE_TOKEN_BUDGET: u64 = 64_000;

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
        if request.codex_protocol.is_some()
            && !(request.context.provider == crate::Provider::Codex
                && model.starts_with("gpt-5.6-"))
        {
            return Err(Error::UnsupportedCapability(
                spec.provider.as_str().into(),
                "GPT-5.6 Codex Responses protocol selection",
            ));
        }
        let codex_protocol = request
            .codex_protocol
            .unwrap_or(CodexProtocol::ResponsesLite);
        if let Some(management) = &request.context_management {
            if matches!(
                management,
                ContextManagement::Server {
                    compact_threshold: 0
                }
            ) {
                return Err(Error::State(
                    "server compact threshold must be greater than zero".into(),
                ));
            }
            let supported = request.context.provider == crate::Provider::Codex
                && model.starts_with("gpt-5.6-")
                && matches!(
                    (codex_protocol, management),
                    (
                        CodexProtocol::ResponsesLite,
                        ContextManagement::RemoteV2 { .. }
                    ) | (
                        CodexProtocol::ResponsesStandard,
                        ContextManagement::Server { .. }
                    )
                );
            if !supported {
                return Err(Error::UnsupportedCapability(
                    spec.provider.as_str().into(),
                    "selected Codex context-management protocol",
                ));
            }
        }
        let base_url = if request.context.provider == crate::Provider::Bedrock {
            String::new()
        } else {
            request.context.resolved_base_url(spec)?
        };
        let previous_usage = match request.context_management.as_ref() {
            Some(ContextManagement::RemoteV2 {
                previous_usage,
                previous_usage_input_item_count,
                ..
            }) => (previous_usage.clone(), *previous_usage_input_item_count),
            _ => (None, None),
        };
        let (last_usage, last_usage_input_item_count) = match previous_usage {
            (Some(usage), Some(item_count)) if item_count <= request.input.len() => {
                (Some(usage), Some(item_count))
            }
            (None, None) => (None, None),
            _ => {
                return Err(Error::State(
                    "previous usage requires its valid input item count checkpoint".into(),
                ));
            }
        };
        Ok(Self {
            base_url,
            running_input: request.input.clone(),
            request,
            model,
            accumulated_items: Vec::new(),
            response_items: Vec::new(),
            round_items: BTreeMap::new(),
            round_completed: false,
            round_kind: RoundKind::Response,
            round_response_id: None,
            round_usage: None,
            round_server_compaction_index: None,
            last_response_id: None,
            last_usage,
            last_usage_input_item_count,
            history_replaced: false,
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
        headers
            .retain(|name, _| !name.eq_ignore_ascii_case("X-OpenAI-Internal-Codex-Responses-Lite"));
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
        let mut input = self
            .running_input
            .iter()
            .map(openai_input_item)
            .collect::<Vec<_>>();
        if self.round_kind == RoundKind::Compaction {
            input.push(json!({"type": "compaction_trigger"}));
            let existing_feature_header = headers
                .keys()
                .find(|name| name.eq_ignore_ascii_case("x-codex-beta-features"))
                .cloned();
            let features = existing_feature_header
                .as_ref()
                .and_then(|name| headers.get(name))
                .map(String::as_str)
                .unwrap_or_default();
            let features = features
                .split(',')
                .map(str::trim)
                .filter(|feature| !feature.is_empty())
                .chain(std::iter::once("remote_compaction_v2"))
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
                .join(",");
            if let Some(name) = existing_feature_header {
                headers.insert(name, features);
            } else {
                headers.insert("x-codex-beta-features".into(), features);
            }
        }
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
        } else if self.uses_codex_responses_standard() {
            body["instructions"] = json!(self.request.instructions);
            body["tools"] = Value::Array(self.request.tools.clone());
            let mut reasoning = json!({"summary": null, "context": "all_turns"});
            if let Some(effort) = &self.request.reasoning_effort {
                reasoning["effort"] = json!(effort);
            }
            body["reasoning"] = reasoning;
            if let Some(ContextManagement::Server { compact_threshold }) =
                &self.request.context_management
            {
                body["context_management"] = json!([{
                    "type": "compaction",
                    "compact_threshold": compact_threshold,
                }]);
            }
        } else {
            body["instructions"] = json!(self.request.instructions);
            body["tools"] = Value::Array(self.request.tools.clone());
        }
        if !self.uses_codex_responses_lite()
            && !self.uses_codex_responses_standard()
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
            && self
                .request
                .codex_protocol
                .unwrap_or(CodexProtocol::ResponsesLite)
                == CodexProtocol::ResponsesLite
    }

    fn uses_codex_responses_standard(&self) -> bool {
        self.request.context.provider == crate::Provider::Codex
            && self.model.starts_with("gpt-5.6-")
            && self.request.codex_protocol == Some(CodexProtocol::ResponsesStandard)
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
        self.round_response_id = None;
        self.round_usage = None;
        self.round_server_compaction_index = None;
        self.round_kind = if self.should_compact() {
            RoundKind::Compaction
        } else {
            RoundKind::Response
        };
        Ok(())
    }

    fn should_compact(&self) -> bool {
        let Some(ContextManagement::RemoteV2 {
            auto_compact_token_limit,
            ..
        }) = &self.request.context_management
        else {
            return false;
        };
        if !self.uses_codex_responses_lite() {
            return false;
        }
        let limit = auto_compact_token_limit.unwrap_or(CODEX_GPT_5_6_AUTO_COMPACT_TOKEN_LIMIT);
        let estimated = match (&self.last_usage, self.last_usage_input_item_count) {
            (Some(usage), Some(item_count)) => {
                usage.total_tokens + estimate_input_tokens(&self.running_input[item_count..])
            }
            _ => estimate_input_tokens(&self.running_input),
        };
        estimated >= limit
    }

    /// Feed one decoded Responses API SSE event into the engine.
    pub fn ingest(&mut self, event: &Value) -> Result<Vec<CoreEvent>> {
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "response.output_text.delta" if self.round_kind == RoundKind::Compaction => {
                Ok(Vec::new())
            }
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
                if self.round_kind == RoundKind::Compaction {
                    Ok(Vec::new())
                } else if self.uses_codex_responses_standard()
                    && item.get("type").and_then(Value::as_str) == Some("compaction")
                {
                    validate_compaction_item(&item)?;
                    self.round_server_compaction_index = Some(output_index);
                    Ok(vec![CoreEvent::ContextCompacted {
                        items: vec![item],
                        usage: None,
                    }])
                } else {
                    Ok(vec![CoreEvent::OutputItem { output_index, item }])
                }
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
                let response = event.get("response").unwrap_or(event);
                self.round_response_id = response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.round_usage = response.get("usage").and_then(parse_response_usage);
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
        if self.round_kind == RoundKind::Compaction {
            let [compaction] = items.as_slice() else {
                return Err(Error::Event(format!(
                    "remote compaction v2 returned {} output items instead of exactly one",
                    items.len()
                )));
            };
            validate_compaction_item(compaction)?;
            let retained_budget = match self.request.context_management.as_ref() {
                Some(ContextManagement::RemoteV2 {
                    retained_message_token_budget,
                    ..
                }) => retained_message_token_budget.unwrap_or(CODEX_RETAINED_MESSAGE_TOKEN_BUDGET),
                _ => CODEX_RETAINED_MESSAGE_TOKEN_BUDGET,
            };
            let mut replacement = retain_compaction_messages(&self.running_input, retained_budget);
            replacement.push(compaction.clone());
            self.running_input = replacement.clone();
            self.history_replaced = true;
            self.last_response_id = self.round_response_id.take();
            let usage = self.round_usage.take();
            // The compaction request reports usage for the superseded context.
            // Recompute from the replacement until the next normal response gives
            // the authoritative active-context usage.
            self.last_usage = None;
            self.last_usage_input_item_count = None;
            return Ok(NextAction::Compacted {
                items: replacement,
                usage,
            });
        }
        self.last_response_id = self.round_response_id.take();
        self.response_items.extend(
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) != Some("compaction"))
                .cloned(),
        );
        if self.round_server_compaction_index.take().is_some() {
            let Some(compaction_index) = items
                .iter()
                .rposition(|item| item.get("type").and_then(Value::as_str) == Some("compaction"))
            else {
                return Err(Error::State(
                    "server compaction index did not resolve to a compaction item".into(),
                ));
            };
            let replacement = items[compaction_index..].to_vec();
            self.accumulated_items = replacement.clone();
            self.running_input = replacement;
            self.history_replaced = true;
        } else {
            self.accumulated_items.extend(items.clone());
            self.running_input.extend(items.clone());
        }
        self.last_usage = self.round_usage.take();
        self.last_usage_input_item_count =
            self.last_usage.as_ref().map(|_| self.running_input.len());
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
                    output_text: output_text_from_items(&self.response_items),
                    output_items: self.response_items.clone(),
                    tool_roundtrips: self.tool_roundtrips,
                    response_id: self.last_response_id.clone(),
                    usage: self.last_usage.clone(),
                    history_update: if self.history_replaced {
                        InferenceHistoryUpdate::Replace {
                            items: self.running_input.clone(),
                        }
                    } else {
                        InferenceHistoryUpdate::Append {
                            items: self.accumulated_items.clone(),
                        }
                    },
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
            self.response_items.push(item.clone());
            self.running_input.push(item.clone());
            events.push(CoreEvent::ToolCallCompleted {
                name: String::new(),
                call_id: output.call_id,
                result: output.result,
                output_item: item,
            });
        }
        Ok(events)
    }
}

fn validate_compaction_item(item: &ResponseItem) -> Result<()> {
    if item.get("type").and_then(Value::as_str) != Some("compaction")
        || item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        return Err(Error::Event(
            "compaction response did not contain encrypted context".into(),
        ));
    }
    Ok(())
}

fn parse_response_usage(value: &Value) -> Option<ResponseUsage> {
    let input_tokens = value.get("input_tokens")?.as_u64()?;
    let output_tokens = value.get("output_tokens")?.as_u64()?;
    let total_tokens = value.get("total_tokens")?.as_u64()?;
    let input_details = value.get("input_tokens_details");
    Some(ResponseUsage {
        input_tokens,
        cached_input_tokens: input_details
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        cache_write_input_tokens: input_details
            .and_then(|details| details.get("cache_write_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        output_tokens,
        total_tokens,
    })
}

fn estimate_input_tokens(items: &[ResponseItem]) -> u64 {
    items
        .iter()
        .map(|item| estimate_value_tokens(&Value::Object(item.clone()), None))
        .sum()
}

fn estimate_value_tokens(value: &Value, key: Option<&str>) -> u64 {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => 1,
        Value::String(text) => {
            if matches!(key, Some("encrypted_content" | "image_url" | "id")) {
                0
            } else {
                (text.len() as u64).div_ceil(4).max(1)
            }
        }
        Value::Array(values) => values
            .iter()
            .map(|value| estimate_value_tokens(value, key))
            .sum(),
        Value::Object(object) => object
            .iter()
            .map(|(key, value)| estimate_value_tokens(value, Some(key)))
            .sum(),
    }
}

fn retain_compaction_messages(items: &[ResponseItem], budget: u64) -> Vec<ResponseItem> {
    let mut remaining = budget;
    let mut retained = Vec::new();
    for item in items.iter().rev() {
        let keep = item.get("type").and_then(Value::as_str) == Some("message")
            && matches!(
                item.get("role").and_then(Value::as_str),
                Some("user" | "developer" | "system")
            );
        if !keep {
            continue;
        }
        let tokens = estimate_input_tokens(std::slice::from_ref(item)).max(1);
        if tokens > remaining {
            break;
        }
        retained.push(item.clone());
        remaining -= tokens;
        if remaining == 0 {
            break;
        }
    }
    retained.reverse();
    retained
}

fn validate_request_context(context: &RequestContext) -> Result<()> {
    if context.provider == crate::Provider::Bedrock {
        if context
            .headers
            .get("x-bedrock-access-key")
            .is_none_or(|k| k.is_empty())
        {
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
    let tool_choice = machine
        .request
        .tool_choice
        .as_ref()
        .and_then(|tc| match tc.as_str() {
            Some("auto") | None => Some(serde_json::json!({"auto": {}})),
            Some("required") => Some(serde_json::json!({"any": {}})),
            Some("none") => None,
            _ => Some(tc.clone()),
        });
    let reasoning = machine.request.reasoning_effort.as_ref().map(
        |effort| serde_json::json!({"type": "enabled", "budget_tokens": effort_to_budget(effort)}),
    );

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
            codex_protocol: None,
            context_management: None,
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
    fn builds_a_standard_codex_request_with_server_compaction() {
        let request = ResponseRequest {
            input: vec![build_message_item("user", "Hello").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-luna".into()),
            instructions: "Answer concisely.".into(),
            tools: vec![json!({"type": "function", "name": "work"})],
            tool_choice: None,
            reasoning_effort: Some("low".into()),
            text_verbosity: "low".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesStandard),
            context_management: Some(ContextManagement::Server {
                compact_threshold: 244_800,
            }),
        };
        let wire = ResponseMachine::new(&ProviderRegistry::default(), request)
            .unwrap()
            .wire_request()
            .unwrap();
        assert!(
            !wire
                .headers
                .contains_key("X-OpenAI-Internal-Codex-Responses-Lite")
        );
        assert_eq!(wire.body["instructions"], "Answer concisely.");
        assert_eq!(wire.body["tools"][0]["name"], "work");
        assert_eq!(wire.body["input"][0]["role"], "user");
        assert_eq!(wire.body["reasoning"]["context"], "all_turns");
        assert_eq!(
            wire.body["context_management"],
            json!([{"type": "compaction", "compact_threshold": 244_800}])
        );
    }

    #[test]
    fn server_compaction_replaces_history_and_continues_through_tools() {
        let request = ResponseRequest {
            input: vec![build_message_item("user", "Use a tool").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-luna".into()),
            instructions: String::new(),
            tools: vec![json!({"type": "function", "name": "work"})],
            tool_choice: None,
            reasoning_effort: Some("low".into()),
            text_verbosity: "low".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesStandard),
            context_management: Some(ContextManagement::Server {
                compact_threshold: 1_000,
            }),
        };
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request).unwrap();
        let compact_events = machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "compaction", "encrypted_content": "opaque"}
            }))
            .unwrap();
        assert!(matches!(
            compact_events.as_slice(),
            [CoreEvent::ContextCompacted { items, usage: None }]
                if items[0]["encrypted_content"] == "opaque"
        ));
        machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "name": "work",
                    "call_id": "call_1",
                    "arguments": "{}"
                }
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_tool",
                    "usage": {"input_tokens": 1_100, "output_tokens": 10, "total_tokens": 1_110}
                }
            }))
            .unwrap();
        assert!(matches!(
            machine.finish_round().unwrap(),
            NextAction::ToolCalls { calls } if calls[0].call_id == "call_1"
        ));
        machine
            .submit_tool_outputs([ToolOutput {
                call_id: "call_1".into(),
                result: json!({"ok": true}),
                content: None,
            }])
            .unwrap();
        machine.begin_round().unwrap();
        let wire = machine.wire_request().unwrap();
        assert_eq!(wire.body["input"][0]["type"], "compaction");
        assert_eq!(wire.body["input"][1]["type"], "function_call");
        assert_eq!(wire.body["input"][2]["type"], "function_call_output");
        machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": "Done"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_done",
                    "usage": {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
                }
            }))
            .unwrap();
        let NextAction::Completed { result } = machine.finish_round().unwrap() else {
            panic!("expected completed")
        };
        assert!(matches!(
            result.history_update,
            InferenceHistoryUpdate::Replace { items }
                if items[0]["type"] == "compaction" && items.last().unwrap()["content"] == "Done"
        ));
    }

    #[test]
    fn server_compaction_after_a_message_preserves_response_output() {
        let request = ResponseRequest {
            input: vec![build_message_item("user", "Hello").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-luna".into()),
            instructions: String::new(),
            tools: vec![],
            tool_choice: None,
            reasoning_effort: Some("low".into()),
            text_verbosity: "low".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesStandard),
            context_management: Some(ContextManagement::Server {
                compact_threshold: 1_000,
            }),
        };
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request).unwrap();
        machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "content": "Visible reply"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {"type": "compaction", "encrypted_content": "opaque"}
            }))
            .unwrap();
        machine
            .ingest(&json!({"type": "response.completed"}))
            .unwrap();

        let NextAction::Completed { result } = machine.finish_round().unwrap() else {
            panic!("expected completed")
        };
        assert_eq!(result.output_text, "Visible reply");
        assert_eq!(result.output_items[0]["type"], "message");
        assert!(matches!(
            result.history_update,
            InferenceHistoryUpdate::Replace { items }
                if items.len() == 1 && items[0]["type"] == "compaction"
        ));
    }

    #[test]
    fn rejects_mismatched_codex_protocol_and_context_management() {
        let mut request = ResponseRequest {
            input: vec![build_message_item("user", "Hello").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-luna".into()),
            instructions: String::new(),
            tools: vec![],
            tool_choice: None,
            reasoning_effort: None,
            text_verbosity: "low".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesLite),
            context_management: Some(ContextManagement::Server {
                compact_threshold: 244_800,
            }),
        };
        assert!(matches!(
            ResponseMachine::new(&ProviderRegistry::default(), request.clone()),
            Err(Error::UnsupportedCapability(_, _))
        ));

        request.codex_protocol = Some(CodexProtocol::ResponsesStandard);
        request.context_management = Some(ContextManagement::RemoteV2 {
            previous_usage: None,
            previous_usage_input_item_count: None,
            auto_compact_token_limit: None,
            retained_message_token_budget: None,
        });
        assert!(matches!(
            ResponseMachine::new(&ProviderRegistry::default(), request),
            Err(Error::UnsupportedCapability(_, _))
        ));
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
    fn captures_response_identity_usage_and_append_history() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        machine
            .ingest(&json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"message", "role":"assistant", "content":"Hi"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type":"response.completed",
                "response": {
                    "id": "resp_123",
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {
                            "cached_tokens": 40,
                            "cache_write_tokens": 12
                        },
                        "output_tokens": 10,
                        "total_tokens": 110
                    }
                }
            }))
            .unwrap();
        let NextAction::Completed { result } = machine.finish_round().unwrap() else {
            panic!("expected completed")
        };
        assert_eq!(result.response_id.as_deref(), Some("resp_123"));
        assert_eq!(
            result.usage,
            Some(ResponseUsage {
                input_tokens: 100,
                cached_input_tokens: 40,
                cache_write_input_tokens: 12,
                output_tokens: 10,
                total_tokens: 110,
            })
        );
        assert!(matches!(
            result.history_update,
            InferenceHistoryUpdate::Append { items } if items.len() == 1
        ));
    }

    #[test]
    fn remote_compaction_v2_replaces_history_before_sampling() {
        let request = ResponseRequest {
            input: vec![
                build_message_item("user", "old request").unwrap(),
                build_message_item("assistant", "old answer").unwrap(),
                build_message_item("user", "new request").unwrap(),
            ],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-sol".into()),
            instructions: "Be helpful.".into(),
            tools: vec![],
            tool_choice: None,
            reasoning_effort: Some("medium".into()),
            text_verbosity: "medium".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesLite),
            context_management: Some(ContextManagement::RemoteV2 {
                previous_usage: Some(ResponseUsage {
                    input_tokens: 245_000,
                    cached_input_tokens: 200_000,
                    cache_write_input_tokens: 0,
                    output_tokens: 100,
                    total_tokens: 245_100,
                }),
                previous_usage_input_item_count: Some(2),
                auto_compact_token_limit: None,
                retained_message_token_budget: None,
            }),
        };
        let mut machine =
            ResponseMachine::new(&ProviderRegistry::default(), request.clone()).unwrap();
        machine.begin_round().unwrap();
        let wire = machine.wire_request().unwrap();
        assert_eq!(
            wire.body["input"].as_array().unwrap().last().unwrap()["type"],
            "compaction_trigger"
        );
        assert_eq!(
            wire.headers["x-codex-beta-features"],
            "remote_compaction_v2"
        );
        machine
            .ingest(&json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "compaction", "encrypted_content": "opaque"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_compact",
                    "usage": {
                        "input_tokens": 245_000,
                        "output_tokens": 800,
                        "total_tokens": 245_800
                    }
                }
            }))
            .unwrap();
        let NextAction::Compacted { items, usage } = machine.finish_round().unwrap() else {
            panic!("expected compaction")
        };
        assert_eq!(usage.unwrap().input_tokens, 245_000);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0]["content"], "old request");
        assert_eq!(items[1]["content"], "new request");
        assert_eq!(items[2]["type"], "compaction");

        machine.begin_round().unwrap();
        let wire = machine.wire_request().unwrap();
        assert_ne!(
            wire.body["input"].as_array().unwrap().last().unwrap()["type"],
            "compaction_trigger"
        );
        machine
            .ingest(&json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"message", "role":"assistant", "content":"Done"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type":"response.completed",
                "response": {
                    "id": "resp_done",
                    "usage": {"input_tokens": 12_000, "output_tokens": 20, "total_tokens": 12_020}
                }
            }))
            .unwrap();
        let NextAction::Completed { result } = machine.finish_round().unwrap() else {
            panic!("expected completed")
        };
        assert!(matches!(
            result.history_update,
            InferenceHistoryUpdate::Replace { items } if items.last().unwrap()["content"] == "Done"
        ));
    }

    #[test]
    fn compacts_between_tool_rounds_after_reported_usage_crosses_the_limit() {
        let request = ResponseRequest {
            input: vec![build_message_item("user", "Use a tool").unwrap()],
            context: RequestContext {
                provider: crate::Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-sol".into()),
            instructions: String::new(),
            tools: vec![json!({"type":"function", "name":"work"})],
            tool_choice: None,
            reasoning_effort: Some("medium".into()),
            text_verbosity: "medium".into(),
            text_format: None,
            codex_protocol: Some(CodexProtocol::ResponsesLite),
            context_management: Some(ContextManagement::RemoteV2 {
                previous_usage: None,
                previous_usage_input_item_count: None,
                auto_compact_token_limit: Some(1_000),
                retained_message_token_budget: None,
            }),
        };
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request).unwrap();
        machine.begin_round().unwrap();
        assert_eq!(machine.round_kind, RoundKind::Response);
        machine
            .ingest(&json!({
                "type":"response.output_item.done",
                "output_index":0,
                "item":{"type":"function_call", "name":"work", "call_id":"call_1", "arguments":"{}"}
            }))
            .unwrap();
        machine
            .ingest(&json!({
                "type":"response.completed",
                "response": {
                    "id": "resp_tool",
                    "usage": {"input_tokens": 1_100, "output_tokens": 10, "total_tokens": 1_110}
                }
            }))
            .unwrap();
        assert!(matches!(
            machine.finish_round().unwrap(),
            NextAction::ToolCalls { .. }
        ));
        machine
            .submit_tool_outputs([ToolOutput {
                call_id: "call_1".into(),
                result: json!({"ok": true}),
                content: None,
            }])
            .unwrap();
        machine.begin_round().unwrap();
        assert_eq!(machine.round_kind, RoundKind::Compaction);
        assert_eq!(
            machine.wire_request().unwrap().body["input"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()["type"],
            "compaction_trigger"
        );
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

    #[test]
    fn exposes_the_exact_submitted_tool_output_item() {
        let mut machine = ResponseMachine::new(&ProviderRegistry::default(), request()).unwrap();
        let events = machine
            .submit_tool_outputs([ToolOutput {
                call_id: "call_image".into(),
                result: json!({"ok": true}),
                content: Some(json!([{"type": "input_text", "text": "done"}])),
            }])
            .unwrap();
        let [CoreEvent::ToolCallCompleted { output_item, .. }] = events.as_slice() else {
            panic!("expected a completed tool event")
        };
        assert_eq!(
            output_item,
            &Map::from_iter([
                ("type".into(), json!("function_call_output")),
                ("call_id".into(), json!("call_image")),
                (
                    "output".into(),
                    json!([{"type": "input_text", "text": "done"}]),
                ),
            ])
        );
    }
}
