use std::collections::BTreeMap;
use std::time::SystemTime;

use hmac::{Hmac, Mac};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{CoreEvent, Error, RequestContext, Result};

type HmacSha256 = Hmac<Sha256>;

pub struct BedrockCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: String,
}

impl BedrockCredentials {
    pub fn from_context(context: &RequestContext) -> Result<Self> {
        let access_key = context
            .headers
            .get("x-bedrock-access-key")
            .ok_or_else(|| Error::State("bedrock context missing access key".into()))?
            .clone();
        let secret_key = context
            .headers
            .get("x-bedrock-secret-key")
            .ok_or_else(|| Error::State("bedrock context missing secret key".into()))?
            .clone();
        let region = context
            .headers
            .get("x-bedrock-region")
            .cloned()
            .unwrap_or_else(|| "us-east-1".into());
        let session_token = context.headers.get("x-bedrock-session-token").cloned();
        Ok(Self {
            access_key,
            secret_key,
            session_token,
            region,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BedrockTransport {
    client: reqwest::Client,
}

impl Default for BedrockTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl BedrockTransport {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("the default HTTP client configuration is valid"),
        }
    }

    /// Stream a converse-stream request, emitting CoreEvents for text deltas
    /// and tool call starts. Returns completed tool calls with their accumulated
    /// input JSON.
    pub async fn converse_stream<F>(
        &self,
        model: &str,
        body: Value,
        credentials: &BedrockCredentials,
        cancellation: &CancellationToken,
        mut on_event: F,
    ) -> Result<Vec<CompletedToolCall>>
    where
        F: FnMut(CoreEvent) -> Result<()>,
    {
        let host = format!("bedrock-runtime.{}.amazonaws.com", credentials.region);
        let path = format!("/model/{}/converse-stream", model);
        let url = format!("https://{}{}", host, path);
        let payload = serde_json::to_vec(&body)?;
        let now = SystemTime::now();
        let headers = sign_request("POST", &path, &host, &payload, credentials, now)?;

        let mut request = self.client.post(&url).body(payload);
        for (name, value) in &headers {
            request = request.header(name.as_str(), value.as_str());
        }
        request = request.header("Content-Type", "application/json");

        let response = tokio::select! {
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            response = request.send() => response?,
        };

        let status = response.status();
        if !status.is_success() {
            return Err(Error::HttpStatus {
                status: status.as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }

        let mut stream = response.bytes_stream();
        let mut decoder = EventStreamDecoder::new();
        let mut tool_calls: Vec<CompletedToolCall> = Vec::new();

        use futures_util::StreamExt;
        while let Some(chunk) = tokio::select! {
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            next = stream.next() => next,
        } {
            let bytes = chunk.map_err(|e| Error::Transport(e.to_string()))?;
            decoder.push(&bytes);
            while let Some(event) = decoder.next_event()? {
                for ev in parse_bedrock_event(&event)? {
                    match ev {
                        BedrockStreamEvent::TextDelta(delta) => {
                            on_event(CoreEvent::TextDelta { delta })?;
                        }
                        BedrockStreamEvent::ToolStart { name, call_id } => {
                            on_event(CoreEvent::ToolCallStarted {
                                name: name.clone(),
                                call_id: call_id.clone(),
                                arguments: json!({}),
                            })?;
                            tool_calls.push(CompletedToolCall {
                                name,
                                call_id,
                                arguments_json: String::new(),
                            });
                        }
                        BedrockStreamEvent::ToolInputDelta(chunk) => {
                            if let Some(last) = tool_calls.last_mut() {
                                last.arguments_json.push_str(&chunk);
                            }
                        }
                        BedrockStreamEvent::Stop => {}
                    }
                }
            }
        }
        Ok(tool_calls)
    }
}

#[derive(Clone, Debug)]
pub struct CompletedToolCall {
    pub name: String,
    pub call_id: String,
    pub arguments_json: String,
}

pub struct BedrockRequest {
    pub model: String,
    pub system: Vec<Value>,
    pub messages: Vec<Value>,
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
    pub reasoning: Option<Value>,
}

impl BedrockRequest {
    pub fn body(&self) -> Value {
        let mut body = json!({
            "messages": self.messages,
            "inferenceConfig": {
                "maxTokens": 16384,
            },
        });
        if !self.system.is_empty() {
            body["system"] = json!(self.system);
        }
        if !self.tools.is_empty() {
            body["toolConfig"] = json!({
                "tools": self.tools,
            });
            if let Some(choice) = &self.tool_choice {
                body["toolConfig"]["toolChoice"] = choice.clone();
            }
        }
        if let Some(reasoning) = &self.reasoning {
            body["additionalModelRequestFields"] = json!({
                "thinking": reasoning,
            });
        }
        body
    }
}

/// Convert LMX's internal representation into Bedrock converse messages.
pub fn build_converse_messages(
    input: &[Map<String, Value>],
    instructions: &str,
) -> (Vec<Value>, Vec<Value>) {
    let mut system = Vec::new();
    if !instructions.is_empty() {
        system.push(json!({"text": instructions}));
    }
    let mut messages: Vec<Value> = Vec::new();
    for item in input {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
        match item_type {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                if role == "system" || role == "developer" {
                    if let Some(content) = item.get("content") {
                        let text = match content {
                            Value::String(s) => s.clone(),
                            Value::Array(parts) => parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n"),
                            _ => continue,
                        };
                        system.push(json!({"text": text}));
                    }
                    continue;
                }
                let bedrock_role = if role == "assistant" {
                    "assistant"
                } else {
                    "user"
                };
                let content = match item.get("content") {
                    Some(Value::String(s)) => vec![json!({"text": s})],
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| {
                            if let Some(text) = p.get("text").and_then(Value::as_str) {
                                Some(json!({"text": text}))
                            } else if let Some(image) = p.get("image_url") {
                                let url =
                                    image.get("url").and_then(Value::as_str).unwrap_or_default();
                                if let Some(data) = url.strip_prefix("data:image/") {
                                    let (media_type, b64) =
                                        data.split_once(";base64,").unwrap_or(("png", ""));
                                    Some(json!({
                                        "image": {
                                            "format": media_type,
                                            "source": {"bytes": b64}
                                        }
                                    }))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect(),
                    _ => continue,
                };
                messages.push(json!({"role": bedrock_role, "content": content}));
            }
            "function_call" => {
                let tool_use = json!({
                    "toolUse": {
                        "toolUseId": item.get("call_id").and_then(Value::as_str).unwrap_or_default(),
                        "name": item.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "input": parse_arguments(item.get("arguments")),
                    }
                });
                append_to_last_or_push(&mut messages, "assistant", tool_use);
            }
            "function_call_output" => {
                let output_text = match item.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => "null".into(),
                };
                let tool_result = json!({
                    "toolResult": {
                        "toolUseId": item.get("call_id").and_then(Value::as_str).unwrap_or_default(),
                        "content": [{"text": output_text}],
                    }
                });
                append_to_last_or_push(&mut messages, "user", tool_result);
            }
            _ => {}
        }
    }
    (system, messages)
}

fn append_to_last_or_push(messages: &mut Vec<Value>, role: &str, content_block: Value) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        content.push(content_block);
        return;
    }
    messages.push(json!({"role": role, "content": [content_block]}));
}

fn parse_arguments(value: Option<&Value>) -> Value {
    match value {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(raw)) if raw.is_empty() => json!({}),
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or_else(|_| json!({})),
        Some(v) => v.clone(),
    }
}

/// Convert LMX tool definitions (OpenAI format) to Bedrock toolSpec format.
pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            let name = function.get("name")?.as_str()?;
            let description = function
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            Some(json!({
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": parameters},
                }
            }))
        })
        .collect()
}

/// Bedrock converse-stream event types that we track.
#[derive(Debug)]
enum BedrockStreamEvent {
    TextDelta(String),
    ToolStart { name: String, call_id: String },
    ToolInputDelta(String),
    Stop,
}

fn parse_bedrock_event(event: &BedrockEvent) -> Result<Vec<BedrockStreamEvent>> {
    let value = &event.payload;
    let mut events = Vec::new();

    match event.event_type.as_str() {
        "contentBlockDelta" => {
            let delta = value.get("delta").unwrap_or(value);
            if let Some(text) = delta.get("text").and_then(Value::as_str)
                && !text.is_empty()
            {
                events.push(BedrockStreamEvent::TextDelta(text.into()));
            }
            if let Some(input) = delta
                .get("toolUse")
                .and_then(|t| t.get("input"))
                .and_then(Value::as_str)
            {
                events.push(BedrockStreamEvent::ToolInputDelta(input.into()));
            }
        }
        "contentBlockStart" => {
            let start = value.get("start").unwrap_or(value);
            if let Some(tool_use) = start.get("toolUse") {
                events.push(BedrockStreamEvent::ToolStart {
                    name: tool_use
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    call_id: tool_use
                        .get("toolUseId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                });
            }
        }
        "messageStop" => {
            events.push(BedrockStreamEvent::Stop);
        }
        _ => {}
    }

    Ok(events)
}

/// Bedrock uses the AWS event-stream binary protocol over HTTP.
struct EventStreamDecoder {
    buffer: Vec<u8>,
}

#[derive(Debug)]
struct BedrockEvent {
    event_type: String,
    payload: Value,
}

impl EventStreamDecoder {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    fn next_event(&mut self) -> Result<Option<BedrockEvent>> {
        // AWS event stream binary format:
        // [4 bytes total length][4 bytes headers length][4 bytes prelude CRC]
        // [headers...][payload...][4 bytes message CRC]
        if self.buffer.len() < 12 {
            return Ok(None);
        }
        let total_length = u32::from_be_bytes([
            self.buffer[0],
            self.buffer[1],
            self.buffer[2],
            self.buffer[3],
        ]) as usize;
        if self.buffer.len() < total_length {
            return Ok(None);
        }
        let headers_length = u32::from_be_bytes([
            self.buffer[4],
            self.buffer[5],
            self.buffer[6],
            self.buffer[7],
        ]) as usize;

        let prelude_end = 12;
        let headers_end = prelude_end + headers_length;
        let payload_end = total_length - 4;

        // Verify prelude CRC
        let prelude_crc_expected = u32::from_be_bytes([
            self.buffer[8],
            self.buffer[9],
            self.buffer[10],
            self.buffer[11],
        ]);
        let prelude_crc_actual = crc32fast::hash(&self.buffer[0..8]);
        if prelude_crc_expected != prelude_crc_actual {
            self.buffer.drain(..total_length);
            return Err(Error::Transport("event stream prelude CRC mismatch".into()));
        }

        // Verify message CRC
        let message_crc_expected = u32::from_be_bytes([
            self.buffer[payload_end],
            self.buffer[payload_end + 1],
            self.buffer[payload_end + 2],
            self.buffer[payload_end + 3],
        ]);
        let message_crc_actual = crc32fast::hash(&self.buffer[0..payload_end]);
        if message_crc_expected != message_crc_actual {
            self.buffer.drain(..total_length);
            return Err(Error::Transport("event stream message CRC mismatch".into()));
        }

        let headers_data = &self.buffer[prelude_end..headers_end];
        let parsed_headers = parse_event_headers(headers_data);

        let payload_bytes = &self.buffer[headers_end..payload_end];

        if parsed_headers.message_type.as_deref() == Some("exception") {
            let body = String::from_utf8_lossy(payload_bytes).into_owned();
            self.buffer.drain(..total_length);
            return Err(Error::Transport(format!("bedrock exception: {body}")));
        }

        let event_type = parsed_headers.event_type.unwrap_or_default();

        let result = if payload_bytes.is_empty() {
            None
        } else {
            match serde_json::from_slice::<Value>(payload_bytes) {
                Ok(payload) => Some(BedrockEvent {
                    event_type,
                    payload,
                }),
                Err(_) => None,
            }
        };

        self.buffer.drain(..total_length);
        Ok(result)
    }
}

#[derive(Default)]
struct EventHeaders {
    event_type: Option<String>,
    message_type: Option<String>,
}

fn parse_event_headers(mut data: &[u8]) -> EventHeaders {
    let mut headers = EventHeaders::default();
    while data.len() > 3 {
        let name_len = data[0] as usize;
        if data.len() < 1 + name_len + 1 {
            break;
        }
        let name = match std::str::from_utf8(&data[1..1 + name_len]) {
            Ok(n) => n,
            Err(_) => break,
        };
        let header_type = data[1 + name_len];
        let value_start = 1 + name_len + 1;

        if header_type == 7 {
            if data.len() < value_start + 2 {
                break;
            }
            let value_len = u16::from_be_bytes([data[value_start], data[value_start + 1]]) as usize;
            if data.len() < value_start + 2 + value_len {
                break;
            }
            let value = std::str::from_utf8(&data[value_start + 2..value_start + 2 + value_len])
                .unwrap_or_default();
            match name {
                ":event-type" => headers.event_type = Some(value.into()),
                ":message-type" => headers.message_type = Some(value.into()),
                ":exception-type" => headers.message_type = Some("exception".into()),
                _ => {}
            }
            data = &data[value_start + 2 + value_len..];
        } else {
            break;
        }
    }
    headers
}

// --- AWS SigV4 implementation ---

fn sign_request(
    method: &str,
    path: &str,
    host: &str,
    payload: &[u8],
    credentials: &BedrockCredentials,
    now: SystemTime,
) -> Result<BTreeMap<String, String>> {
    let datetime = format_datetime(now);
    let date = &datetime[..8];
    let service = "bedrock";
    let region = &credentials.region;

    let payload_hash = hex::encode(Sha256::digest(payload));

    let mut headers = BTreeMap::new();
    headers.insert("host".into(), host.into());
    headers.insert("x-amz-date".into(), datetime.clone());
    headers.insert("x-amz-content-sha256".into(), payload_hash.clone());
    if let Some(token) = &credentials.session_token {
        headers.insert("x-amz-security-token".into(), token.clone());
    }

    let signed_headers: Vec<&str> = headers.keys().map(String::as_str).collect();
    let signed_headers_str = signed_headers.join(";");

    let canonical_headers: String = headers.iter().map(|(k, v)| format!("{k}:{v}\n")).collect();

    let canonical_request =
        format!("{method}\n{path}\n\n{canonical_headers}\n{signed_headers_str}\n{payload_hash}");

    let credential_scope = format!("{date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{datetime}\n{credential_scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );

    let signing_key = derive_signing_key(&credentials.secret_key, date, region, service)?;
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes())?);

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key, credential_scope, signed_headers_str, signature
    );

    headers.insert("authorization".into(), authorization);
    Ok(headers)
}

fn derive_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Result<Vec<u8>> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes())?;
    let k_region = hmac_sha256(&k_date, region.as_bytes())?;
    let k_service = hmac_sha256(&k_region, service.as_bytes())?;
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Result<Vec<u8>> {
    let mut mac = HmacSha256::new_from_slice(key)
        .map_err(|e| Error::Transport(format!("HMAC key error: {e}")))?;
    mac.update(data);
    Ok(mac.finalize().into_bytes().to_vec())
}

fn format_datetime(time: SystemTime) -> String {
    let duration = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since Unix epoch to Y/M/D (simplified calendar arithmetic)
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}{month:02}{day:02}T{hours:02}{minutes:02}{seconds:02}Z")
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CoreEvent, NextAction, Provider, ProviderRegistry, RequestContext, ResponseMachine,
        ResponseRequest, build_message_item, execute_round,
    };

    fn test_context() -> Option<RequestContext> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok()?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok()?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into());
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        let mut headers = BTreeMap::new();
        headers.insert("x-bedrock-access-key".into(), access_key);
        headers.insert("x-bedrock-secret-key".into(), secret_key);
        headers.insert("x-bedrock-region".into(), region);
        if let Some(token) = session_token {
            headers.insert("x-bedrock-session-token".into(), token);
        }
        Some(RequestContext {
            provider: Provider::Bedrock,
            api_key: String::new(),
            base_url: None,
            headers,
            query: BTreeMap::new(),
        })
    }

    #[tokio::test]
    async fn live_converse_stream() {
        let Some(context) = test_context() else {
            eprintln!("skipping: AWS_ACCESS_KEY_ID not set");
            return;
        };

        let credentials = BedrockCredentials::from_context(&context).unwrap();
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": [{"text": "Say hello in exactly 3 words."}]}],
            "system": [{"text": "You are a helpful assistant."}],
            "inferenceConfig": {"maxTokens": 256},
        });

        let transport = BedrockTransport::new();
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut events = Vec::new();
        transport
            .converse_stream(
                "us.anthropic.claude-sonnet-4-6",
                body,
                &credentials,
                &cancel,
                |event| {
                    events.push(event);
                    Ok(())
                },
            )
            .await
            .unwrap();

        let text: String = events
            .iter()
            .filter_map(|e| match e {
                CoreEvent::TextDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert!(!text.is_empty(), "expected non-empty response from Bedrock");
    }

    #[tokio::test]
    async fn live_response_machine_round() {
        let Some(context) = test_context() else {
            eprintln!("skipping: AWS_ACCESS_KEY_ID not set");
            return;
        };

        let registry = ProviderRegistry::from_environment().unwrap();
        let request = ResponseRequest {
            input: vec![
                build_message_item("user", "What is 2+2? Reply with just the number.").unwrap(),
            ],
            context,
            model: Some("us.anthropic.claude-sonnet-4-6".into()),
            instructions: String::new(),
            tools: vec![],
            tool_choice: None,
            reasoning_effort: None,
            text_verbosity: "low".into(),
            text_format: None,
            context_management: None,
        };

        let mut machine = ResponseMachine::new(&registry, request).unwrap();
        let round = execute_round(&mut machine).await.unwrap();

        assert!(
            matches!(&round.next, NextAction::Completed { result } if result.output_text.contains('4')),
            "expected response containing '4', got: {:?}",
            round.next
        );
    }
}
