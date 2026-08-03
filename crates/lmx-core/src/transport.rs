use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;

use crate::{Error, Result, WireRequest};

/// The core-owned OpenAI-compatible HTTP/SSE transport.
///
/// Bindings never use an SDK-specific HTTP client. They obtain a `WireRequest`
/// from `ResponseMachine`, hand it to this transport, and feed decoded events
/// back to the same machine.
#[derive(Clone, Debug)]
pub struct OpenAiTransport {
    client: reqwest::Client,
}

impl Default for OpenAiTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiTransport {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(90))
                .build()
                .expect("the default HTTP client configuration is valid"),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Read an SSE response into decoded Responses API events.
    ///
    /// Keeping the decoded events as JSON makes the boundary stable for all
    /// provider-specific event shapes while normalization remains in
    /// `ResponseMachine`.
    pub async fn stream_round(&self, wire: WireRequest) -> Result<Vec<Value>> {
        let mut request = self.client.post(&wire.url).json(&wire.body);
        for (name, value) in &wire.headers {
            request = request.header(name, value);
        }
        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::HttpStatus {
                status: status.as_u16(),
                body: response.text().await.unwrap_or_default(),
            });
        }

        let mut source = response.bytes_stream().eventsource();
        let mut events = Vec::new();
        while let Some(next) = source.next().await {
            let event = next.map_err(|error| Error::Transport(error.to_string()))?;
            if event.data == "[DONE]" || event.data.trim().is_empty() {
                continue;
            }
            events.push(serde_json::from_str(&event.data)?);
        }
        Ok(events)
    }
}
