use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

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

    /// Decode an SSE response incrementally.
    ///
    /// Keeping the decoded events as JSON makes the boundary stable for all
    /// provider-specific event shapes while normalization remains in
    /// `ResponseMachine`.
    pub async fn stream_round<F>(
        &self,
        wire: WireRequest,
        cancellation: &CancellationToken,
        mut on_event: F,
    ) -> Result<()>
    where
        F: FnMut(Value) -> Result<()>,
    {
        let mut request = self
            .client
            .post(&wire.url)
            .body(serde_json::to_vec(&wire.body)?);
        for (name, value) in &wire.headers {
            request = request.header(name, value);
        }
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

        let mut source = response.bytes_stream().eventsource();
        while let Some(next) = tokio::select! {
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            next = source.next() => next,
        } {
            let event = next.map_err(|error| Error::Transport(error.to_string()))?;
            if event.data == "[DONE]" || event.data.trim().is_empty() {
                continue;
            }
            on_event(serde_json::from_str(&event.data)?)?;
        }
        Ok(())
    }
}
