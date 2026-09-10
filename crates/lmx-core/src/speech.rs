use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{Error, Provider, RequestContext, Result, endpoint_url};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechFormat {
    #[default]
    Wav,
    Mp3,
    Opus,
    Aac,
    Flac,
    Pcm,
}

impl SpeechFormat {
    fn content_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Pcm => "audio/pcm",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechRequest {
    pub context: RequestContext,
    #[serde(default = "default_model")]
    pub model: String,
    pub input: String,
    pub voice: String,
    pub instructions: Option<String>,
    #[serde(default)]
    pub format: SpeechFormat,
    pub speed: Option<f64>,
}

fn default_model() -> String {
    "gpt-4o-mini-tts".into()
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechResult {
    pub model: String,
    pub voice: String,
    pub format: SpeechFormat,
    pub content_base64: String,
    pub content_type: String,
}

fn speech_body(request: &SpeechRequest) -> Result<Value> {
    if request.context.provider != Provider::Openai {
        return Err(Error::UnsupportedCapability(
            request.context.provider.as_str().into(),
            "speech generation",
        ));
    }
    if request.input.trim().is_empty()
        || request.voice.trim().is_empty()
        || request.model.trim().is_empty()
    {
        return Err(Error::State(
            "speech input, voice, and model must not be empty".into(),
        ));
    }
    if request
        .speed
        .is_some_and(|speed| !speed.is_finite() || !(0.25..=4.0).contains(&speed))
    {
        return Err(Error::State(
            "speech speed must be between 0.25 and 4".into(),
        ));
    }
    let mut body = json!({
        "model": request.model, "input": request.input, "voice": request.voice,
        "response_format": request.format,
    });
    if let Some(instructions) = &request.instructions {
        body["instructions"] = json!(instructions);
    }
    if let Some(speed) = request.speed {
        body["speed"] = json!(speed);
    }
    Ok(body)
}

pub async fn generate_speech(
    request: SpeechRequest,
    cancellation: &CancellationToken,
) -> Result<SpeechResult> {
    let body = speech_body(&request)?;
    let base = request
        .context
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let mut call = client
        .post(endpoint_url(base, "audio/speech", &request.context.query)?)
        .json(&body);
    for (name, value) in &request.context.headers {
        call = call.header(name, value);
    }
    call = call
        .bearer_auth(&request.context.api_key)
        .header("User-Agent", format!("lmx/{}", crate::VERSION));
    let (content, content_type) = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(Error::Cancelled),
        result = async {
            let response = call.send().await?;
            if !response.status().is_success() {
                return Err(Error::HttpStatus { status: response.status().as_u16(), body: response.text().await.unwrap_or_default() });
            }
            let content_type = request.format.content_type().to_owned();
            let content = response.bytes().await?;
            if content.is_empty() { return Err(Error::Event("speech response contained no audio".into())); }
            Ok((content, content_type))
        } => result?,
    };
    Ok(SpeechResult {
        model: request.model,
        voice: request.voice,
        format: request.format,
        content_base64: STANDARD.encode(content),
        content_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> SpeechRequest {
        serde_json::from_value(json!({"context":{"provider":"openai","apiKey":"test"},"input":"  Hello.  ","voice":"cedar","instructions":"Quiet relief"})).unwrap()
    }
    #[test]
    fn preserves_script_and_separates_delivery() {
        assert_eq!(
            speech_body(&request()).unwrap(),
            json!({"model":"gpt-4o-mini-tts","input":"  Hello.  ","voice":"cedar","instructions":"Quiet relief","response_format":"wav"})
        );
    }
    #[test]
    fn rejects_other_credentials_and_invalid_speed() {
        let mut request = request();
        request.context.provider = Provider::Codex;
        assert!(matches!(
            speech_body(&request),
            Err(Error::UnsupportedCapability(_, _))
        ));
        request.context.provider = Provider::Openai;
        request.speed = Some(0.0);
        assert!(speech_body(&request).is_err());
    }
    #[tokio::test]
    async fn cancellation_prevents_request() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(matches!(
            generate_speech(request(), &token).await,
            Err(Error::Cancelled)
        ));
    }
}
