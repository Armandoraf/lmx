use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Error, RequestContext, Result, endpoint_url};

const SIZES: &[&str] = &["720x1280", "1280x720", "1024x1792", "1792x1024"];
const SECONDS: &[u32] = &[4, 8, 12];

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoRequest {
    pub prompt: String,
    pub context: RequestContext,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub seconds: Option<u32>,
    #[serde(default)]
    pub duration_seconds: Option<u32>,
    #[serde(default)]
    pub input_reference: Option<String>,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_seconds: f64,
}

fn default_poll_interval() -> f64 {
    10.0
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoJob {
    pub job_id: String,
    pub provider: String,
    pub model: String,
    pub prompt: String,
    pub size: String,
    pub width: u32,
    pub height: u32,
    pub duration_seconds: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_reference: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoResult {
    pub job: VideoJob,
    pub content_base64: String,
    pub content_type: String,
}

fn size(request: &VideoRequest) -> Result<(String, u32, u32)> {
    let size = request
        .size
        .clone()
        .or_else(|| match (request.width, request.height) {
            (Some(w), Some(h)) => Some(format!("{w}x{h}")),
            _ => None,
        })
        .unwrap_or_else(|| SIZES[0].into());
    if !SIZES.contains(&size.as_str()) {
        return Err(Error::State(format!(
            "unsupported video size {size}; supported: {}",
            SIZES.join(", ")
        )));
    }
    let (width, height) = size
        .split_once('x')
        .ok_or_else(|| Error::State("video size must use WIDTHxHEIGHT format".into()))?;
    let width = width
        .parse()
        .map_err(|_| Error::State("video width must be an integer".into()))?;
    let height = height
        .parse()
        .map_err(|_| Error::State("video height must be an integer".into()))?;
    Ok((size, width, height))
}

fn endpoint(context: &RequestContext, path: &str) -> Result<String> {
    let base_url = context
        .base_url
        .as_deref()
        .ok_or_else(|| Error::MissingBaseUrl(context.provider.as_str().into()))?;
    endpoint_url(base_url, path, &context.query)
}

async fn checked_json(response: reqwest::Response) -> Result<Value> {
    if !response.status().is_success() {
        return Err(Error::HttpStatus {
            status: response.status().as_u16(),
            body: response.text().await.unwrap_or_default(),
        });
    }
    Ok(response.json().await?)
}

pub async fn generate_video(request: VideoRequest) -> Result<VideoResult> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(Error::State("prompt must not be empty".into()));
    }
    let provider = request.context.provider.as_str();
    if provider == "bedrock" {
        return Err(Error::UnsupportedCapability(
            "bedrock".into(),
            "video generation without the optional aws backend",
        ));
    }
    if provider != "openai" && provider != "azure" {
        return Err(Error::UnknownProvider(provider.into()));
    }
    let model = request.model.clone().unwrap_or_else(|| "sora-2".into());
    let (size, width, height) = size(&request)?;
    let seconds = request
        .seconds
        .or(request.duration_seconds)
        .unwrap_or(SECONDS[0]);
    if !SECONDS.contains(&seconds) {
        return Err(Error::State(format!(
            "unsupported seconds {seconds}; supported: 4, 8, 12"
        )));
    }
    let mut body = json!({"model":model,"prompt":prompt,"size":size,"seconds":seconds});
    if let Some(reference) = &request.input_reference {
        if reference.trim().is_empty() {
            return Err(Error::State("input_reference must not be empty".into()));
        }
        body["input_reference"] = if reference.starts_with("http://")
            || reference.starts_with("https://")
            || reference.starts_with("data:")
        {
            json!({"image_url":reference})
        } else {
            json!({"file_id":reference})
        };
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let mut create = client
        .post(endpoint(&request.context, "videos")?)
        .json(&body);
    create = create
        .header(
            "Authorization",
            format!("Bearer {}", request.context.api_key),
        )
        .header("User-Agent", format!("lmx/{}", crate::VERSION));
    for (name, value) in &request.context.headers {
        create = create.header(name, value);
    }
    let mut video = checked_json(create.send().await?).await?;
    loop {
        let status = video
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if status == "completed" {
            break;
        }
        if matches!(status, "failed" | "cancelled" | "expired") {
            return Err(Error::State(format!(
                "video generation ended with status {status}: {}",
                video.get("error").unwrap_or(&Value::Null)
            )));
        }
        let id = video
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Event("video response did not include an id".into()))?;
        tokio::time::sleep(std::time::Duration::from_secs_f64(
            request.poll_interval_seconds.max(0.01),
        ))
        .await;
        let mut poll = client
            .get(endpoint(&request.context, &format!("videos/{id}"))?)
            .header(
                "Authorization",
                format!("Bearer {}", request.context.api_key),
            );
        for (name, value) in &request.context.headers {
            poll = poll.header(name, value);
        }
        video = checked_json(poll.send().await?).await?;
    }
    let id = video
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Event("video response did not include an id".into()))?;
    let mut download = client
        .get(endpoint(
            &request.context,
            &format!("videos/{id}/content?variant=video"),
        )?)
        .header(
            "Authorization",
            format!("Bearer {}", request.context.api_key),
        );
    for (name, value) in &request.context.headers {
        download = download.header(name, value);
    }
    let download = download.send().await?;
    if !download.status().is_success() {
        return Err(Error::HttpStatus {
            status: download.status().as_u16(),
            body: download.text().await.unwrap_or_default(),
        });
    }
    let content_type = download
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("video/mp4")
        .into();
    let content = download.bytes().await?;
    Ok(VideoResult {
        job: VideoJob {
            job_id: id.into(),
            provider: provider.into(),
            model,
            prompt: prompt.into(),
            size,
            width,
            height,
            duration_seconds: seconds,
            input_reference: request.input_reference,
            status: video
                .get("status")
                .and_then(Value::as_str)
                .map(str::to_owned),
            progress: video
                .get("progress")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
        },
        content_base64: STANDARD.encode(content),
        content_type,
    })
}
