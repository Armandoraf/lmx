use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::{
    CoreEvent, Error, NextAction, Provider, ProviderRegistry, RequestContext, ResponseMachine,
    ResponseRequest, Result, endpoint_url, execute_round, execute_round_with_observer,
    prompt_for_white_key, remove_white_key_background,
};

const OPENAI_SIZES: &[&str] = &["1024x1024", "1536x1024", "1024x1536"];
const NANOGPT_SIZES: &[&str] = &["1024x1024"];
const GPT_IMAGE_2_MAX_EDGE: u32 = 3_840;
const GPT_IMAGE_2_DIMENSION_MULTIPLE: u32 = 16;
const GPT_IMAGE_2_MIN_PIXELS: u64 = 655_360;
const GPT_IMAGE_2_MAX_PIXELS: u64 = 8_294_400;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageRequest {
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
    pub quality: Option<String>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub input_images: Vec<String>,
    #[serde(default)]
    pub input_fidelity: Option<String>,
    #[serde(default = "default_image_count")]
    pub count: u8,
    #[serde(default)]
    pub extra_body: BTreeMap<String, Value>,
}

fn default_image_count() -> u8 {
    1
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageStreamRequest {
    #[serde(flatten)]
    pub image: ImageRequest,
    #[serde(default = "default_partial_images")]
    pub partial_images: u8,
}

fn default_partial_images() -> u8 {
    2
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageJob {
    pub provider: String,
    pub model: String,
    pub prompt: String,
    pub size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_processing: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageResult {
    pub job: ImageJob,
    pub content_base64: String,
    pub content_type: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageBatchResult {
    pub images: Vec<ImageResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageStreamEvent {
    Partial {
        image_index: u8,
        partial_index: u8,
        result: ImageResult,
    },
    Completed {
        image_index: u8,
        result: ImageResult,
    },
    BatchCompleted {
        #[serde(skip_serializing_if = "Option::is_none")]
        usage: Option<Value>,
    },
}

struct ImageSpec {
    default_model: &'static str,
    models: &'static [&'static str],
    sizes: &'static [&'static str],
    base_url: &'static str,
}

fn image_spec(provider: &str) -> Result<ImageSpec> {
    match provider {
        "openai" => Ok(ImageSpec {
            default_model: "gpt-image-1",
            models: &["gpt-image-1", "gpt-image-2"],
            sizes: OPENAI_SIZES,
            base_url: "https://api.openai.com/v1",
        }),
        "azure" => Ok(ImageSpec {
            default_model: "gpt-image-2",
            models: &["gpt-image-2", "gpt-image-1"],
            sizes: OPENAI_SIZES,
            base_url: "",
        }),
        "nanogpt" => Ok(ImageSpec {
            default_model: "qwen-image",
            models: &["qwen-image", "z-image-turbo", "hidream", "chroma"],
            sizes: NANOGPT_SIZES,
            base_url: "https://nano-gpt.com/v1",
        }),
        other => Err(Error::UnknownProvider(other.into())),
    }
}

#[derive(Debug)]
struct ResolvedSize {
    value: String,
    width: Option<u32>,
    height: Option<u32>,
}

fn parse_dimensions(size: &str) -> Result<(u32, u32)> {
    let (width, height) = size
        .split_once('x')
        .ok_or_else(|| Error::State(format!("image size must use WIDTHxHEIGHT format: {size}")))?;
    let width = width
        .parse()
        .map_err(|_| Error::State("image width must be an integer".into()))?;
    let height = height
        .parse()
        .map_err(|_| Error::State("image height must be an integer".into()))?;
    Ok((width, height))
}

fn validate_gpt_image_2_size(width: u32, height: u32) -> Result<()> {
    let longest_edge = width.max(height);
    let shortest_edge = width.min(height);
    let pixels = u64::from(width) * u64::from(height);
    let valid = longest_edge <= GPT_IMAGE_2_MAX_EDGE
        && width.is_multiple_of(GPT_IMAGE_2_DIMENSION_MULTIPLE)
        && height.is_multiple_of(GPT_IMAGE_2_DIMENSION_MULTIPLE)
        && u64::from(longest_edge) <= u64::from(shortest_edge) * 3
        && (GPT_IMAGE_2_MIN_PIXELS..=GPT_IMAGE_2_MAX_PIXELS).contains(&pixels);
    if valid {
        return Ok(());
    }
    Err(Error::State(format!(
        "unsupported gpt-image-2 size {width}x{height}; dimensions must be multiples of {GPT_IMAGE_2_DIMENSION_MULTIPLE}, each edge at most {GPT_IMAGE_2_MAX_EDGE}px, aspect ratio at most 3:1, and total pixels between {GPT_IMAGE_2_MIN_PIXELS} and {GPT_IMAGE_2_MAX_PIXELS}"
    )))
}

fn resolve_size(request: &ImageRequest, model: &str, sizes: &[&str]) -> Result<ResolvedSize> {
    let size = request
        .size
        .clone()
        .or_else(|| match (request.width, request.height) {
            (Some(width), Some(height)) => Some(format!("{width}x{height}")),
            _ => None,
        })
        .unwrap_or_else(|| sizes[0].into());
    if is_gpt_image_2(model) && size == "auto" {
        return Ok(ResolvedSize {
            value: size,
            width: None,
            height: None,
        });
    }
    if !is_gpt_image_2(model) && !sizes.contains(&size.as_str()) {
        return Err(Error::State(format!(
            "unsupported image size {size}; supported: {}",
            sizes.join(", ")
        )));
    }
    let (width, height) = parse_dimensions(&size)?;
    if is_gpt_image_2(model) {
        validate_gpt_image_2_size(width, height)?;
    }
    Ok(ResolvedSize {
        value: size,
        width: Some(width),
        height: Some(height),
    })
}

fn is_gpt_image_2(model: &str) -> bool {
    model == "gpt-image-2" || model.starts_with("gpt-image-2-")
}

fn request_url(context: &RequestContext, fallback: &str, route: &str) -> Result<String> {
    let base = context
        .base_url
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback);
    if base.is_empty() {
        return Err(Error::MissingBaseUrl(context.provider.as_str().into()));
    }
    endpoint_url(base, &format!("images/{route}"), &context.query)
}

pub async fn generate_image(request: ImageRequest) -> Result<ImageResult> {
    if request.count != 1 {
        return Err(Error::State(
            "generate_image requires count=1; use generate_images for batches".into(),
        ));
    }
    let batch = generate_images(request).await?;
    batch
        .images
        .into_iter()
        .next()
        .ok_or_else(|| Error::Event("image response did not include any images".into()))
}

pub async fn generate_images(request: ImageRequest) -> Result<ImageBatchResult> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(Error::State("prompt must not be empty".into()));
    }
    if !(1..=10).contains(&request.count) {
        return Err(Error::State("count must be between 1 and 10".into()));
    }
    if request.context.provider == Provider::Codex {
        return generate_codex_images(request).await;
    }
    let provider = request.context.provider.as_str();
    let spec = image_spec(provider)?;
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| spec.default_model.into());
    if !spec.models.contains(&model.as_str()) {
        return Err(Error::UnsupportedModel {
            provider: provider.into(),
            model,
        });
    }
    if request.background.is_some() && provider == "nanogpt" {
        return Err(Error::UnsupportedCapability(
            provider.into(),
            "image background selection",
        ));
    }
    let size = resolve_size(&request, &model, spec.sizes)?;
    let white_keying =
        request.background.as_deref() == Some("transparent") && is_gpt_image_2(&model);
    let provider_prompt = if white_keying {
        prompt_for_white_key(prompt)
    } else {
        prompt.into()
    };
    let provider_size = match model.as_str() {
        "qwen-image" => "auto".into(),
        "z-image-turbo" => "1024*1024".into(),
        _ => size.value.clone(),
    };
    let mut body = json!({
        "model": model,
        "prompt": provider_prompt,
        "size": provider_size,
        "n": request.count,
    });
    if provider == "nanogpt" {
        body["response_format"] = json!("url");
        let mut extra = match model.as_str() {
            "qwen-image" => {
                json!({"num_inference_steps":30,"guidance_scale":2.5,"enable_safety_checker":true,"negative_prompt":" "})
            }
            "hidream" => json!({"num_inference_steps":50,"guidance_scale":5}),
            "chroma" => json!({"negative_prompt":"","guidance_scale":4.5,"num_inference_steps":25}),
            _ => json!({}),
        };
        for (key, value) in &request.extra_body {
            extra[key] = value.clone();
        }
        if !extra.as_object().is_some_and(|value| value.is_empty()) {
            body["extra_body"] = extra;
        }
    } else {
        body["quality"] = json!(request.quality.as_deref().unwrap_or("high"));
        if let Some(background) = &request.background {
            body["background"] = json!(if white_keying { "opaque" } else { background });
        }
    }
    let mut headers = request.context.headers.clone();
    headers.insert(
        "Authorization".into(),
        format!("Bearer {}", request.context.api_key),
    );
    headers
        .entry("User-Agent".into())
        .or_insert_with(|| format!("lmx/{}", crate::VERSION));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let mut call = if !request.input_images.is_empty() && provider != "nanogpt" {
        let mut form = reqwest::multipart::Form::new()
            .text("model", model.clone())
            .text("prompt", provider_prompt.clone())
            .text("size", provider_size.clone())
            .text("n", request.count.to_string())
            .text(
                "quality",
                request.quality.clone().unwrap_or_else(|| "high".into()),
            );
        if let Some(background) = &request.background {
            form = form.text(
                "background",
                if white_keying { "opaque" } else { background }.to_owned(),
            );
        }
        if let Some(fidelity) = &request.input_fidelity
            && !is_gpt_image_2(&model)
        {
            form = form.text("input_fidelity", fidelity.clone());
        }
        for path in &request.input_images {
            let content = std::fs::read(path)
                .map_err(|_| Error::State(format!("input image not found: {path}")))?;
            form = form.part(
                "image[]",
                reqwest::multipart::Part::bytes(content)
                    .file_name(path.rsplit('/').next().unwrap_or("image.png").to_owned()),
            );
        }
        client
            .post(request_url(&request.context, spec.base_url, "edits")?)
            .multipart(form)
    } else {
        if provider == "nanogpt" && !request.input_images.is_empty() {
            let urls = request
                .input_images
                .iter()
                .map(|path| {
                    if path.starts_with("data:") {
                        Ok(path.clone())
                    } else {
                        let bytes = std::fs::read(path)
                            .map_err(|_| Error::State(format!("input image not found: {path}")))?;
                        Ok(format!("data:image/png;base64,{}", STANDARD.encode(bytes)))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            body["extra_body"]["imageDataUrls"] = json!(urls);
        }
        client
            .post(request_url(&request.context, spec.base_url, "generations")?)
            .json(&body)
    };
    for (name, value) in headers {
        call = call.header(name, value);
    }
    let response = call.send().await?;
    if !response.status().is_success() {
        return Err(Error::HttpStatus {
            status: response.status().as_u16(),
            body: response.text().await.unwrap_or_default(),
        });
    }
    let response: Value = response.json().await?;
    let images = response
        .get("data")
        .and_then(Value::as_array)
        .filter(|items| items.len() == usize::from(request.count))
        .ok_or_else(|| Error::Event("image response did not include any images".into()))?;
    let job = ImageJob {
        provider: provider.into(),
        model,
        prompt: prompt.into(),
        size: size.value,
        width: size.width,
        height: size.height,
        mime_type: "image/png".into(),
        background: request.background,
        background_processing: white_keying.then_some("white_keying".into()),
    };
    let mut results = Vec::with_capacity(images.len());
    for image in images {
        results.push(
            image_result_from_response_image(job.clone(), image, &client, white_keying).await?,
        );
    }
    Ok(ImageBatchResult {
        images: results,
        usage: response.get("usage").cloned(),
    })
}

/// Generate or edit through Codex's existing Responses transport. Unlike the
/// direct Image API providers, Codex invokes the built-in image generation
/// tool and returns its base64 result as an output item.
async fn generate_codex_images(request: ImageRequest) -> Result<ImageBatchResult> {
    let count = request.count;
    let mut images = Vec::with_capacity(usize::from(count));
    for _ in 0..count {
        let mut one = request.clone();
        one.count = 1;
        images.push(generate_codex_image(one).await?);
    }
    Ok(ImageBatchResult {
        images,
        usage: None,
    })
}

async fn generate_codex_image(request: ImageRequest) -> Result<ImageResult> {
    let response_request = codex_image_response_request(&request)?;
    let mut machine =
        ResponseMachine::new(&ProviderRegistry::from_environment()?, response_request)?;
    let round = execute_round(&mut machine).await?;
    let NextAction::Completed { result } = round.next else {
        return Err(Error::Event(
            "Codex image request unexpectedly requested a client-side tool call".into(),
        ));
    };
    codex_image_result(&request, &result.model, &result.output_items)
}

fn codex_image_response_request(request: &ImageRequest) -> Result<ResponseRequest> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(Error::State("prompt must not be empty".into()));
    }
    if request.input_fidelity.is_some() {
        return Err(Error::UnsupportedCapability(
            "codex".into(),
            "input fidelity selection for Responses image generation",
        ));
    }
    if !request.extra_body.is_empty() {
        return Err(Error::UnsupportedCapability(
            "codex".into(),
            "extra image request fields for Responses image generation",
        ));
    }

    let action = if request.input_images.is_empty() {
        "generate"
    } else {
        "edit"
    };
    let mut image_tool = json!({"type": "image_generation", "action": action});
    if let Some(quality) = &request.quality {
        image_tool["quality"] = json!(quality);
    }
    if let Some(size) = codex_requested_size(request)? {
        image_tool["size"] = json!(size);
    }
    if let Some(background) = &request.background {
        image_tool["background"] = json!(background);
    }

    let mut content = vec![json!({"type": "input_text", "text": prompt})];
    for path in &request.input_images {
        content.push(json!({
            "type": "input_image",
            "image_url": image_path_data_url(path)?,
        }));
    }
    Ok(ResponseRequest {
        input: vec![serde_json::from_value(json!({
            "type": "message",
            "role": "user",
            "content": content,
        }))?],
        context: request.context.clone(),
        model: request.model.clone(),
        instructions: String::new(),
        tools: vec![image_tool],
        tool_choice: Some(json!("required")),
        reasoning_effort: None,
        text_verbosity: "low".into(),
        text_format: None,
    })
}

fn codex_requested_size(request: &ImageRequest) -> Result<Option<String>> {
    match (&request.size, request.width, request.height) {
        (Some(size), _, _) => Ok(Some(size.clone())),
        (None, Some(width), Some(height)) => Ok(Some(format!("{width}x{height}"))),
        (None, None, None) => Ok(None),
        _ => Err(Error::State(
            "image width and height must be provided together".into(),
        )),
    }
}

fn image_path_data_url(path: &str) -> Result<String> {
    let bytes =
        std::fs::read(path).map_err(|_| Error::State(format!("input image not found: {path}")))?;
    let extension = path
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let media_type = match extension.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    };
    Ok(format!(
        "data:{media_type};base64,{}",
        STANDARD.encode(bytes)
    ))
}

fn codex_image_result(
    request: &ImageRequest,
    model: &str,
    output_items: &[serde_json::Map<String, Value>],
) -> Result<ImageResult> {
    let result = output_items
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("image_generation_call"))
        .and_then(|item| item.get("result").and_then(Value::as_str))
        .filter(|result| !result.is_empty())
        .ok_or_else(|| {
            Error::Event("Codex response did not include an image generation result".into())
        })?;
    let job = codex_image_job(request, model)?;
    image_result_from_base64(job, result, false)
}

fn codex_image_job(request: &ImageRequest, model: &str) -> Result<ImageJob> {
    let size = codex_requested_size(request)?.unwrap_or_else(|| "auto".into());
    let (width, height) = if size == "auto" {
        (None, None)
    } else {
        let (width, height) = parse_dimensions(&size)?;
        (Some(width), Some(height))
    };
    Ok(ImageJob {
        provider: "codex".into(),
        model: model.into(),
        prompt: request.prompt.trim().into(),
        size,
        width,
        height,
        mime_type: "image/png".into(),
        background: request.background.clone(),
        background_processing: None,
    })
}

/// Stream partial images from the OpenAI-compatible Image API.
///
/// Image generation and editing use distinct request encodings, but both emit
/// the same SSE payload shape. Keeping that normalization here lets bindings
/// expose one stable image-streaming API.
pub async fn stream_image<F>(
    request: ImageStreamRequest,
    cancellation: &CancellationToken,
    mut on_event: F,
) -> Result<()>
where
    F: FnMut(ImageStreamEvent) -> Result<()>,
{
    if request.partial_images > 3 {
        return Err(Error::State(
            "partial_images must be between 0 and 3".into(),
        ));
    }
    let image_request = request.image;
    if !(1..=10).contains(&image_request.count) {
        return Err(Error::State("count must be between 1 and 10".into()));
    }
    if image_request.count > 1 {
        let batch = generate_images(image_request).await?;
        for (image_index, result) in batch.images.into_iter().enumerate() {
            on_event(ImageStreamEvent::Completed {
                image_index: image_index
                    .try_into()
                    .map_err(|_| Error::Event("image index exceeds u8".into()))?,
                result,
            })?;
        }
        on_event(ImageStreamEvent::BatchCompleted { usage: batch.usage })?;
        return Ok(());
    }
    if image_request.context.provider == Provider::Codex {
        return stream_codex_image(
            image_request,
            request.partial_images,
            cancellation,
            on_event,
        )
        .await;
    }
    let prompt = image_request.prompt.trim();
    if prompt.is_empty() {
        return Err(Error::State("prompt must not be empty".into()));
    }
    let provider = image_request.context.provider.as_str();
    if provider == "nanogpt" {
        return Err(Error::UnsupportedCapability(
            provider.into(),
            "streaming image generation",
        ));
    }
    let spec = image_spec(provider)?;
    let model = image_request
        .model
        .clone()
        .unwrap_or_else(|| spec.default_model.into());
    if !spec.models.contains(&model.as_str()) {
        return Err(Error::UnsupportedModel {
            provider: provider.into(),
            model,
        });
    }
    if !model.starts_with("gpt-image-") {
        return Err(Error::UnsupportedCapability(
            provider.into(),
            "streaming image generation for this model",
        ));
    }
    let size = resolve_size(&image_request, &model, spec.sizes)?;
    let white_keying =
        image_request.background.as_deref() == Some("transparent") && is_gpt_image_2(&model);
    let provider_prompt = if white_keying {
        prompt_for_white_key(prompt)
    } else {
        prompt.into()
    };
    let provider_size = size.value.clone();
    let quality = image_request
        .quality
        .clone()
        .unwrap_or_else(|| "high".into());
    let background = image_request
        .background
        .as_ref()
        .map(|background| if white_keying { "opaque" } else { background }.to_owned());
    let job = ImageJob {
        provider: provider.into(),
        model: model.clone(),
        prompt: prompt.into(),
        size: size.value,
        width: size.width,
        height: size.height,
        mime_type: "image/png".into(),
        background: image_request.background.clone(),
        background_processing: white_keying.then_some("white_keying".into()),
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let mut headers = image_request.context.headers.clone();
    headers.insert(
        "Authorization".into(),
        format!("Bearer {}", image_request.context.api_key),
    );
    headers
        .entry("User-Agent".into())
        .or_insert_with(|| format!("lmx/{}", crate::VERSION));
    let mut call = if image_request.input_images.is_empty() {
        let mut body = json!({
            "model": model,
            "prompt": provider_prompt,
            "size": provider_size,
            "n": image_request.count,
            "quality": quality,
            "stream": true,
            "partial_images": request.partial_images,
        });
        if let Some(background) = &background {
            body["background"] = json!(background);
        }
        client
            .post(request_url(
                &image_request.context,
                spec.base_url,
                "generations",
            )?)
            .json(&body)
    } else {
        let mut form = reqwest::multipart::Form::new()
            .text("model", model)
            .text("prompt", provider_prompt)
            .text("size", provider_size)
            .text("n", image_request.count.to_string())
            .text("quality", quality)
            .text("stream", "true")
            .text("partial_images", request.partial_images.to_string());
        if let Some(background) = background {
            form = form.text("background", background);
        }
        if let Some(fidelity) = &image_request.input_fidelity
            && !is_gpt_image_2(&job.model)
        {
            form = form.text("input_fidelity", fidelity.clone());
        }
        for path in &image_request.input_images {
            let content = std::fs::read(path)
                .map_err(|_| Error::State(format!("input image not found: {path}")))?;
            form = form.part(
                "image[]",
                reqwest::multipart::Part::bytes(content)
                    .file_name(path.rsplit('/').next().unwrap_or("image.png").to_owned()),
            );
        }
        client
            .post(request_url(&image_request.context, spec.base_url, "edits")?)
            .multipart(form)
    };
    for (name, value) in headers {
        call = call.header(name, value);
    }
    let response = tokio::select! {
        _ = cancellation.cancelled() => return Err(Error::Cancelled),
        response = call.send() => response?,
    };
    if !response.status().is_success() {
        return Err(Error::HttpStatus {
            status: response.status().as_u16(),
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
        let payload: Value = serde_json::from_str(&event.data)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or(&event.event);
        let encoded = payload
            .get("b64_json")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        match event_type {
            "image_generation.partial_image" | "image_edit.partial_image" => {
                let encoded = encoded.ok_or_else(|| {
                    Error::Event("partial image event did not include b64_json".into())
                })?;
                let index = payload
                    .get("partial_image_index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        Error::Event(
                            "partial image event did not include partial_image_index".into(),
                        )
                    })?
                    .try_into()
                    .map_err(|_| Error::Event("partial image index exceeds u8".into()))?;
                on_event(ImageStreamEvent::Partial {
                    image_index: 0,
                    partial_index: index,
                    result: image_result_from_base64(job.clone(), encoded, white_keying)?,
                })?;
            }
            "image_generation.completed" | "image_edit.completed" => {
                let encoded = encoded.ok_or_else(|| {
                    Error::Event("completed image event did not include b64_json".into())
                })?;
                on_event(ImageStreamEvent::Completed {
                    image_index: 0,
                    result: image_result_from_base64(job.clone(), encoded, white_keying)?,
                })?;
                on_event(ImageStreamEvent::BatchCompleted {
                    usage: payload.get("usage").cloned(),
                })?;
                return Ok(());
            }
            _ => {}
        }
    }
    Err(Error::Event(
        "image stream ended before a completed event".into(),
    ))
}

async fn stream_codex_image<F>(
    request: ImageRequest,
    partial_images: u8,
    cancellation: &CancellationToken,
    mut on_event: F,
) -> Result<()>
where
    F: FnMut(ImageStreamEvent) -> Result<()>,
{
    let mut response_request = codex_image_response_request(&request)?;
    response_request.tools[0]["partial_images"] = json!(partial_images);
    let mut machine =
        ResponseMachine::new(&ProviderRegistry::from_environment()?, response_request)?;
    let job = codex_image_job(&request, machine.model())?;
    let next = execute_round_with_observer(&mut machine, cancellation, |event| {
        if let CoreEvent::ImageGenerationPartial {
            partial_image_index,
            partial_image_base64,
        } = event
        {
            on_event(ImageStreamEvent::Partial {
                image_index: 0,
                partial_index: partial_image_index,
                result: image_result_from_base64(job.clone(), &partial_image_base64, false)?,
            })?;
        }
        Ok(())
    })
    .await?;
    let NextAction::Completed { result } = next else {
        return Err(Error::Event(
            "Codex image request unexpectedly requested a client-side tool call".into(),
        ));
    };
    on_event(ImageStreamEvent::Completed {
        image_index: 0,
        result: codex_image_result(&request, &result.model, &result.output_items)?,
    })?;
    on_event(ImageStreamEvent::BatchCompleted { usage: None })
}

async fn image_result_from_response_image(
    mut job: ImageJob,
    image: &Value,
    client: &reqwest::Client,
    white_keying: bool,
) -> Result<ImageResult> {
    let (mut content, content_type) = if let Some(encoded) = image
        .get("b64_json")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        (
            STANDARD
                .decode(encoded)
                .map_err(|error| Error::Event(format!("invalid base64 image payload: {error}")))?,
            "image/png".into(),
        )
    } else if let Some(url) = image
        .get("url")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        let download = client.get(url).send().await?;
        if !download.status().is_success() {
            return Err(Error::HttpStatus {
                status: download.status().as_u16(),
                body: download.text().await.unwrap_or_default(),
            });
        };
        let content_type = download
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("image/png")
            .to_owned();
        (download.bytes().await?.to_vec(), content_type)
    } else {
        return Err(Error::Event(
            "image response did not include b64_json or url content".into(),
        ));
    };
    let content_type = if white_keying {
        content = remove_white_key_background(&content)?;
        "image/png".into()
    } else {
        content_type
    };
    job.mime_type = content_type.clone();
    Ok(ImageResult {
        job,
        content_base64: STANDARD.encode(content),
        content_type,
    })
}

fn image_result_from_base64(
    job: ImageJob,
    encoded: &str,
    white_keying: bool,
) -> Result<ImageResult> {
    let mut content = STANDARD
        .decode(encoded)
        .map_err(|error| Error::Event(format!("invalid base64 image payload: {error}")))?;
    let content_type = if white_keying {
        content = remove_white_key_background(&content)?;
        "image/png".into()
    } else {
        job.mime_type.clone()
    };
    Ok(ImageResult {
        job,
        content_base64: STANDARD.encode(content),
        content_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Provider;

    fn request(size: Option<&str>, width: Option<u32>, height: Option<u32>) -> ImageRequest {
        ImageRequest {
            prompt: "test".into(),
            context: RequestContext {
                provider: Provider::Azure,
                api_key: "test".into(),
                base_url: None,
                headers: BTreeMap::new(),
                query: BTreeMap::new(),
            },
            model: None,
            size: size.map(str::to_owned),
            width,
            height,
            quality: None,
            background: None,
            input_images: Vec::new(),
            input_fidelity: None,
            count: 1,
            extra_body: BTreeMap::new(),
        }
    }

    #[test]
    fn gpt_image_2_accepts_auto_and_constrained_dimensions() {
        let automatic = resolve_size(
            &request(Some("auto"), None, None),
            "gpt-image-2",
            OPENAI_SIZES,
        )
        .unwrap();
        assert_eq!(automatic.value, "auto");
        assert_eq!(automatic.width, None);
        assert_eq!(automatic.height, None);

        for size in ["2048x2048", "3840x2160", "2160x3840"] {
            let resolved = resolve_size(
                &request(Some(size), None, None),
                "gpt-image-2",
                OPENAI_SIZES,
            )
            .unwrap();
            assert_eq!(resolved.value, size);
            assert_eq!(
                resolved.width.zip(resolved.height),
                size.split_once('x')
                    .map(|(w, h)| (w.parse().unwrap(), h.parse().unwrap()))
            );
        }
    }

    #[test]
    fn gpt_image_2_keeps_the_existing_default_size() {
        let resolved =
            resolve_size(&request(None, None, None), "gpt-image-2", OPENAI_SIZES).unwrap();
        assert_eq!(resolved.value, "1024x1024");
        assert_eq!(resolved.width, Some(1024));
        assert_eq!(resolved.height, Some(1024));
    }

    #[test]
    fn gpt_image_2_rejects_dimensions_outside_its_constraints() {
        for size in [
            "1025x1024",
            "4096x1024",
            "1024x4096",
            "512x512",
            "3840x2176",
        ] {
            let error = resolve_size(
                &request(Some(size), None, None),
                "gpt-image-2",
                OPENAI_SIZES,
            )
            .unwrap_err();
            assert!(error.to_string().contains("unsupported gpt-image-2 size"));
        }
    }

    #[test]
    fn other_models_keep_their_fixed_size_allowlists() {
        let resolved =
            resolve_size(&request(None, None, None), "gpt-image-1", OPENAI_SIZES).unwrap();
        assert_eq!(resolved.value, "1024x1024");

        let error = resolve_size(
            &request(Some("2048x2048"), None, None),
            "gpt-image-1",
            OPENAI_SIZES,
        )
        .unwrap_err();
        assert!(error.to_string().contains("unsupported image size"));
    }

    #[test]
    fn codex_images_use_the_responses_image_tool_and_normalize_its_result() {
        let request = ImageRequest {
            prompt: "A cobalt-blue square".into(),
            context: RequestContext {
                provider: Provider::Codex,
                api_key: "request-token".into(),
                base_url: None,
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), "account-123".into())]),
                query: BTreeMap::new(),
            },
            model: Some("gpt-5.6-sol".into()),
            size: Some("1024x1024".into()),
            width: None,
            height: None,
            quality: Some("high".into()),
            background: None,
            input_images: Vec::new(),
            input_fidelity: None,
            count: 1,
            extra_body: BTreeMap::new(),
        };
        let response_request = codex_image_response_request(&request).unwrap();
        let wire = ResponseMachine::new(&ProviderRegistry::default(), response_request)
            .unwrap()
            .wire_request()
            .unwrap();
        assert_eq!(wire.url, "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(wire.body["tool_choice"], "required");
        assert_eq!(wire.body["tools"][0]["type"], "image_generation");
        assert_eq!(wire.body["tools"][0]["action"], "generate");
        assert_eq!(wire.body["tools"][0]["size"], "1024x1024");
        assert_eq!(wire.body["input"][0]["content"][0]["type"], "input_text");

        let output = serde_json::from_value(json!({
            "type": "image_generation_call",
            "result": STANDARD.encode(b"generated-image"),
        }))
        .unwrap();
        let image = codex_image_result(&request, "gpt-5.6-sol", &[output]).unwrap();
        assert_eq!(
            STANDARD.decode(image.content_base64).unwrap(),
            b"generated-image"
        );
        assert_eq!(image.job.provider, "codex");
    }
}
