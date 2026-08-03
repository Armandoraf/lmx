use std::collections::BTreeMap;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    Error, RequestContext, Result, endpoint_url, prompt_for_chroma_key,
    remove_chroma_key_background,
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
    #[serde(default)]
    pub extra_body: BTreeMap<String, Value>,
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
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(Error::State("prompt must not be empty".into()));
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
    let chroma_key = request.background.as_deref() == Some("transparent") && is_gpt_image_2(&model);
    let provider_prompt = if chroma_key {
        prompt_for_chroma_key(prompt)
    } else {
        prompt.into()
    };
    let provider_size = match model.as_str() {
        "qwen-image" => "auto".into(),
        "z-image-turbo" => "1024*1024".into(),
        _ => size.value.clone(),
    };
    let mut body =
        json!({"model": model, "prompt": provider_prompt, "size": provider_size, "n": 1});
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
            body["background"] = json!(if chroma_key { "opaque" } else { background });
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
            .text("n", "1")
            .text(
                "quality",
                request.quality.clone().unwrap_or_else(|| "high".into()),
            );
        if let Some(background) = &request.background {
            form = form.text(
                "background",
                if chroma_key { "opaque" } else { background }.to_owned(),
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
    let image = response
        .get("data")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .ok_or_else(|| Error::Event("image response did not include any images".into()))?;
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
    let content_type = if chroma_key {
        content = remove_chroma_key_background(&content)?;
        "image/png".into()
    } else {
        content_type
    };
    Ok(ImageResult {
        job: ImageJob {
            provider: provider.into(),
            model,
            prompt: prompt.into(),
            size: size.value,
            width: size.width,
            height: size.height,
            mime_type: content_type.clone(),
            background: request.background,
            background_processing: chroma_key.then_some("chroma_key".into()),
        },
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
}
