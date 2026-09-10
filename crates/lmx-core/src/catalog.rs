//! Account-scoped model discovery. Credentials never enter the process-wide registry.
use crate::{Error, Provider, RequestContext, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex as StdMutex, OnceLock},
    time::{Duration, Instant},
};
use tokio::sync::Mutex;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    pub id: String,
    pub display_name: String,
    pub context_window: Option<u64>,
    pub compact_threshold: Option<u64>,
    pub reasoning_efforts: Vec<String>,
    pub default_reasoning_effort: Option<String>,
    pub supports_images: bool,
    pub supports_tool_search: bool,
}

#[derive(Deserialize)]
struct CodexCatalog {
    models: Vec<CodexModel>,
}
#[derive(Deserialize)]
struct ReasoningLevel {
    effort: String,
}
#[derive(Deserialize)]
struct CodexModel {
    slug: String,
    display_name: String,
    visibility: String,
    context_window: Option<u64>,
    auto_compact_token_limit: Option<u64>,
    supported_reasoning_levels: Vec<ReasoningLevel>,
    default_reasoning_level: Option<String>,
    #[serde(default)]
    supports_search_tool: bool,
    #[serde(default)]
    input_modalities: Vec<String>,
}

fn codex_models(catalog: CodexCatalog) -> Vec<ModelSpec> {
    catalog
        .models
        .into_iter()
        .filter(|m| m.visibility == "list")
        .map(|m| {
            let limit = m.context_window.map(|n| n * 9 / 10);
            let compact_threshold = match (m.auto_compact_token_limit, limit) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            ModelSpec {
                id: m.slug,
                display_name: m.display_name,
                context_window: m.context_window,
                compact_threshold,
                // Ultra is orchestration, not a single-response reasoning effort.
                reasoning_efforts: m
                    .supported_reasoning_levels
                    .into_iter()
                    .map(|r| r.effort)
                    .filter(|e| e != "ultra")
                    .collect(),
                default_reasoning_effort: m.default_reasoning_level,
                supports_images: m.input_modalities.iter().any(|m| m == "image"),
                supports_tool_search: m.supports_search_tool,
            }
        })
        .collect()
}

#[derive(Deserialize)]
struct ApiCatalog {
    data: Vec<ApiModel>,
}
#[derive(Deserialize)]
struct ApiModel {
    id: String,
}

// API discovery is restricted to general-purpose GPT assistant models from 5.3 on.
fn assistant_model(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("gpt-") else {
        return false;
    };
    let version = rest.split('-').next().unwrap_or_default();
    let mut numbers = version.split('.');
    let Ok(major) = numbers.next().unwrap_or_default().parse::<u32>() else {
        return false;
    };
    let minor = numbers.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    (major > 5 || (major == 5 && minor >= 3))
        && !["chat", "pro", "audio", "realtime", "search", "cyber"]
            .iter()
            .any(|s| rest.split('-').any(|p| p == *s))
}

type CatalogEntry = Arc<Mutex<Option<(Instant, Vec<ModelSpec>)>>>;
type CatalogCache = BTreeMap<Vec<u8>, CatalogEntry>;
static CACHE: OnceLock<StdMutex<CatalogCache>> = OnceLock::new();
const TTL: Duration = Duration::from_secs(300);

pub async fn discover_models(context: &RequestContext) -> Result<Vec<ModelSpec>> {
    if !matches!(context.provider, Provider::Codex | Provider::Openai) {
        return Err(Error::UnsupportedCapability(
            context.provider.as_str().into(),
            "model discovery",
        ));
    }
    crate::response::validate_request_context(context)?;
    let key = Sha256::digest(serde_json::to_vec(context)?).to_vec();
    let entry = {
        let mut cache = CACHE
            .get_or_init(|| StdMutex::new(BTreeMap::new()))
            .lock()
            .map_err(|_| Error::State("model catalog cache is unavailable".into()))?;
        cache.retain(|_, entry| {
            Arc::strong_count(entry) > 1
                || entry
                    .try_lock()
                    .map(|v| v.as_ref().is_some_and(|(at, _)| at.elapsed() < TTL))
                    .unwrap_or(true)
        });
        cache
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone()
    };
    // Coalesce requests for one identity without blocking discovery for other accounts.
    let mut cached = entry.lock().await;
    if let Some((at, models)) = cached.as_ref()
        && at.elapsed() < TTL
    {
        return Ok(models.clone());
    }
    let base = match context.provider {
        Provider::Codex => "https://chatgpt.com/backend-api/codex",
        _ => context
            .base_url
            .as_deref()
            .unwrap_or("https://api.openai.com/v1"),
    };
    let mut query = context.query.clone();
    if context.provider == Provider::Codex {
        // The upstream catalog requires the supported Codex client release identity.
        query
            .entry("client_version".into())
            .or_insert_with(|| "0.153.4".into());
    }
    let url = crate::endpoint_url(base, "models", &query)?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()?;
    let mut request = client.get(url).bearer_auth(&context.api_key);
    for (name, value) in &context.headers {
        request = request.header(name, value);
    }
    let response = request.send().await?.error_for_status()?;
    let models: Vec<ModelSpec> = if context.provider == Provider::Codex {
        codex_models(response.json().await?)
    } else {
        let catalog: ApiCatalog = response.json().await?;
        let mut models: Vec<_> = catalog
            .data
            .into_iter()
            .filter(|m| assistant_model(&m.id))
            .map(|m| ModelSpec {
                display_name: m.id.clone(),
                id: m.id,
                context_window: None,
                compact_threshold: None,
                reasoning_efforts: vec![],
                default_reasoning_effort: None,
                supports_images: true,
                supports_tool_search: false,
            })
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models.dedup_by(|a, b| a.id == b.id);
        models
    };
    if models.is_empty() {
        return Err(Error::State(format!(
            "{} discovery returned no assistant models",
            context.provider.as_str()
        )));
    }
    *cached = Some((Instant::now(), models.clone()));
    Ok(models)
}

pub async fn prepare_response_request(
    request: crate::ResponseRequest,
) -> Result<crate::ResponseRequest> {
    if request.context.provider != Provider::Codex {
        return Ok(request);
    }
    let models = discover_models(&request.context).await?;
    apply_model_defaults(request, &models)
}

fn apply_model_defaults(
    mut request: crate::ResponseRequest,
    models: &[ModelSpec],
) -> Result<crate::ResponseRequest> {
    let configured = crate::ProviderRegistry::from_environment()?
        .get(&Provider::Codex)?
        .default_model
        .clone();
    let model = match request.model.as_deref() {
        Some(id) => models.iter().find(|m| m.id == id),
        None => models
            .iter()
            .find(|m| m.id == configured)
            .or_else(|| models.first()),
    }
    .ok_or_else(|| Error::UnsupportedModel {
        provider: "codex".into(),
        model: request.model.clone().unwrap_or_default(),
    })?;
    request.model = Some(model.id.clone());
    if let Some(effort) = &request.reasoning_effort {
        if !model.reasoning_efforts.contains(effort) {
            return Err(Error::State(format!(
                "model {} does not support reasoning effort {effort}",
                model.id
            )));
        }
    } else {
        request.reasoning_effort = model.default_reasoning_effort.clone();
    }
    if request.codex_protocol.is_none() {
        request.codex_protocol = Some(match request.context_management {
            Some(crate::ContextManagement::RemoteV2 { .. }) => crate::CodexProtocol::ResponsesLite,
            _ => crate::CodexProtocol::ResponsesStandard,
        });
    }
    if let Some(crate::ContextManagement::RemoteV2 {
        auto_compact_token_limit,
        ..
    }) = &mut request.context_management
        && auto_compact_token_limit.is_none()
    {
        *auto_compact_token_limit = model.compact_threshold;
    }
    if request.context_management.is_none()
        && request.codex_protocol == Some(crate::CodexProtocol::ResponsesStandard)
    {
        request.context_management = model
            .compact_threshold
            .map(|compact_threshold| crate::ContextManagement::Server { compact_threshold });
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_models_and_specialized_models() {
        for id in ["gpt-5.3-codex", "gpt-6-astra", "gpt-7", "gpt-12.2-mini"] {
            assert!(assistant_model(id), "{id}");
        }
        for id in [
            "gpt-5.2",
            "gpt-image-2",
            "gpt-6-audio",
            "gpt-5.3-chat-latest",
            "gpt-6-pro",
            "text-embedding-3-large",
        ] {
            assert!(!assistant_model(id), "{id}");
        }
    }
    #[test]
    fn catalog_controls_visibility_limits_and_efforts() {
        let input = serde_json::json!({"models":[
            {"slug":"gpt-7","display_name":"Next","visibility":"list","context_window":300000,"auto_compact_token_limit":290000,"supported_reasoning_levels":[{"effort":"max"},{"effort":"ultra"}],"default_reasoning_level":"max","supports_search_tool":true,"input_modalities":["text","image"]},
            {"slug":"private","display_name":"Private","visibility":"hide","context_window":100000,"supported_reasoning_levels":[]}
        ]});
        let models = codex_models(serde_json::from_value(input).unwrap());
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].compact_threshold, Some(270000));
        assert_eq!(models[0].reasoning_efforts, ["max"]);
        assert!(models[0].supports_tool_search);
        assert!(models[0].supports_images);
    }

    #[test]
    fn discovered_future_model_gets_standard_compaction_without_name_gates() {
        let model = ModelSpec {
            id: "gpt-7-next".into(),
            display_name: "Next".into(),
            context_window: Some(500000),
            compact_threshold: Some(450000),
            reasoning_efforts: vec!["max".into()],
            default_reasoning_effort: Some("max".into()),
            supports_images: true,
            supports_tool_search: true,
        };
        let request: crate::ResponseRequest = serde_json::from_value(serde_json::json!({
            "context": {"provider":"codex", "apiKey":"test", "headers":{"ChatGPT-Account-ID":"account"}},
            "model":"gpt-7-next", "input":[], "instructions":"Work", "tools":[{"type":"function","name":"work"}]
        })).unwrap();
        let prepared = apply_model_defaults(request.clone(), std::slice::from_ref(&model)).unwrap();
        let wire = crate::ResponseMachine::new(&crate::ProviderRegistry::default(), prepared)
            .unwrap()
            .wire_request()
            .unwrap();
        assert_eq!(
            wire.body["context_management"][0]["compact_threshold"],
            450000
        );
        assert_eq!(wire.body["reasoning"]["effort"], "max");
        assert_eq!(wire.body["tools"][0]["name"], "work");
        assert!(
            !wire
                .headers
                .contains_key("X-OpenAI-Internal-Codex-Responses-Lite")
        );
        let mut explicit = request.clone();
        explicit.context_management = Some(crate::ContextManagement::Server {
            compact_threshold: 10000,
        });
        let explicit = apply_model_defaults(explicit, std::slice::from_ref(&model)).unwrap();
        assert!(matches!(
            explicit.context_management,
            Some(crate::ContextManagement::Server {
                compact_threshold: 10000
            })
        ));
        let mut invalid = request;
        invalid.reasoning_effort = Some("ultra".into());
        assert!(apply_model_defaults(invalid, &[model]).is_err());
    }

    #[tokio::test]
    async fn discovery_is_cached_and_isolated_by_credentials() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            for index in 0..2 {
                let (mut socket, _) = listener.accept().unwrap();
                let mut buffer = [0; 8192];
                let length = socket.read(&mut buffer).unwrap();
                let text = String::from_utf8_lossy(&buffer[..length]);
                assert!(text.starts_with("GET /v1/models"));
                assert!(text.contains(if index == 0 {
                    "Bearer first"
                } else {
                    "Bearer second"
                }));
                let body = format!("{{\"data\":[{{\"id\":\"gpt-7-account-{index}\"}}]}}");
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
            }
        });
        let mut context = RequestContext {
            provider: Provider::Openai,
            api_key: "first".into(),
            base_url: Some(format!("http://{address}/v1")),
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
        };
        let (first, concurrent) =
            tokio::join!(discover_models(&context), discover_models(&context));
        assert_eq!(first.unwrap()[0].id, "gpt-7-account-0");
        assert_eq!(concurrent.unwrap()[0].id, "gpt-7-account-0");
        assert_eq!(
            discover_models(&context).await.unwrap()[0].id,
            "gpt-7-account-0"
        );
        context.api_key = "second".into();
        assert_eq!(
            discover_models(&context).await.unwrap()[0].id,
            "gpt-7-account-1"
        );
        worker.join().unwrap();
    }
}
