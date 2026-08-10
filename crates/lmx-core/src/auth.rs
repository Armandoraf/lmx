use std::collections::BTreeMap;

use crate::{Error, Provider, RequestContext, Result};

pub fn normalize_azure_endpoint(endpoint: &str) -> Result<String> {
    let mut cleaned = endpoint.trim().trim_end_matches('/').to_owned();
    if cleaned.is_empty() {
        return Err(Error::State("azure endpoint must not be empty".into()));
    }
    if cleaned.contains(".cognitiveservices.azure.com") {
        cleaned = cleaned.replace(".cognitiveservices.azure.com", ".openai.azure.com");
    }
    if cleaned.ends_with("/openai") {
        cleaned.push_str("/v1");
    }
    if !cleaned.ends_with("/openai/v1") && cleaned.contains(".openai.azure.com") {
        cleaned.push_str("/openai/v1");
    }
    if !cleaned.ends_with("/openai/v1") {
        return Err(Error::State(
            "azure provider requires an /openai/v1 endpoint".into(),
        ));
    }
    Ok(cleaned)
}

/// Build a request context using environment-backed provider credentials.
///
/// Codex deliberately has no implicit credential source: callers must provide
/// a request-scoped context with their host-managed OAuth credentials.
pub fn load_request_context(provider: Provider) -> Result<RequestContext> {
    match provider {
        Provider::Codex => Err(Error::State(
            "provider 'codex' requires a request-scoped context; LMX never reads, persists, or refreshes ChatGPT OAuth credentials".into(),
        )),
        Provider::Openai => env_context(
            provider,
            "OPENAI_API_KEY",
            Some("https://api.openai.com/v1".into()),
        ),
        Provider::Nanogpt => env_context(
            provider,
            "NANOGPT_API_KEY",
            Some(
                std::env::var("NANOGPT_BASE_URL")
                    .unwrap_or_else(|_| "https://nano-gpt.com/api/v1".into()),
            ),
        ),
        Provider::Azure => {
            let api_key = std::env::var("AZURE_OPENAI_API_KEY")
                .or_else(|_| std::env::var("AZURE_OPENAI_KEY"))
                .map_err(|_| {
                    Error::State(
                        "AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY is required when provider 'azure' is used".into(),
                    )
                })?;
            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
                Error::State(
                    "AZURE_OPENAI_ENDPOINT is required when provider 'azure' is used".into(),
                )
            })?;
            let mut query = BTreeMap::new();
            if let Ok(version) = std::env::var("AZURE_OPENAI_API_VERSION")
                && !version.trim().is_empty()
            {
                query.insert("api-version".into(), version);
            }
            Ok(RequestContext {
                provider,
                api_key: api_key.clone(),
                base_url: Some(normalize_azure_endpoint(&endpoint)?),
                headers: BTreeMap::from([("api-key".into(), api_key)]),
                query,
            })
        }
    }
}

fn env_context(provider: Provider, key: &str, base_url: Option<String>) -> Result<RequestContext> {
    let api_key = std::env::var(key).map_err(|_| {
        Error::State(format!(
            "{key} is required when provider '{}' is used",
            provider.as_str()
        ))
    })?;
    if api_key.trim().is_empty() {
        return Err(Error::State(format!(
            "{key} is required when provider '{}' is used",
            provider.as_str()
        )));
    }
    Ok(RequestContext {
        provider,
        api_key,
        base_url,
        headers: BTreeMap::new(),
        query: BTreeMap::new(),
    })
}
