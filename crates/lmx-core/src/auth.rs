use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{Value, json};

use crate::{Error, Provider, RequestContext, Result};

const REFRESH_URL: &str = "https://auth.openai.com/oauth/token";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

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

fn codex_auth_path() -> PathBuf {
    if let Some(path) = std::env::var_os("CODEX_AUTH_PATH") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("LMX_HOME")
        .or_else(|| std::env::var_os("LLMX_HOME"))
        .unwrap_or_else(|| ".lmx".into());
    PathBuf::from(home).join("auth.json")
}

pub fn load_request_context(provider: Provider) -> Result<RequestContext> {
    match provider {
        Provider::Codex => {
            let path = codex_auth_path();
            let raw = std::fs::read_to_string(&path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    Error::State(format!("auth file not found: {}", path.display()))
                } else {
                    Error::Io(error)
                }
            })?;
            let document: Value = serde_json::from_str(&raw)?;
            if document.get("auth_mode").and_then(Value::as_str) == Some("apikey") {
                return Err(Error::State(format!(
                    "{} contains API-key auth, not ChatGPT OAuth tokens",
                    path.display()
                )));
            }
            let tokens = document
                .get("tokens")
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    Error::State(format!(
                        "{} does not contain valid ChatGPT OAuth tokens",
                        path.display()
                    ))
                })?;
            let access_token = tokens
                .get("access_token")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| Error::State("Codex auth file has no access token".into()))?;
            let account_id = tokens
                .get("account_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| Error::State("Codex auth file has no account id".into()))?;
            Ok(RequestContext {
                provider,
                api_key: access_token.into(),
                base_url: Some("https://chatgpt.com/backend-api/codex".into()),
                headers: BTreeMap::from([("ChatGPT-Account-ID".into(), account_id.into())]),
                query: BTreeMap::new(),
            })
        }
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
            let api_key = std::env::var("AZURE_OPENAI_API_KEY").or_else(|_| std::env::var("AZURE_OPENAI_KEY")).map_err(|_| Error::State("AZURE_OPENAI_API_KEY or AZURE_OPENAI_KEY is required when provider 'azure' is used".into()))?;
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
        Provider::Bedrock => Ok(RequestContext {
            provider,
            api_key: String::new(),
            base_url: None,
            headers: BTreeMap::new(),
            query: BTreeMap::new(),
        }),
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

pub async fn refresh_codex_auth() -> Result<RequestContext> {
    let path = codex_auth_path();
    let raw = std::fs::read_to_string(&path)?;
    let mut document: Value = serde_json::from_str(&raw)?;
    let refresh_token = document
        .pointer("/tokens/refresh_token")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::State("Codex auth file has no refresh token".into()))?
        .to_owned();
    let response = reqwest::Client::new().post(REFRESH_URL).header("Content-Type", "application/json").json(&json!({"client_id":CLIENT_ID,"grant_type":"refresh_token","refresh_token":refresh_token})).send().await?;
    if !response.status().is_success() {
        return Err(Error::HttpStatus {
            status: response.status().as_u16(),
            body: response.text().await.unwrap_or_default(),
        });
    }
    let refreshed: Value = response.json().await?;
    let tokens = document
        .pointer_mut("/tokens")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| Error::State("Codex auth file has invalid tokens".into()))?;
    for field in ["access_token", "refresh_token", "id_token"] {
        if let Some(value) = refreshed.get(field).filter(|value| value.is_string()) {
            tokens.insert(field.into(), value.clone());
        }
    }
    document["last_refresh"] = json!(chrono_like_timestamp());
    std::fs::write(
        &path,
        format!("{}\n", serde_json::to_string_pretty(&document)?),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    load_request_context(Provider::Codex)
}

fn chrono_like_timestamp() -> String {
    format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )
}
