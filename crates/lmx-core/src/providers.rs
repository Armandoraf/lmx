use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Codex,
    Openai,
    Nanogpt,
    Azure,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Openai => "openai",
            Self::Nanogpt => "nanogpt",
            Self::Azure => "azure",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub supports_structured_output: bool,
    pub supports_streaming: bool,
    pub supports_images: bool,
    pub supports_pdf: bool,
    pub supports_reasoning: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSpec {
    pub provider: Provider,
    pub default_model: String,
    pub available_models: Vec<String>,
    pub base_url: Option<String>,
    pub capabilities: ProviderCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestContext {
    pub provider: Provider,
    pub api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub query: BTreeMap<String, String>,
}

impl RequestContext {
    pub fn resolved_base_url(&self, spec: &ProviderSpec) -> Result<String> {
        self.base_url
            .clone()
            .or_else(|| spec.base_url.clone())
            .ok_or_else(|| Error::MissingBaseUrl(self.provider.as_str().to_owned()))
    }
}

#[derive(Clone, Debug)]
pub struct ProviderRegistry {
    specs: BTreeMap<Provider, ProviderSpec>,
}

const DISCOVERY_TTL: Duration = Duration::from_secs(300);
static DISCOVERED_REGISTRY: OnceLock<Mutex<Option<(Instant, ProviderRegistry)>>> = OnceLock::new();

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::with_models(
            vec![
                "gpt-5.6-sol".into(),
                "gpt-5.6-terra".into(),
                "gpt-5.6-luna".into(),
            ],
            vec!["gpt-5.5".into(), "gpt-5.5-mini".into()],
            vec![
                "moonshotai/kimi-k2.6".into(),
                "zai-org/glm-5.1".into(),
                "deepseek/deepseek-v3.2".into(),
            ],
            vec!["gpt-5-mini".into(), "gpt-5-nano".into(), "gpt-5.5".into()],
        )
    }
}

impl ProviderRegistry {
    pub fn from_environment() -> Result<Self> {
        if let Some((_, registry)) = DISCOVERED_REGISTRY
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| Error::State("provider registry cache is unavailable".into()))?
            .clone()
        {
            return Ok(registry);
        }
        Self::from_environment_uncached()
    }

    fn from_environment_uncached() -> Result<Self> {
        let codex_models = models_from_environment(
            "CODEX_MODELS",
            vec![
                "gpt-5.6-sol".into(),
                "gpt-5.6-terra".into(),
                "gpt-5.6-luna".into(),
            ],
        )?;
        let openai_models = models_from_environment(
            "OPENAI_MODELS",
            vec![
                std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5.5".into()),
                "gpt-5.5-mini".into(),
            ],
        )?;
        let nanogpt_default =
            std::env::var("NANOGPT_MODEL").unwrap_or_else(|_| "moonshotai/kimi-k2.6".into());
        let nanogpt_models = models_from_environment(
            "NANOGPT_MODELS",
            vec![
                nanogpt_default,
                "moonshotai/kimi-k2.6".into(),
                "zai-org/glm-5.1".into(),
                "deepseek/deepseek-v3.2".into(),
            ],
        )?;
        let azure_models = models_from_environment(
            "AZURE_OPENAI_MODELS",
            vec![
                std::env::var("AZURE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-5-mini".into()),
                "gpt-5-mini".into(),
                "gpt-5-nano".into(),
                "gpt-5.5".into(),
            ],
        )?;
        Ok(Self::with_models(
            codex_models,
            openai_models,
            nanogpt_models,
            azure_models,
        ))
    }

    /// Fetch the account-visible language models and cache the resulting registry for
    /// subsequent response requests in this process.
    pub async fn discover() -> Result<Self> {
        if let Some((discovered_at, registry)) = DISCOVERED_REGISTRY
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| Error::State("provider registry cache is unavailable".into()))?
            .clone()
            && discovered_at.elapsed() < DISCOVERY_TTL
        {
            return Ok(registry);
        }
        let mut registry = Self::from_environment_uncached()?;
        if std::env::var_os("NANOGPT_API_KEY").is_some() {
            registry.replace_models(Provider::Nanogpt, discover_nanogpt_models().await?)?;
        }
        *DISCOVERED_REGISTRY
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| Error::State("provider registry cache is unavailable".into()))? =
            Some((Instant::now(), registry.clone()));
        Ok(registry)
    }

    fn replace_models(&mut self, provider: Provider, models: Vec<String>) -> Result<()> {
        let models = deduplicate_models(models);
        let default_model = models.first().cloned().ok_or_else(|| {
            Error::State(format!(
                "{provider:?} discovery returned no language models"
            ))
        })?;
        let spec = self
            .specs
            .get_mut(&provider)
            .ok_or_else(|| Error::UnknownProvider(provider.as_str().into()))?;
        spec.default_model = default_model;
        spec.available_models = models;
        Ok(())
    }

    fn with_models(
        codex_models: Vec<String>,
        openai_models: Vec<String>,
        nanogpt_models: Vec<String>,
        azure_models: Vec<String>,
    ) -> Self {
        let capabilities = ProviderCapabilities {
            supports_tools: true,
            supports_structured_output: true,
            supports_streaming: true,
            supports_images: true,
            supports_pdf: true,
            supports_reasoning: true,
        };
        let mut specs = BTreeMap::new();
        specs.insert(
            Provider::Codex,
            ProviderSpec {
                provider: Provider::Codex,
                default_model: codex_models[0].clone(),
                available_models: codex_models,
                base_url: Some("https://chatgpt.com/backend-api/codex".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Openai,
            ProviderSpec {
                provider: Provider::Openai,
                default_model: openai_models[0].clone(),
                available_models: openai_models,
                base_url: Some("https://api.openai.com/v1".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Nanogpt,
            ProviderSpec {
                provider: Provider::Nanogpt,
                default_model: nanogpt_models[0].clone(),
                available_models: nanogpt_models,
                base_url: Some("https://nano-gpt.com/api/v1".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Azure,
            ProviderSpec {
                provider: Provider::Azure,
                default_model: azure_models[0].clone(),
                available_models: azure_models,
                base_url: None,
                capabilities,
            },
        );
        Self { specs }
    }
    pub fn get(&self, provider: &Provider) -> Result<&ProviderSpec> {
        self.specs
            .get(provider)
            .ok_or_else(|| Error::UnknownProvider(provider.as_str().to_owned()))
    }

    pub fn specs(&self) -> impl Iterator<Item = &ProviderSpec> {
        self.specs.values()
    }

    pub fn as_json(&self) -> serde_json::Value {
        serde_json::to_value(self.specs().collect::<Vec<_>>())
            .expect("ProviderSpec is serializable")
    }
}

#[derive(Deserialize)]
struct OpenAiModelList {
    data: Vec<OpenAiModel>,
}

#[derive(Deserialize)]
struct OpenAiModel {
    id: String,
}

async fn discover_nanogpt_models() -> Result<Vec<String>> {
    let api_key = std::env::var("NANOGPT_API_KEY").map_err(|_| {
        Error::State("NANOGPT_API_KEY is required for NanoGPT model discovery".into())
    })?;
    let base_url =
        std::env::var("NANOGPT_BASE_URL").unwrap_or_else(|_| "https://nano-gpt.com/api/v1".into());
    let models = reqwest::Client::new()
        .get(format!("{}/models", base_url.trim_end_matches('/')))
        .bearer_auth(api_key)
        .send()
        .await?
        .error_for_status()?
        .json::<OpenAiModelList>()
        .await?;
    select_nanogpt_models(models.data.into_iter().map(|model| model.id))
}

fn select_nanogpt_models(ids: impl IntoIterator<Item = String>) -> Result<Vec<String>> {
    let ids = ids.into_iter().collect::<Vec<_>>();
    let latest = |prefix: &str, thinking: bool, is_supported: fn(&str) -> bool| {
        ids.iter()
            .filter(|id| {
                id.starts_with(prefix) && id.ends_with(":thinking") == thinking && is_supported(id)
            })
            .max_by_key(|id| version_key(id))
            .cloned()
    };
    let deepseek = [
        latest("deepseek/deepseek-v", false, is_deepseek_pro),
        latest("deepseek/deepseek-v", true, is_deepseek_pro),
    ];
    let glm = [
        latest("zai-org/glm-", false, is_glm),
        latest("zai-org/glm-", true, is_glm),
    ];
    let kimi = [
        latest("moonshotai/kimi-k", false, is_kimi),
        latest("moonshotai/kimi-k", true, is_kimi),
    ];
    let selected: Vec<String> = deepseek
        .into_iter()
        .chain(glm)
        .chain(kimi)
        .flatten()
        .collect();
    if selected.is_empty() {
        return Err(Error::State(
            "NanoGPT returned no supported DeepSeek, GLM, or Kimi models".into(),
        ));
    }
    Ok(selected)
}

fn is_deepseek_pro(id: &str) -> bool {
    let id = id.trim_end_matches(":thinking");
    let Some(version) = id.strip_prefix("deepseek/deepseek-v") else {
        return false;
    };
    let Some(version) = version.strip_suffix("-pro") else {
        return false;
    };
    !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
}

fn is_glm(id: &str) -> bool {
    let version = id
        .trim_end_matches(":thinking")
        .strip_prefix("zai-org/glm-");
    version.is_some_and(|version| {
        version
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    })
}

fn is_kimi(id: &str) -> bool {
    let version = id
        .trim_end_matches(":thinking")
        .strip_prefix("moonshotai/kimi-k");
    version.is_some_and(|version| {
        version
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
    })
}

fn version_key(model: &str) -> Vec<u32> {
    model
        .split(|character: char| !character.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect()
}

fn models_from_environment(name: &str, fallback: Vec<String>) -> Result<Vec<String>> {
    let raw = match std::env::var(name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return Ok(deduplicate_models(fallback)),
        Err(error) => return Err(Error::State(format!("could not read {name}: {error}"))),
    };
    let models = deduplicate_models(raw.split(',').map(str::trim).map(str::to_owned).collect());
    if models.is_empty() {
        return Err(Error::State(format!(
            "{name} must contain at least one model slug"
        )));
    }
    Ok(models)
}

fn deduplicate_models(models: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for model in models {
        if !model.is_empty() && !unique.contains(&model) {
            unique.push(model);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::select_nanogpt_models;

    #[test]
    fn selects_the_latest_supported_variant_for_each_nanogpt_family() {
        let models = select_nanogpt_models(
            [
                "deepseek/deepseek-v3.2",
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-pro:thinking",
                "deepseek/deepseek-v4-pro-cheaper:thinking",
                "zai-org/glm-5.1",
                "zai-org/glm-5.2",
                "zai-org/glm-5.2:thinking",
                "moonshotai/kimi-k2.7-code",
                "moonshotai/kimi-k3",
                "moonshotai/kimi-k2.6:thinking",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(
            models,
            [
                "deepseek/deepseek-v4-pro",
                "deepseek/deepseek-v4-pro:thinking",
                "zai-org/glm-5.2",
                "zai-org/glm-5.2:thinking",
                "moonshotai/kimi-k3",
                "moonshotai/kimi-k2.6:thinking",
            ]
        );
    }
}
