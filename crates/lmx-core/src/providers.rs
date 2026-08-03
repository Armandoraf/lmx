use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Codex,
    Openai,
    Nanogpt,
    Azure,
    Bedrock,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Openai => "openai",
            Self::Nanogpt => "nanogpt",
            Self::Azure => "azure",
            Self::Bedrock => "bedrock",
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

impl Default for ProviderRegistry {
    fn default() -> Self {
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
                default_model: "gpt-5.5".into(),
                available_models: vec!["gpt-5.5".into(), "gpt-5.5-mini".into()],
                base_url: Some("https://chatgpt.com/backend-api/codex".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Openai,
            ProviderSpec {
                provider: Provider::Openai,
                default_model: "gpt-5.5".into(),
                available_models: vec!["gpt-5.5".into(), "gpt-5.5-mini".into()],
                base_url: Some("https://api.openai.com/v1".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Nanogpt,
            ProviderSpec {
                provider: Provider::Nanogpt,
                default_model: "moonshotai/kimi-k2.6".into(),
                available_models: vec![
                    "moonshotai/kimi-k2.6".into(),
                    "zai-org/glm-5.1".into(),
                    "deepseek/deepseek-v3.2".into(),
                ],
                base_url: Some("https://nano-gpt.com/api/v1".into()),
                capabilities: capabilities.clone(),
            },
        );
        specs.insert(
            Provider::Azure,
            ProviderSpec {
                provider: Provider::Azure,
                default_model: "gpt-5-mini".into(),
                available_models: vec!["gpt-5-mini".into(), "gpt-5-nano".into(), "gpt-5.5".into()],
                base_url: None,
                capabilities,
            },
        );
        specs.insert(
            Provider::Bedrock,
            ProviderSpec {
                provider: Provider::Bedrock,
                default_model: "us.anthropic.claude-sonnet-4-6".into(),
                available_models: vec![
                    "us.anthropic.claude-sonnet-4-6".into(),
                    "us.anthropic.claude-opus-4-7".into(),
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0".into(),
                ],
                base_url: None,
                capabilities: ProviderCapabilities {
                    supports_tools: true,
                    supports_structured_output: false,
                    supports_streaming: true,
                    supports_images: true,
                    supports_pdf: false,
                    supports_reasoning: false,
                },
            },
        );
        Self { specs }
    }
}

impl ProviderRegistry {
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
