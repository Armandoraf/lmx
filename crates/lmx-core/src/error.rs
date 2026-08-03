use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("response cancelled")]
    Cancelled,
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("provider {0} has no OpenAI-compatible base URL")]
    MissingBaseUrl(String),
    #[error("model {model:?} is not available for provider {provider:?}")]
    UnsupportedModel { provider: String, model: String },
    #[error("provider {0} does not support {1}")]
    UnsupportedCapability(String, &'static str),
    #[error("invalid response state: {0}")]
    State(String),
    #[error("invalid provider event: {0}")]
    Event(String),
    #[error("invalid function-call arguments: {0}")]
    FunctionArguments(String),
    #[error("HTTP transport failed: {0}")]
    Transport(String),
    #[error("provider returned HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("filesystem operation failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
