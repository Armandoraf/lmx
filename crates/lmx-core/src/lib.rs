//! LMX's single authoritative implementation.
//!
//! The public Python and TypeScript packages are adapters over this crate.  They
//! never construct provider payloads or normalize provider events themselves.

mod auth;
mod endpoint;
mod error;
mod images;
mod matting;
mod providers;
mod response;
mod transport;
mod video;

pub use auth::{load_request_context, normalize_azure_endpoint, refresh_codex_auth};
pub use endpoint::endpoint_url;
pub use error::{Error, Result};
pub use images::{ImageJob, ImageRequest, ImageResult, generate_image};
pub use matting::{
    MATTE_KEY_HEX, prompt_for_foreground_matte, remove_background_with_foreground_matte,
};
pub use providers::{Provider, ProviderRegistry, ProviderSpec, RequestContext};
pub use response::{
    CoreEvent, NextAction, ResponseFrame, ResponseMachine, ResponseRequest, ResponseResult,
    RoundResult, ToolCall, ToolOutput, WireRequest, build_message_item, execute_round,
    execute_round_with_observer, normalize_tool_output, output_text_from_items,
    tool_failure_output,
};
pub use transport::OpenAiTransport;
pub use video::{VideoJob, VideoRequest, VideoResult, generate_video};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
