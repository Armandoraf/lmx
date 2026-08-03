//! LMX's single authoritative implementation.
//!
//! The public Python and TypeScript packages are adapters over this crate.  They
//! never construct provider payloads or normalize provider events themselves.

mod auth;
mod endpoint;
mod error;
mod images;
mod keying;
mod providers;
mod response;
mod transport;
mod video;

pub use auth::{load_request_context, normalize_azure_endpoint, refresh_codex_auth};
pub use endpoint::endpoint_url;
pub use error::{Error, Result};
pub use images::{
    ImageJob, ImageRequest, ImageResult, ImageStreamEvent, ImageStreamRequest, generate_image,
    stream_image,
};
pub use keying::{WHITE_KEY_HEX, prompt_for_white_key, remove_white_key_background};
pub use providers::{Provider, ProviderRegistry, ProviderSpec, RequestContext};
pub use response::{
    CoreEvent, NextAction, ResponseFrame, ResponseMachine, ResponseRequest, ResponseResult,
    RoundResult, ToolCall, ToolOutput, WireRequest, build_message_item, execute_round,
    execute_round_with_cancellation, execute_round_with_observer, normalize_tool_output,
    output_text_from_items, tool_failure_output,
};
pub use tokio_util::sync::CancellationToken;
pub use transport::OpenAiTransport;
pub use video::{VideoJob, VideoRequest, VideoResult, generate_video};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
