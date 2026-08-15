# Changelog

All notable changes to LMX are documented here.

## [0.4.0] - 2026-08-15

- Add an explicit `responses_standard` Codex transport for ChatGPT OAuth,
  alongside the existing `responses_lite` transport, with no automatic
  fallback between protocols.
- Support Responses server-side compaction through `context_management`, and
  stream opaque compaction items as context-replacement events that remain
  valid across subsequent tool rounds.
- Make context management a discriminated API so Lite remote V2 compaction and
  Standard server compaction cannot be mixed accidentally.

## [0.3.0] - 2026-08-15

- Track provider response IDs and detailed input, cache-write, cache-hit,
  output, and total token usage for Responses requests.
- Add Codex-compatible remote V2 compaction for ChatGPT OAuth GPT-5.6 models,
  checked before sampling and between tool rounds at 90% of the 272k context
  window.
- Return explicit append-or-replace native history updates so hosts can persist
  encrypted reasoning and compacted context without response-ID chaining.

## [0.2.9] - 2026-08-15

- Expose the exact provider input item submitted for each completed tool call,
  allowing hosts to checkpoint and faithfully replay stateless Responses
  conversations, including encrypted reasoning and multimodal tool outputs.

## [0.2.4] - 2026-08-10

- Restored the GitHub Packages release configuration used by Effigy.

## [0.2.3] - 2026-08-10

- Fixed Codex image generation and editing to use the direct ChatGPT OAuth
  Images API, matching Codex's built-in image extension.
- Build the Linux ARM native package on a native ARM runner.

## [0.2.2] - 2026-08-10

- Added Codex-backed image generation and editing through the direct Responses
  transport, including streamed partial-image previews.
- Fixed the release workflow's Linux ARM native-build target installation.

## [0.2.1] - 2026-08-10

- Added a direct Codex Responses provider for host-managed, request-scoped
  ChatGPT OAuth credentials. LMX never reads, persists, or refreshes tokens.

## [0.2.0] - 2026-08-06

- Prepared LMX for public distribution with public installation docs, OSS
  governance files, and release validation.
- Removed the unsupported Codex/ChatGPT OAuth provider. OpenAI requests use
  documented API-key authentication.
- Removed the unimplemented Bedrock provider from the public registry.
