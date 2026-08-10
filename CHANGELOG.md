# Changelog

All notable changes to LMX are documented here.

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
