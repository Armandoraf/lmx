# Changelog

All notable changes to LMX are documented here.

## [0.2.1] - 2026-08-10

- Added a direct Codex Responses provider for host-managed, request-scoped
  ChatGPT OAuth credentials. LMX never reads, persists, or refreshes tokens.

## [0.2.0] - 2026-08-06

- Prepared LMX for public distribution with public installation docs, OSS
  governance files, and release validation.
- Removed the unsupported Codex/ChatGPT OAuth provider. OpenAI requests use
  documented API-key authentication.
- Removed the unimplemented Bedrock provider from the public registry.
