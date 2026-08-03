# LMX

LMX is one provider engine with Python and TypeScript adapters. Its behavior is
implemented in Rust exactly once; the SDKs do not maintain parallel request,
provider, streaming, or image-processing implementations.

## Layout

```text
crates/lmx-core/       authoritative provider engine
crates/lmx-python/     PyO3 binding layer
crates/lmx-node/       napi-rs binding layer
python/                published Python package
typescript/            published npm package
tests/fixtures/        cross-language protocol fixtures
```

`lmx-core` owns provider defaults, OpenAI-compatible request construction,
HTTP/SSE transport, response-event normalization, tool-round state, and
white-key transparency processing. Python and TypeScript convert their public
values at the edge and delegate all of those decisions to Rust.

## Core contract

A response follows one stateful path:

```text
typed SDK request → ResponseMachine → WireRequest → OpenAiTransport
                                              ↓
              SDK tool callback ← ToolCalls ← normalized core events
```

The wrapper executes a user-provided tool and supplies its `ToolOutput` back to
the same `ResponseMachine`. The core decides whether to start the next provider
round or complete the response. This keeps callback execution language-native
without duplicating orchestration.

## Streaming images

`streamImage` (TypeScript) and `stream_image` (Python) yield normalized
`partial` preview events followed by a `completed` event. They use
`/images/generations` when no input images are supplied and `/images/edits` when
input images are present. `partialImages` defaults to `2` and
accepts `0` through `3`.

```ts
for await (const event of streamImage({
  provider: 'openai', model: 'gpt-image-2', prompt: 'A blue square', partialImages: 2
})) {
  // event.result.content is a Uint8Array
}
```

## Development

```sh
cargo test --workspace
cd typescript && npm run typecheck
```

Build the Python extension with Maturin from `python/`; build the Node addon
with napi-rs from `typescript/`. The Node package declares macOS arm64/x64 and
Linux glibc arm64/x64 targets. The release process builds each target, merges
the artifacts, then runs `npm run release:artifacts` and
`npm run release:platforms` to publish the platform packages consumed as
optional dependencies by `@armandoraf/lmx`.

## Migration policy

No provider behavior belongs in `python/` or `typescript/`. A feature starts in
`lmx-core`, receives core tests and fixtures, then gains only the adapter code
required to expose it idiomatically in each SDK.
