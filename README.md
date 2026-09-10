# LMX

LMX is a provider engine for applications that need one response, streaming,
tool-calling, image, and video interface in both Python and TypeScript. The
implementation lives in Rust; the language packages are thin, idiomatic
adapters over the same request construction and event-normalization engine.

LMX is MIT licensed. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md),
and [SECURITY.md](SECURITY.md).

## Supported providers

LMX supports providers with user-supplied credentials:

| Provider | Authentication | Configuration |
| --- | --- | --- |
| Codex | Request-scoped, host-managed ChatGPT OAuth | `context.apiKey` and `context.headers["ChatGPT-Account-ID"]` |
| OpenAI | API key | `OPENAI_API_KEY` |
| Azure OpenAI | API key | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, optional `AZURE_OPENAI_API_VERSION` |
| NanoGPT | API key | `NANOGPT_API_KEY`, optional `NANOGPT_BASE_URL` |

Codex credentials are intentionally request-scoped. LMX does not read Codex
credential files, persist tokens, or refresh OAuth credentials. The host that
obtains the OAuth identity supplies it for each request; on an authentication
failure, it refreshes the identity and starts a new request.

```ts
const result = await respond({
  context: {
    provider: 'codex',
    apiKey: hostManagedAccessToken,
    headers: { 'ChatGPT-Account-ID': hostManagedAccountId },
  },
  model: 'gpt-5.6-sol',
  input: [{ type: 'message', role: 'user', content: 'Review this change.' }],
});
```

Codex response calls discover the caller's visible models and default to Standard
Responses with server compaction at the catalog's threshold (or 90% of its default
context window). New models require no LMX release. Explicit transport and
compaction settings still take precedence:

```ts
const result = await respond({
  context: {
    provider: 'codex',
    apiKey: hostManagedAccessToken,
    headers: { 'ChatGPT-Account-ID': hostManagedAccountId },
  },
  model: 'gpt-5.6-sol',
  codexProtocol: 'responses_standard',
  contextManagement: { mode: 'server', compactThreshold: 244_800 },
  input: replayedNativeItems,
});
```

Standard mode omits the internal Lite header. When the backend compacts the
conversation, LMX emits a `context_compacted` event and completes with a
`replace` history update containing the opaque compaction item and subsequent
output items. Persist and replay that native history unchanged.

LMX sends these credentials only to the Codex Responses backend. It does not
provide a ChatGPT login flow or an implicit `provider: "codex"` credential
loader.

## Install

Python 3.11+:

```sh
pip install lmx-sdk
```

Node.js 18+:

```sh
npm install @armandoraf/lmx
```

The release workflow builds prebuilt Node binaries for macOS (arm64 and x64)
and Linux glibc (arm64 and x64). Other targets can build from source with a
Rust toolchain.

## Quickstart

Set an API key:

```sh
export OPENAI_API_KEY="your-api-key"
```

Python:

```python
import lmx

result = lmx.respond({
    "provider": "openai",
    "model": "gpt-5.5",
    "input": [{"type": "message", "role": "user", "content": "Say hello in five words."}],
})
print(result["outputText"])
```

TypeScript:

```ts
import { respond } from '@armandoraf/lmx';

const result = await respond({
  provider: 'openai',
  model: 'gpt-5.5',
  input: [{ type: 'message', role: 'user', content: 'Say hello in five words.' }],
});
console.log(result.outputText);
```

For streamed responses, use `stream_response` in Python or `streamResponse` in
TypeScript. Both yield normalized text, output-item, tool-call, and completion
events. `stream_image` / `streamImage` yield progressive image previews and
completed images.

## Speech

OpenAI speech generation uses the same Rust engine and credential handling as
other media APIs. TypeScript returns `Uint8Array`; Python returns `bytes`.
The default model is `gpt-4o-mini-tts` and the default format is WAV.

```ts
const recording = await generateSpeech({
  provider: 'openai',
  input: 'I thought you were gone.',
  voice: 'cedar',
  instructions: 'Quiet relief, trying to sound casual.',
  format: 'wav',
  signal: abortController.signal,
});
```

Python exposes `lmx.generate_speech({...})` with the same request fields except
`signal`. Both return `content`, `contentType`, `model`, `voice`, and `format`.
Supported formats are WAV, MP3, Opus, AAC, FLAC, and raw PCM. Optional `speed`
ranges from 0.25 to 4. Delivery instructions are separate from the spoken input.
Use OpenAI API credentials; the Codex provider does not support speech.
This API generates recordings; it does not create a live voice session.

## Architecture

```text
typed SDK request → ResponseMachine → WireRequest → provider transport
                                              ↓
              SDK tool callback ← ToolCalls ← normalized core events
```

`lmx-core` owns provider defaults, OpenAI-compatible request construction,
HTTP/SSE transport, response-event normalization, tool-round state, and
white-key transparency processing. Python and TypeScript execute callbacks in
the host language while delegating provider behavior to the core.

## Development

```sh
cargo test --workspace
cd python && maturin develop && python -m unittest discover -s tests
cd typescript && npm ci && npm test
```

Release packages are built from a version tag after CI passes. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the release and verification workflow.
