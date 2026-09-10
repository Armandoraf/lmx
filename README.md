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
Defaults are `gpt-live-1` for `openai`, `gpt-live-1-codex` for `codex`, and
24 kHz mono PCM16 WAV output.

```ts
const recording = await generateSpeech({
  provider: 'openai',
  input: 'I thought you were gone.',
  voice: 'gleam',
  instructions: 'Quiet relief, trying to sound casual.',
  format: 'wav',
  signal: abortController.signal,
});
```

Python exposes `lmx.generate_speech({...})` with the same request fields except
`signal`. Both return `content`, `contentType`, `model`, `voice`, `format`,
`transcript`, and provider-reported final `usage` (or `null` when absent).
Formats are WAV and raw PCM.
Delivery instructions control tone and pace separately from the spoken input.
The `openai` provider uses a Platform API key and a public Live WebSocket session.
For subscription voice, pass a `codex` request context instead:

```ts
const recording = await generateSpeech({
  context: {
    provider: 'codex',
    apiKey: accessToken,
    headers: { 'ChatGPT-Account-ID': accountId },
  },
  voice: 'cove',
  input: 'I thought you were gone.',
  instructions: 'Quiet relief.',
});
```

Codex uses ChatGPT call creation, native WebRTC/Opus media and a Live control
WebSocket. Voices are `arbor`, `breeze`, `cove`, `ember`, `juniper`, `maple`,
`sol`, `spruce`, and `vale`. This is Codex's subscription integration, not the
public Platform API contract. LMX never reads, stores or refreshes OAuth tokens;
the caller owns credentials and their lifecycle. There is no credential fallback.
Neither path calls a reasoning backend. Silent input advances the session clock.
LMX checks transcript words against the script (ignoring case/punctuation), waits
for two seconds of PCM silence, requests closure, and waits for `session.closed`.
Changed words, missing audio, moderation, incomplete finalization, and a 120-second
take deadline fail instead of returning a partial recording. Silence detection
is an application heuristic, not a Live completion event; audition recordings.
Leading/trailing silence is trimmed with 200 ms padding. API errors are not retried.
Codex packet loss/reordering fails a take rather than silently damaging the recording.
Native packages bundle Opus; consumers need no Python, browser, or system codec.
See the [Live session documentation](https://developers.openai.com/api/docs/guides/live-conversations).

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
