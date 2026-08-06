# LMX

LMX is a provider engine for applications that need one response, streaming,
tool-calling, image, and video interface in both Python and TypeScript. The
implementation lives in Rust; the language packages are thin, idiomatic
adapters over the same request construction and event-normalization engine.

LMX is MIT licensed. See [LICENSE](LICENSE), [CONTRIBUTING.md](CONTRIBUTING.md),
and [SECURITY.md](SECURITY.md).

## Supported providers

LMX supports providers with documented, user-supplied credentials:

| Provider | Authentication | Configuration |
| --- | --- | --- |
| OpenAI | API key | `OPENAI_API_KEY` |
| Azure OpenAI | API key | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, optional `AZURE_OPENAI_API_VERSION` |
| NanoGPT | API key | `NANOGPT_API_KEY`, optional `NANOGPT_BASE_URL` |

LMX does not read, reuse, or refresh ChatGPT/Codex OAuth credentials. Use the
OpenAI API with an API key for OpenAI requests.

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

## Design

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
