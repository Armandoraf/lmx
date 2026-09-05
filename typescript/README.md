# @armandoraf/lmx

`@armandoraf/lmx` is the TypeScript adapter for LMX, a Rust-backed engine for
documented LLM provider APIs. It supports normalized streaming, tool calls,
image generation, and video generation through a single interface.

```sh
npm install @armandoraf/lmx
```

```ts
import { respond } from '@armandoraf/lmx';

const result = await respond({
  provider: 'openai',
  model: 'gpt-5.5',
  input: [{ type: 'message', role: 'user', content: 'Say hello.' }],
});
console.log(result.outputText);
```

Set `OPENAI_API_KEY` before running this example. Codex calls instead accept
request-scoped ChatGPT credentials supplied by the host.

`await discoverProviderRegistry(codexContext)` returns account-visible Codex
models and `modelDetails` (context window, compaction threshold, reasoning
efforts, image inputs, and tool search). Discovery is cached for five minutes
per credential identity, with concurrent lookups coalesced. Tokens are never
persisted. Hidden models and the multi-agent Ultra mode are not exposed.

OpenAI API models are discovered from `/v1/models` when `OPENAI_API_KEY` is set.
That endpoint supplies names but no context or reasoning metadata. `CODEX_MODELS`
optionally restricts the discovered Codex catalog; `OPENAI_MODELS` provides an
explicit API catalog. Configured defaults do not change when new models appear.
An unavailable provider returns an empty catalog and `discoveryError`; other
providers remain usable. Failed environment-provider discovery is retried after
15 seconds rather than serving a static model list.

Codex responses automatically use Standard Responses and catalog-derived server
compaction. Callers no longer need model-name checks or compaction constants.

See the [repository README](https://github.com/Armandoraf/lmx#readme) for
providers, Python usage, development, and security reporting.

MIT © 2026 Armandoraf
