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

Set `OPENAI_API_KEY` before running this example. LMX supports documented
provider credentials only; it does not reuse ChatGPT or Codex OAuth tokens.

See the [repository README](https://github.com/Armandoraf/lmx#readme) for
providers, Python usage, development, and security reporting.

MIT © 2026 Armandoraf
