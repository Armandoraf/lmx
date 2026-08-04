/** Thin TypeScript adapter over the Rust-owned LMX engine. */

import { z } from 'zod';

import * as native from '../native.js';

type Native = {
  version(): string;
  providerRegistryJson(): string;
  discoverProviderRegistryJson(): Promise<string>;
  loadRequestContextJson(provider: string): string;
  buildMessageItemJson(role: string, text: string): string;
  outputTextFromItemsJson(items: string): string;
  normalizeToolOutputJson(callId: string, value: string): string;
  toolFailureOutputJson(callId: string, error: string): string;
  buildWireRequestJson(request: string): string;
  generateImageJson(request: string): Promise<string>;
  generateImagesJson(request: string): Promise<string>;
  ImageStream: new (request: string) => {
    nextEventJson(): Promise<string | null>;
    cancel(): void;
  };
  generateVideoJson(request: string): Promise<string>;
  ResponseSession: new (request: string) => {
    executeRoundJson(): Promise<string>;
    submitToolOutputsJson(outputs: string): string;
    cancel(): void;
    startRound(): void;
    nextFrameJson(): Promise<string>;
  };
};

const core = native as Native;

const packageVersion = '0.1.16';

if (core.version() !== packageVersion) {
  throw new Error(
    `@armandoraf/lmx JavaScript (${packageVersion}) and native (${core.version()}) versions must match. Reinstall dependencies.`,
  );
}

export type WireRequest = {
  method: 'POST';
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
};

export const version = (): string => core.version();

export type ProviderName = 'codex' | 'openai' | 'nanogpt' | 'azure' | 'bedrock';
export type ResponseItem = Record<string, unknown>;
export type ProviderCapabilities = {
  supportsTools: boolean;
  supportsStructuredOutput: boolean;
  supportsStreaming: boolean;
  supportsImages: boolean;
  supportsPdf: boolean;
  supportsReasoning: boolean;
};
export type ProviderSpec = {
  provider: ProviderName;
  defaultModel: string;
  availableModels: string[];
  baseUrl?: string;
  capabilities: ProviderCapabilities;
};
export type ToolOutputContentItem =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string; detail?: 'auto' | 'low' | 'high' | 'original' };
export type ToolResult =
  | { type: 'json'; result: unknown }
  | { type: 'content'; result: unknown; content: ToolOutputContentItem[] };
export type ToolHandler = (arguments_: Record<string, unknown>) => ToolResult | Promise<ToolResult>;
export type TextDeltaEvent = { type: 'text_delta'; delta: string };
export type OutputItemEvent = { type: 'output_item'; outputIndex: number; item: ResponseItem };
export type ToolCallStartedEvent = {
  type: 'tool_call_started'; name: string; callId: string; arguments: Record<string, unknown>;
};
export type ToolCallCompletedEvent = {
  type: 'tool_call_completed'; name: string; callId: string; result: unknown;
};
export type FailedEvent = { type: 'failed'; error: string };
export type CompletedEvent = {
  type: 'completed'; provider: ProviderName; model: string; outputItems: ResponseItem[];
  outputText: string; toolRoundtrips: number;
};
export type ResponseEvent =
  | TextDeltaEvent
  | OutputItemEvent
  | ToolCallStartedEvent
  | ToolCallCompletedEvent
  | FailedEvent
  | CompletedEvent;
export type ResponseRequest = {
  input: ResponseItem[];
  context?: Record<string, unknown>;
  provider?: ProviderName;
  model?: string;
  instructions?: string;
  tools?: ResponseItem[];
  toolHandlers?: Record<string, ToolHandler>;
  reasoningEffort?: string;
  textVerbosity?: string;
  signal?: AbortSignal;
};

export const providerRegistry = (): ProviderSpec[] =>
  JSON.parse(core.providerRegistryJson()) as ProviderSpec[];

/** Resolve account-visible provider models and cache them in the native engine. */
export const discoverProviderRegistry = async (): Promise<ProviderSpec[]> =>
  JSON.parse(await core.discoverProviderRegistryJson()) as ProviderSpec[];

export const availableProviders = (): ProviderName[] =>
  providerRegistry().map(({ provider }) => provider);

export function getProvider(provider: ProviderName): ProviderSpec {
  const spec = providerRegistry().find(candidate => candidate.provider === provider);
  if (!spec) throw new Error(`unknown provider: ${provider}`);
  return spec;
}

export const availableModelsForProvider = (provider: ProviderName): readonly string[] =>
  getProvider(provider).availableModels;

export const defaultModelForProvider = (provider: ProviderName): string =>
  getProvider(provider).defaultModel;

export const loadRequestContext = (provider: string): Record<string, unknown> =>
  JSON.parse(core.loadRequestContextJson(JSON.stringify(provider))) as Record<string, unknown>;

export const buildMessageItem = (params: { role: string; text: string }): Record<string, unknown> =>
  JSON.parse(core.buildMessageItemJson(params.role, params.text)) as Record<string, unknown>;

export const outputTextFromItems = (items: Record<string, unknown>[]): string =>
  core.outputTextFromItemsJson(JSON.stringify(items));

export const buildWireRequest = (request: Record<string, unknown>): WireRequest =>
  JSON.parse(core.buildWireRequestJson(JSON.stringify(request))) as WireRequest;

export type ImageJob = {
  provider: string;
  model: string;
  prompt: string;
  size: string;
  width?: number;
  height?: number;
  mimeType: string;
  background?: string;
  backgroundProcessing?: string;
};

export type ImageResult = {
  job: ImageJob;
  content: Uint8Array;
  contentType: string;
};

export type ImageBatchResult = {
  images: ImageResult[];
  usage?: Record<string, unknown>;
};

export type ImagePartialEvent = {
  type: 'partial'; imageIndex: number; partialIndex: number; result: ImageResult;
};
export type ImageCompletedEvent = {
  type: 'completed'; imageIndex: number; result: ImageResult;
};
export type ImageBatchCompletedEvent = {
  type: 'batch_completed'; usage?: Record<string, unknown>;
};
export type ImageStreamEvent = ImagePartialEvent | ImageCompletedEvent | ImageBatchCompletedEvent;

export const generateImage = async (request: Record<string, unknown>): Promise<ImageResult> => {
  const payload = { ...request };
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const result = JSON.parse(await core.generateImageJson(JSON.stringify(payload))) as {
    job: ImageJob;
    contentBase64: string;
    contentType: string;
  };
  return {
    job: result.job,
    content: Uint8Array.from(Buffer.from(result.contentBase64, 'base64')),
    contentType: result.contentType
  };
};

export const generateImages = async (
  request: Record<string, unknown>,
): Promise<ImageBatchResult> => {
  const payload = { ...request };
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const result = JSON.parse(await core.generateImagesJson(JSON.stringify(payload))) as {
    images: Array<{ job: ImageJob; contentBase64: string; contentType: string }>;
    usage?: Record<string, unknown>;
  };
  return {
    images: result.images.map(image => ({
      job: image.job,
      content: Uint8Array.from(Buffer.from(image.contentBase64, 'base64')),
      contentType: image.contentType,
    })),
    usage: result.usage,
  };
};

/** Stream progressive previews and the final image from the Image API. */
export async function* streamImage(
  request: Record<string, unknown> & { signal?: AbortSignal },
): AsyncGenerator<ImageStreamEvent> {
  const { signal, ...requestPayload } = request;
  const payload = { ...requestPayload };
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const stream = new core.ImageStream(JSON.stringify(payload));
  const cancel = () => stream.cancel();
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });
  try {
    while (true) {
      if (signal?.aborted) throw abortError(signal);
      const encoded = await awaitWithSignal(stream.nextEventJson(), signal);
      if (encoded === null) return;
      const event = JSON.parse(encoded) as {
        type: 'partial' | 'completed' | 'batch_completed';
        imageIndex?: number;
        partialIndex?: number;
        result?: { job: ImageJob; contentBase64: string; contentType: string };
        usage?: Record<string, unknown>;
      };
      if (event.type === 'batch_completed') {
        yield { type: 'batch_completed', usage: event.usage };
        return;
      }
      const result: ImageResult = {
        job: event.result!.job,
        content: Uint8Array.from(Buffer.from(event.result!.contentBase64, 'base64')),
        contentType: event.result!.contentType,
      };
      if (event.type === 'partial') {
        yield {
          type: 'partial',
          imageIndex: event.imageIndex!,
          partialIndex: event.partialIndex!,
          result,
        };
      } else {
        yield { type: 'completed', imageIndex: event.imageIndex!, result };
      }
    }
  } finally {
    signal?.removeEventListener('abort', cancel);
    stream.cancel();
  }
}

export type VideoResult = {
  job: Record<string, unknown>;
  content: Uint8Array;
  contentType: string;
};

export const generateVideo = async (request: Record<string, unknown>): Promise<VideoResult> => {
  const payload = { ...request };
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const result = JSON.parse(await core.generateVideoJson(JSON.stringify(payload))) as {
    job: Record<string, unknown>;
    contentBase64: string;
    contentType: string;
  };
  return {
    job: result.job,
    content: Uint8Array.from(Buffer.from(result.contentBase64, 'base64')),
    contentType: result.contentType
  };
};

function normalizeToolOutput(callId: string, value: unknown): Record<string, unknown> {
  return JSON.parse(core.normalizeToolOutputJson(callId, JSON.stringify(value))) as Record<string, unknown>;
}

function toolFailureOutput(callId: string, error: string): Record<string, unknown> {
  return JSON.parse(core.toolFailureOutputJson(callId, error)) as Record<string, unknown>;
}

function abortError(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new DOMException('Response cancelled', 'AbortError');
}

async function awaitWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) throw abortError(signal);

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal));
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });
}

export async function* streamResponse(request: ResponseRequest): AsyncGenerator<ResponseEvent> {
  const { toolHandlers = {}, signal, ...payload } = request;
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const session = new core.ResponseSession(JSON.stringify(payload));
  const cancel = () => session.cancel();
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });

  try {
    while (true) {
      if (signal?.aborted) throw abortError(signal);
      session.startRound();
      let next: Record<string, unknown>;
      while (true) {
        const frame = JSON.parse(await session.nextFrameJson()) as {
          type: string;
          event?: Record<string, unknown>;
          next?: Record<string, unknown>;
          error?: string;
        };
        if (frame.type === 'event') {
          const event = frame.event!;
          if (event.type === 'output_item') {
            yield {
              type: 'output_item',
              outputIndex: Number(event.output_index),
              item: event.item as ResponseItem
            };
          } else {
            yield event as TextDeltaEvent;
          }
          continue;
        }
        if (frame.type === 'failed') {
          if (signal?.aborted) throw abortError(signal);
          throw new Error(frame.error);
        }
        next = frame.next!;
        break;
      }
      if (next.type === 'completed') {
        yield { type: 'completed', ...(next.result as Omit<CompletedEvent, 'type'>) };
        return;
      }
      const calls = next.calls as Array<{ name: string; callId: string; arguments: Record<string, unknown> }>;
      const outputs: Array<Record<string, unknown>> = [];
      for (const call of calls) {
        yield { type: 'tool_call_started', ...call };
        let output: Record<string, unknown>;
        try {
          if (signal?.aborted) throw abortError(signal);
          const value = toolHandlers[call.name]
            ? await awaitWithSignal(Promise.resolve(toolHandlers[call.name](call.arguments)), signal)
            : { ok: false, error: `unknown function tool: ${call.name}` };
          if (signal?.aborted) throw abortError(signal);
          output = normalizeToolOutput(call.callId, value);
        } catch (error) {
          if (signal?.aborted) throw abortError(signal);
          output = toolFailureOutput(call.callId, error instanceof Error ? error.message : String(error));
        }
        outputs.push(output);
        yield { type: 'tool_call_completed', name: call.name, callId: call.callId, result: output.result };
      }
      if (signal?.aborted) throw abortError(signal);
      session.submitToolOutputsJson(JSON.stringify(outputs));
    }
  } finally {
    signal?.removeEventListener('abort', cancel);
    session.cancel();
  }
}

export type ResponseResult = {
  provider: ProviderName;
  model: string;
  outputItems: ResponseItem[];
  outputText: string;
  toolRoundtrips: number;
};

export const respond = async (request: ResponseRequest): Promise<ResponseResult> => {
  let result: CompletedEvent | undefined;
  for await (const event of streamResponse(request)) {
    if (event.type === 'completed') result = event;
  }
  if (!result) throw new Error('response did not complete');
  const { type: _type, ...response } = result;
  return response as ResponseResult;
};

export async function structuredResponse<T>(request: ResponseRequest & {
  textFormat: {
    safeParse(value: unknown): { success: true; data: T } | { success: false };
  };
  textFormatName?: string;
}): Promise<T> {
  const { textFormat, textFormatName = 'structured_response', ...responseRequest } = request;
  const wireRequest: ResponseRequest & { textFormat: unknown } = {
    ...responseRequest,
    textFormat: {
      type: 'json_schema',
      name: textFormatName,
      schema: z.toJSONSchema(textFormat as z.ZodType),
      strict: true
    }
  };
  const result = await respond(wireRequest);
  const candidates: unknown[] = [result.outputText];
  for (const item of (result.outputItems as Array<Record<string, unknown>> | undefined) ?? []) {
    const content = item.content;
    if (typeof content === 'string') candidates.push(content);
    if (Array.isArray(content)) {
      for (const part of content) {
        if (part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string') {
          candidates.push((part as { text: string }).text);
        }
      }
    }
  }
  for (const candidate of candidates.reverse()) {
    if (typeof candidate !== 'string') continue;
    try {
      const parsed = textFormat.safeParse(JSON.parse(candidate) as unknown);
      if (parsed.success) return parsed.data;
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error('structured response did not contain parsed output');
}
