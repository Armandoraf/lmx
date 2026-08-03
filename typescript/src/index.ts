/** Thin TypeScript adapter over the Rust-owned LMX engine. */

import * as native from '../native.js';

type Native = {
  version(): string;
  providerRegistryJson(): string;
  loadRequestContextJson(provider: string): string;
  buildMessageItemJson(role: string, text: string): string;
  outputTextFromItemsJson(items: string): string;
  buildWireRequestJson(request: string): string;
  generateImageJson(request: string): Promise<string>;
  generateVideoJson(request: string): Promise<string>;
  ResponseSession: new (request: string) => {
    executeRoundJson(): Promise<string>;
    submitToolOutputsJson(outputs: string): string;
  };
};

const core = native as Native;

export type WireRequest = {
  method: 'POST';
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
};

export const version = (): string => core.version();

export const providerRegistry = (): unknown[] =>
  JSON.parse(core.providerRegistryJson()) as unknown[];

export const loadRequestContext = (provider: string): Record<string, unknown> =>
  JSON.parse(core.loadRequestContextJson(JSON.stringify(provider))) as Record<string, unknown>;

export const buildMessageItem = (params: { role: string; text: string }): Record<string, unknown> =>
  JSON.parse(core.buildMessageItemJson(params.role, params.text)) as Record<string, unknown>;

export const outputTextFromItems = (items: Record<string, unknown>[]): string =>
  core.outputTextFromItemsJson(JSON.stringify(items));

export const buildWireRequest = (request: Record<string, unknown>): WireRequest =>
  JSON.parse(core.buildWireRequestJson(JSON.stringify(request))) as WireRequest;

export type ImageResult = {
  job: Record<string, unknown>;
  content: Uint8Array;
  contentType: string;
};

export const generateImage = async (request: Record<string, unknown>): Promise<ImageResult> => {
  const payload = { ...request };
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const result = JSON.parse(await core.generateImageJson(JSON.stringify(payload))) as {
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

type ToolHandler = (arguments_: Record<string, unknown>) => unknown | Promise<unknown>;
type ResponseRequest = Record<string, unknown> & { toolHandlers?: Record<string, ToolHandler> };

export async function* streamResponse(request: ResponseRequest): AsyncGenerator<Record<string, unknown>> {
  const { toolHandlers = {}, ...payload } = request;
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const session = new core.ResponseSession(JSON.stringify(payload));
  while (true) {
    const round = JSON.parse(await session.executeRoundJson()) as {
      events: Record<string, unknown>[];
      next: Record<string, unknown>;
    };
    for (const event of round.events) yield event;
    if (round.next.type === 'completed') {
      yield { type: 'completed', ...(round.next.result as Record<string, unknown>) };
      return;
    }
    const calls = round.next.calls as Array<{ name: string; callId: string; arguments: Record<string, unknown> }>;
    const outputs: Array<Record<string, unknown>> = [];
    for (const call of calls) {
      yield { type: 'tool_call_started', ...call };
      let result: unknown;
      let content: unknown;
      try {
        const value = toolHandlers[call.name]
          ? await toolHandlers[call.name](call.arguments)
          : { ok: false, error: `unknown function tool: ${call.name}` };
        if (value && typeof value === 'object' && (value as { type?: string }).type === 'content') {
          result = (value as { result: unknown }).result;
          content = (value as { content: unknown }).content;
        } else {
          result = value && typeof value === 'object' && (value as { type?: string }).type === 'json'
            ? (value as { result: unknown }).result
            : value;
        }
      } catch (error) {
        result = { ok: false, error: error instanceof Error ? error.message : String(error) };
      }
      outputs.push({ callId: call.callId, result, ...(content === undefined ? {} : { content }) });
      yield { type: 'tool_call_completed', name: call.name, callId: call.callId, result };
    }
    session.submitToolOutputsJson(JSON.stringify(outputs));
  }
}

export const respond = async (request: ResponseRequest): Promise<Record<string, unknown>> => {
  let result: Record<string, unknown> | undefined;
  for await (const event of streamResponse(request)) {
    if (event.type === 'completed') result = event;
  }
  if (!result) throw new Error('response did not complete');
  const { type: _type, ...response } = result;
  return response;
};

export async function structuredResponse<T>(request: ResponseRequest & {
  textFormat: unknown;
  schema: { safeParse(value: unknown): { success: true; data: T } | { success: false } };
}): Promise<T> {
  const { schema, ...responseRequest } = request;
  const result = await respond(responseRequest);
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
      const parsed = schema.safeParse(JSON.parse(candidate) as unknown);
      if (parsed.success) return parsed.data;
    } catch {
      // Try the next candidate.
    }
  }
  throw new Error('structured response did not contain parsed output');
}
