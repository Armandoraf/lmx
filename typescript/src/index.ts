/** Thin TypeScript adapter over the Rust-owned LMX engine. */

import * as native from '../native.js';

type Native = {
  version(): string;
  providerRegistryJson(): string;
  loadRequestContextJson(provider: string): string;
  buildMessageItemJson(role: string, text: string): string;
  outputTextFromItemsJson(items: string): string;
  normalizeToolOutputJson(callId: string, value: string): string;
  toolFailureOutputJson(callId: string, error: string): string;
  buildWireRequestJson(request: string): string;
  generateImageJson(request: string): Promise<string>;
  generateVideoJson(request: string): Promise<string>;
  ResponseSession: new (request: string) => {
    executeRoundJson(): Promise<string>;
    submitToolOutputsJson(outputs: string): string;
    startRound(): void;
    nextFrameJson(): Promise<string>;
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

function normalizeToolOutput(callId: string, value: unknown): Record<string, unknown> {
  return JSON.parse(core.normalizeToolOutputJson(callId, JSON.stringify(value))) as Record<string, unknown>;
}

function toolFailureOutput(callId: string, error: string): Record<string, unknown> {
  return JSON.parse(core.toolFailureOutputJson(callId, error)) as Record<string, unknown>;
}

export async function* streamResponse(request: ResponseRequest): AsyncGenerator<Record<string, unknown>> {
  const { toolHandlers = {}, ...payload } = request;
  if (!payload.context && typeof payload.provider === 'string') {
    payload.context = loadRequestContext(payload.provider);
    delete payload.provider;
  }
  const session = new core.ResponseSession(JSON.stringify(payload));
  while (true) {
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
        yield frame.event!;
        continue;
      }
      if (frame.type === 'failed') throw new Error(frame.error);
      next = frame.next!;
      break;
    }
    if (next.type === 'completed') {
      yield { type: 'completed', ...(next.result as Record<string, unknown>) };
      return;
    }
    const calls = next.calls as Array<{ name: string; callId: string; arguments: Record<string, unknown> }>;
    const outputs: Array<Record<string, unknown>> = [];
    for (const call of calls) {
      yield { type: 'tool_call_started', ...call };
      let output: Record<string, unknown>;
      try {
        const value = toolHandlers[call.name]
          ? await toolHandlers[call.name](call.arguments)
          : { ok: false, error: `unknown function tool: ${call.name}` };
        output = normalizeToolOutput(call.callId, value);
      } catch (error) {
        output = toolFailureOutput(call.callId, error instanceof Error ? error.message : String(error));
      }
      outputs.push(output);
      yield { type: 'tool_call_completed', name: call.name, callId: call.callId, result: output.result };
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
