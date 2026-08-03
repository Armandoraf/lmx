import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { z } from 'zod';

import {
  availableModelsForProvider,
  defaultModelForProvider,
  generateImage,
  generateVideo,
  getProvider,
  streamImage,
  streamResponse,
  structuredResponse,
  version
} from '../dist/index.js';

test('the JavaScript package and native binding report the same release version', () => {
  assert.equal(version(), '0.1.9');
});

async function withServer(handler, run) {
  const server = createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const address = server.address();
    await run(`http://127.0.0.1:${address.port}`);
  } finally {
    await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
}

test('the Rust response engine preserves tool-round state', async () => {
  const requests = [];
  await withServer((request, response) => {
    let body = '';
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests.push(JSON.parse(body));
      const item = requests.length === 1
        ? { type: 'function_call', call_id: 'call_add', name: 'add', arguments: '{"a":2,"b":3}' }
        : { type: 'message', role: 'assistant', content: 'five' };
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(`data: ${JSON.stringify({ type: 'response.output_item.done', output_index: 0, item })}\n\ndata: ${JSON.stringify({ type: 'response.completed' })}\n\n`);
    });
  }, async baseUrl => {
    const events = [];
    for await (const event of streamResponse({
      context: { provider: 'azure', apiKey: 'test', baseUrl },
      model: 'gpt-5.5',
      input: [{ type: 'message', role: 'user', content: 'Add.' }],
      tools: [{ type: 'function', name: 'add' }],
      toolHandlers: { add: ({ a, b }) => ({ type: 'json', result: { sum: a + b } }) }
    })) events.push(event);
    assert.equal(events.at(-1)?.type, 'completed');
  });
  assert.equal(requests.length, 2);
  assert.equal(requests[1].input.at(-1).output, '{"sum":5}');
});

test('streamResponse yields a text delta before the SSE stream completes', async () => {
  let completed = false;
  await withServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.write(`data: ${JSON.stringify({ type: 'response.output_text.delta', delta: 'Hello' })}\n\n`);
    setTimeout(() => {
      completed = true;
      response.end(`data: ${JSON.stringify({ type: 'response.output_item.done', output_index: 0, item: { type: 'message', role: 'assistant', content: 'Hello' } })}\n\ndata: ${JSON.stringify({ type: 'response.completed' })}\n\n`);
    }, 80);
  }, async baseUrl => {
    const stream = streamResponse({
      context: { provider: 'azure', apiKey: 'test', baseUrl },
      model: 'gpt-5.5',
      input: [{ type: 'message', role: 'user', content: 'Hello' }]
    });
    const first = await stream.next();
    assert.deepEqual(first.value, { type: 'text_delta', delta: 'Hello' });
    assert.equal(completed, false);
    for await (const _event of stream) {
      // Drain the completed response so the native session can release cleanly.
    }
  });
});

test('streamResponse abort closes an active provider stream', async () => {
  let waitForDisconnect;
  await withServer((_request, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.write(`data: ${JSON.stringify({ type: 'response.output_text.delta', delta: 'Hello' })}\n\n`);
    waitForDisconnect = new Promise(resolve => {
      const timer = setTimeout(() => resolve(false), 1_000);
      response.on('close', () => {
        clearTimeout(timer);
        resolve(true);
      });
    });
  }, async baseUrl => {
    const controller = new AbortController();
    const stream = streamResponse({
      context: { provider: 'azure', apiKey: 'test', baseUrl },
      model: 'gpt-5.5',
      input: [{ type: 'message', role: 'user', content: 'Hello' }],
      signal: controller.signal
    });
    assert.deepEqual((await stream.next()).value, { type: 'text_delta', delta: 'Hello' });
    controller.abort();
    await assert.rejects(stream.next(), error => error?.name === 'AbortError');
  });
  assert.equal(await waitForDisconnect, true);
});

test('streamResponse abort does not invoke a pending tool or start another round', async () => {
  let requests = 0;
  let invoked = false;
  await withServer((request, response) => {
    requests += 1;
    request.resume();
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.end(`data: ${JSON.stringify({
      type: 'response.output_item.done',
      output_index: 0,
      item: { type: 'function_call', call_id: 'call_wait', name: 'wait', arguments: '{}' }
    })}\n\ndata: ${JSON.stringify({ type: 'response.completed' })}\n\n`);
  }, async baseUrl => {
    const controller = new AbortController();
    const stream = streamResponse({
      context: { provider: 'azure', apiKey: 'test', baseUrl },
      model: 'gpt-5.5',
      input: [{ type: 'message', role: 'user', content: 'Wait.' }],
      signal: controller.signal,
      tools: [{ type: 'function', name: 'wait' }],
      toolHandlers: {
        wait: () => {
          invoked = true;
          return { type: 'json', result: { ok: true } };
        }
      }
    });
    await stream.next(); // output_item
    assert.equal((await stream.next()).value.type, 'tool_call_started');
    controller.abort();
    await assert.rejects(stream.next(), error => error?.name === 'AbortError');
  });
  assert.equal(invoked, false);
  assert.equal(requests, 1);
});

test('the Rust media engine generates OpenAI-compatible image and video requests', async () => {
  let editBody = '';
  await withServer((request, response) => {
    if (request.url === '/images/generations') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ data: [{ b64_json: Buffer.from('image-bytes').toString('base64') }] }));
      return;
    }
    if (request.url === '/images/edits') {
      request.setEncoding('latin1');
      request.on('data', chunk => { editBody += chunk; });
      request.on('end', () => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ data: [{ b64_json: Buffer.from('edited-image-bytes').toString('base64') }] }));
      });
      return;
    }
    if (request.url === '/videos') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ id: 'video_1', status: 'completed', progress: 100 }));
      return;
    }
    if (request.url === '/videos/video_1/content?variant=video') {
      response.writeHead(200, { 'content-type': 'video/mp4' });
      response.end('video-bytes');
      return;
    }
    response.writeHead(404);
    response.end();
  }, async baseUrl => {
    const context = { provider: 'azure', apiKey: 'test', baseUrl };
    const image = await generateImage({ context, model: 'gpt-image-2', prompt: 'A blue square' });
    const tempDir = await mkdtemp(join(tmpdir(), 'lmx-image-test-'));
    const inputImage = join(tempDir, 'reference.png');
    await writeFile(inputImage, 'reference-image-bytes');
    const editedImage = await generateImage({
      context,
      model: 'gpt-image-2',
      prompt: 'Turn the blue square red',
      inputImages: [inputImage]
    });
    const video = await generateVideo({ context, model: 'sora-2', prompt: 'A blue square moving' });
    assert.equal(Buffer.from(image.content).toString(), 'image-bytes');
    assert.equal(Buffer.from(editedImage.content).toString(), 'edited-image-bytes');
    assert.equal(Buffer.from(video.content).toString(), 'video-bytes');
  });
  assert.match(editBody, /name="image\[\]"/);
});

test('streamImage yields partial and completed images for generation and edits', async () => {
  const requests = [];
  await withServer((request, response) => {
    let body = '';
    request.setEncoding('latin1');
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests.push({ url: request.url, body, contentType: request.headers['content-type'] });
      const prefix = request.url === '/images/edits' ? 'image_edit' : 'image_generation';
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.write(`event: ${prefix}.partial_image\n`);
      response.write(`data: ${JSON.stringify({
        type: `${prefix}.partial_image`,
        partial_image_index: 0,
        b64_json: Buffer.from('preview').toString('base64')
      })}\n\n`);
      response.end(`event: ${prefix}.completed\ndata: ${JSON.stringify({
        type: `${prefix}.completed`,
        b64_json: Buffer.from('final').toString('base64'),
        usage: { total_tokens: 12 }
      })}\n\n`);
    });
  }, async baseUrl => {
    const context = { provider: 'azure', apiKey: 'test', baseUrl };
    const generation = [];
    for await (const event of streamImage({
      context, model: 'gpt-image-2', prompt: 'A blue square', partialImages: 1
    })) generation.push(event);
    assert.deepEqual(generation.map(event => event.type), ['partial', 'completed']);
    assert.equal(Buffer.from(generation[0].result.content).toString(), 'preview');
    assert.equal(Buffer.from(generation[1].result.content).toString(), 'final');
    assert.equal(generation[1].usage?.total_tokens, 12);

    const tempDir = await mkdtemp(join(tmpdir(), 'lmx-image-stream-test-'));
    const inputImage = join(tempDir, 'reference.png');
    await writeFile(inputImage, 'reference-image-bytes');
    const edit = [];
    for await (const event of streamImage({
      context, model: 'gpt-image-2', prompt: 'Turn it red', inputImages: [inputImage], partialImages: 3
    })) edit.push(event);
    assert.deepEqual(edit.map(event => event.type), ['partial', 'completed']);
  });
  assert.deepEqual(JSON.parse(requests[0].body), {
    model: 'gpt-image-2', prompt: 'A blue square', size: '1024x1024', n: 1,
    quality: 'high', stream: true, partial_images: 1
  });
  assert.match(requests[1].contentType, /^multipart\/form-data/);
  assert.match(requests[1].body, /name="stream"\r\n\r\ntrue/);
  assert.match(requests[1].body, /name="partial_images"\r\n\r\n3/);
});

test('the registry reads configured model catalogs in the Rust core', () => {
  const previous = process.env.CODEX_MODELS;
  process.env.CODEX_MODELS = 'gpt-5.6-terra,gpt-5.5';
  try {
    assert.equal(defaultModelForProvider('codex'), 'gpt-5.6-terra');
    assert.deepEqual(availableModelsForProvider('codex'), ['gpt-5.6-terra', 'gpt-5.5']);
    assert.equal(getProvider('codex').capabilities.supportsStructuredOutput, true);
  } finally {
    if (previous === undefined) delete process.env.CODEX_MODELS;
    else process.env.CODEX_MODELS = previous;
  }
});

test('the Effigy structured-output contract sends a JSON schema from Zod', async () => {
  const schema = z.object({ title: z.string() });
  await withServer((request, response) => {
    let body = '';
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const payload = JSON.parse(body);
      assert.deepEqual(payload.text.format.type, 'json_schema');
      assert.equal(payload.text.format.name, 'effigy_assistant_thread_title');
      assert.equal(payload.text.format.strict, true);
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      response.end(`data: ${JSON.stringify({
        type: 'response.output_item.done',
        output_index: 0,
        item: { type: 'message', role: 'assistant', content: '{"title":"A title"}' }
      })}\n\ndata: ${JSON.stringify({ type: 'response.completed' })}\n\n`);
    });
  }, async baseUrl => {
    const result = await structuredResponse({
      context: { provider: 'azure', apiKey: 'test', baseUrl },
      model: 'gpt-5.5',
      input: [{ type: 'message', role: 'user', content: 'Name this.' }],
      textFormat: schema,
      textFormatName: 'effigy_assistant_thread_title'
    });
    assert.deepEqual(result, { title: 'A title' });
  });
});
