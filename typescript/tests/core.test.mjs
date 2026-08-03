import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { generateImage, generateVideo, streamResponse } from '../dist/index.js';

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

test('the Rust media engine generates OpenAI-compatible image and video requests', async () => {
  await withServer((request, response) => {
    if (request.url === '/images/generations') {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ data: [{ b64_json: Buffer.from('image-bytes').toString('base64') }] }));
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
    const video = await generateVideo({ context, model: 'sora-2', prompt: 'A blue square moving' });
    assert.equal(Buffer.from(image.content).toString(), 'image-bytes');
    assert.equal(Buffer.from(video.content).toString(), 'video-bytes');
  });
});
