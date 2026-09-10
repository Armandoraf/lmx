import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';
import { generateSpeech, getProvider } from '../dist/index.js';

async function serverTest(handler, run) {
  const server = createServer(handler);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    await run({ provider: 'openai', apiKey: 'test-key', baseUrl: `http://127.0.0.1:${server.address().port}/v1` });
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}

test('speech sends script and delivery independently and returns exact binary audio', async () => {
  const audio = Buffer.from([82, 73, 70, 70, 0, 255, 128]);
  await serverTest(async (request, response) => {
    assert.equal(request.url, '/v1/audio/speech');
    assert.equal(request.headers.authorization, 'Bearer test-key');
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    assert.deepEqual(JSON.parse(Buffer.concat(chunks)), {
      model: 'gpt-4o-mini-tts', voice: 'cedar', input: 'Hello.',
      instructions: 'Whisper.', response_format: 'wav', speed: 0.9,
    });
    response.writeHead(200, { 'Content-Type': 'audio/wav' });
    response.end(audio);
  }, async context => {
    const result = await generateSpeech({ context, input: 'Hello.', voice: 'cedar', instructions: 'Whisper.', speed: 0.9 });
    assert.deepEqual(Buffer.from(result.content), audio);
    assert.equal(result.contentType, 'audio/wav');
    assert.equal(result.format, 'wav');
  });
  assert.equal(getProvider('openai').capabilities.supportsSpeech, true);
  assert.equal(getProvider('codex').capabilities.supportsSpeech, false);
});

test('speech preserves provider failures', async () => {
  await serverTest((_, response) => {
    response.writeHead(429);
    response.end('speech rate limit');
  }, context => assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'cedar' }), /429.*speech rate limit/));
});

test('speech cancellation stops an in-flight body download', async () => {
  const controller = new AbortController();
  await serverTest((_, response) => {
    response.writeHead(200, { 'Content-Type': 'audio/wav' });
    response.write('RIFF');
    controller.abort();
  }, context => assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'cedar', signal: controller.signal }), { name: 'AbortError' }));
});
