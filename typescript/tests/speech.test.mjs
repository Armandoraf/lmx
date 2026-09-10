import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';
import { WebSocketServer } from 'ws';
import { generateSpeech, getProvider } from '../dist/index.js';

async function serverTest(handler, run) {
  const server = createServer();
  const ws = new WebSocketServer({ server });
  ws.on('connection', (socket, request) => {
    assert.equal(request.url, '/v1/live/sessions');
    assert.equal(request.headers.authorization, 'Bearer test-key');
    const send = event => socket.send(JSON.stringify(event));
    socket.on('message', bytes => handler(JSON.parse(bytes), send, socket));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    await run({ provider: 'openai', apiKey: 'test-key', baseUrl: `http://127.0.0.1:${server.address().port}/v1` });
  } finally {
    for (const socket of ws.clients) socket.terminate();
    await new Promise(resolve => ws.close(resolve));
    await new Promise(resolve => server.close(resolve));
  }
}

const samples = Buffer.alloc(4800);
for (let i = 0; i < samples.length; i += 2) samples.writeInt16LE(1000, i);
const silence = Buffer.alloc(96000);
function audio(send, bytes) { send({ type: 'session.output_audio.delta', delta: bytes.toString('base64') }); }
function transcript(send, text) { send({ type: 'session.output_transcript.delta', delta: text }); }
function perform(event, send) {
  if (event.type === 'session.start') send({ type: 'session.started', session: { id: 'live_test' } });
  if (event.type === 'session.instructions.append') send({ type: 'session.instructions.appended', client_event_id: event.event_id });
  if (event.type === 'session.commentary.append') {
    audio(send, samples);
    transcript(send, 'Hello.');
    audio(send, silence);
  }
  if (event.type === 'session.close') send({ type: 'session.closed', reason: 'close_requested', usage: { seconds: 4 } });
}

test('Live records the scripted performance, trims silence, returns WAV and final usage', async () => {
  const events = [];
  await serverTest((event, send) => {
    events.push(event);
    perform(event, send);
  }, async context => {
    const result = await generateSpeech({ context, input: 'Hello.', voice: 'gleam', instructions: 'Whisper slowly.' });
    const wav = Buffer.from(result.content);
    assert.equal(wav.subarray(0, 4).toString(), 'RIFF');
    assert.equal(wav.readUInt32LE(24), 24000);
    assert.equal(wav.readUInt32LE(40), wav.length - 44);
    assert.deepEqual(wav.subarray(44, 44 + samples.length), samples);
    assert.equal(wav.length, 44 + samples.length + 9600);
    assert.equal(result.model, 'gpt-live-1');
    assert.equal(result.transcript, 'Hello.');
    assert.deepEqual(result.usage, { seconds: 4 });
    assert.equal(result.contentType, 'audio/wav');
    assert.equal(result.format, 'wav');
  });
  const session = events[0].session;
  assert.equal(session.model, 'gpt-live-1');
  assert.equal(session.audio.output.voice, 'gleam');
  assert.equal(session.input[0].content[0].text, 'Hello.');
  assert.match(session.instructions, /Whisper slowly/);
  assert.deepEqual(session.delegation, { type: 'client' });
  assert.equal(session.store, false);
  assert.equal(events.filter(e => e.type === 'session.close').length, 1);
  assert.equal(getProvider('openai').capabilities.supportsSpeech, true);
  assert.equal(getProvider('codex').capabilities.supportsSpeech, true);
});

test('Live accepts transcript after audio and PCM split across deltas', async () => {
  await serverTest((event, send) => {
    if (event.type === 'session.commentary.append') {
      audio(send, samples.subarray(0, 1));
      audio(send, samples.subarray(1));
      audio(send, silence);
      transcript(send, 'HELLO!');
    } else perform(event, send);
  }, async context => {
    const result = await generateSpeech({ context, input: 'Hello.', voice: 'gleam', format: 'pcm' });
    assert.equal(result.contentType, 'audio/pcm');
    assert.deepEqual(Buffer.from(result.content).subarray(0, samples.length), samples);
  });
});

for (const scenario of ['paraphrase', 'missing audio', 'moderation', 'disconnect', 'extra words during close']) {
  test(`Live rejects ${scenario} without returning a recording`, async () => {
    await serverTest((event, send, socket) => {
      if (event.type === 'session.commentary.append') {
        if (scenario === 'disconnect') { socket.close(); return; }
        if (scenario === 'moderation') { send({ type: 'error', error: { message: 'moderation interrupted audio' } }); return; }
        transcript(send, scenario === 'paraphrase' ? 'Good morning.' : 'Hello.');
        if (scenario !== 'missing audio') audio(send, samples);
        audio(send, silence);
        if (scenario !== 'extra words during close') send({ type: 'session.closed', reason: 'expired', usage: { seconds: 4 } });
      } else if (event.type === 'session.close' && scenario === 'extra words during close') {
        transcript(send, ' How are you?');
        send({ type: 'session.closed', reason: 'close_requested', usage: { seconds: 4 } });
      } else perform(event, send);
    }, context => assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'gleam' }), /Live|moderation/));
  });
}

test('Live cancellation closes the in-flight socket', async () => {
  const controller = new AbortController();
  await serverTest((event, send) => {
    if (event.type === 'session.start') controller.abort();
  }, context => assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'gleam', signal: controller.signal }), { name: 'AbortError' }));
});

test('Live preserves HTTP authentication failures', async () => {
  const server = createServer();
  server.on('upgrade', (_, socket) => socket.end('HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    await assert.rejects(generateSpeech({ context: { provider: 'openai', apiKey: 'invalid', baseUrl: `http://127.0.0.1:${server.address().port}/v1` }, input: 'Hello.', voice: 'gleam' }), /401/);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});

test('Codex uses request-scoped OAuth, its own model and call payload, and preserves rejection', async () => {
  let called = false;
  const server = createServer(async (request, response) => {
    called = true;
    assert.equal(request.url, '/realtime/calls?intent=quicksilver&architecture=avas');
    assert.equal(request.headers.authorization, 'Bearer codex-token');
    assert.equal(request.headers['chatgpt-account-id'], 'account');
    assert.equal(request.headers['openai-alpha'], 'quicksilver=v2');
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    assert.match(body.sdp, /m=audio/);
    assert.match(body.sdp, /m=application/);
    assert.equal(body.session.model, 'gpt-live-1-codex');
    assert.equal(body.session.audio.output.voice, 'cove');
    assert.equal(body.session.initial_items[0].content[0].text, 'Hello.');
    response.writeHead(403);
    response.end('Voice session access denied');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const context = { provider: 'codex', apiKey: 'codex-token', headers: { 'ChatGPT-Account-ID': 'account' }, baseUrl: `http://127.0.0.1:${server.address().port}` };
  try {
    await assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'cove' }), /403.*Voice session access denied/);
    assert.equal(called, true);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(generateSpeech({ context, input: 'Hello.', voice: 'cove', signal: controller.signal }), { name: 'AbortError' });
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
