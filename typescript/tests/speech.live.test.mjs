import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import test from 'node:test';
import { generateSpeech } from '../dist/index.js';

test('Codex OAuth produces a transcript-matched native WebRTC recording', {
  skip: !process.env.LMX_TEST_CODEX_AUTH,
  timeout: 170000,
}, async () => {
  const { tokens } = JSON.parse(await readFile(process.env.LMX_TEST_CODEX_AUTH, 'utf8'));
  const result = await generateSpeech({
    context: { provider: 'codex', apiKey: tokens.access_token, headers: { 'ChatGPT-Account-ID': tokens.account_id } },
    voice: 'cove', input: 'I thought you were gone.', instructions: 'Quiet relief.',
  });
  assert.equal(result.model, 'gpt-live-1-codex');
  assert.equal(result.transcript.toLowerCase().replace(/[^a-z ]/g, '').trim(), 'i thought you were gone');
  const wav = Buffer.from(result.content);
  assert.equal(wav.subarray(0, 4).toString(), 'RIFF');
  assert.equal(wav.readUInt32LE(24), 24000);
  assert.equal(wav.readUInt32LE(40), wav.length - 44);
  assert.ok(wav.length > 48000);
  if (process.env.LMX_TEST_SPEECH_OUTPUT) await writeFile(process.env.LMX_TEST_SPEECH_OUTPUT, wav);
  console.log(JSON.stringify({ model: result.model, transcript: result.transcript, bytes: wav.length, usage: result.usage }));
});
