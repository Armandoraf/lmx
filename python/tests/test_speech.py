import base64
import json
import unittest
from unittest.mock import patch
import lmx


class SpeechTests(unittest.TestCase):
    def test_speech_adapter_preserves_bytes_transcript_and_usage(self):
        audio = b"RIFF\x00\xff\x80"
        result_json = json.dumps({
            "contentBase64": base64.b64encode(audio).decode(),
            "contentType": "audio/wav", "model": "gpt-live-1", "voice": "gleam",
            "format": "wav", "transcript": "Hello.", "usage": {"seconds": 4},
        })
        request = {"context": {"provider": "openai", "apiKey": "test"},
                   "input": "Hello.", "voice": "gleam", "instructions": "Quietly."}
        with patch("lmx.generate_speech_json", return_value=result_json) as native:
            result = lmx.generate_speech(request)
        self.assertEqual(json.loads(native.call_args.args[0]), request)
        self.assertEqual(result["content"], audio)
        self.assertEqual(result["transcript"], "Hello.")
        self.assertEqual(result["usage"], {"seconds": 4})
        self.assertNotIn("contentBase64", result)

    def test_native_requires_codex_account(self):
        with self.assertRaisesRegex(Exception, "ChatGPT-Account-ID"):
            lmx.generate_speech({"context": {"provider": "codex", "apiKey": "test"},
                                 "input": "Hello.", "voice": "gleam"})
