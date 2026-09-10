import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import lmx


class SpeechTests(unittest.TestCase):
    def test_speech_binary_and_payload(self):
        requests = []
        audio = b"RIFF\x00\xff\x80"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.end_headers()
                self.wfile.write(audio)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            result = lmx.generate_speech({
                'context': {'provider': 'openai', 'apiKey': 'test', 'baseUrl': f'http://127.0.0.1:{server.server_port}/v1'},
                'input': 'Hello.', 'voice': 'cedar', 'instructions': 'Quietly.',
            })
            self.assertEqual(result['content'], audio)
            self.assertEqual(result['contentType'], 'audio/wav')
            self.assertEqual(requests, [('/v1/audio/speech', {
                'input': 'Hello.', 'voice': 'cedar', 'instructions': 'Quietly.',
                'model': 'gpt-4o-mini-tts', 'response_format': 'wav',
            })])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
