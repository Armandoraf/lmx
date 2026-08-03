import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import lmx


class AdapterTests(unittest.TestCase):
    def test_message_and_output_text_are_owned_by_the_core(self) -> None:
        message = lmx.build_message_item(role="user", text="Hello")
        self.assertEqual(message, {"type": "message", "role": "user", "content": "Hello"})
        self.assertEqual(
            lmx.output_text_from_items([
                {"type": "message", "role": "assistant", "content": "First"},
                {"type": "message", "role": "user", "content": "Ignored"},
                {"type": "message", "role": "assistant", "content": [{"text": "Second"}]},
            ]),
            "First\nSecond",
        )

    def test_stream_response_yields_before_the_sse_response_completes(self) -> None:
        completed = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': 'Hello'})}\n\n".encode()
                )
                self.wfile.flush()
                time.sleep(0.08)
                completed.set()
                self.wfile.write(
                    f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'type': 'message', 'role': 'assistant', 'content': 'Hello'}})}\n\n".encode()
                )
                self.wfile.write(f"data: {json.dumps({'type': 'response.completed'})}\n\n".encode())

            def log_message(self, _format: str, *_args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            stream = lmx.stream_response(
                {
                    "context": {
                        "provider": "azure",
                        "apiKey": "test",
                        "baseUrl": f"http://{host}:{port}",
                    },
                    "model": "gpt-5.5",
                    "input": [{"type": "message", "role": "user", "content": "Hello"}],
                }
            )
            self.assertEqual(next(stream), {"type": "text_delta", "delta": "Hello"})
            self.assertFalse(completed.is_set())
            list(stream)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
