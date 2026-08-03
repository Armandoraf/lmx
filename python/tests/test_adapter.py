import unittest

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
