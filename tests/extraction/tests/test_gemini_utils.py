from __future__ import annotations

import unittest

from viewing_context_pipeline.extraction.common.gemini import parse_json_response


class GeminiUtilsTests(unittest.TestCase):
    def test_parse_json_response_plain_json(self) -> None:
        self.assertEqual(parse_json_response('{"a": 1}'), {"a": 1})

    def test_parse_json_response_code_fence(self) -> None:
        raw = """```json
{"a": 1}
```"""
        self.assertEqual(parse_json_response(raw), {"a": 1})


if __name__ == "__main__":
    unittest.main()
