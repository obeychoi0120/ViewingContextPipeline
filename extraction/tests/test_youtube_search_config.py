from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.video_data_collection.config import load_config
from src.video_data_collection.youtube_api import collect_video_candidates


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeSearch:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        video_id = "EN123" if kwargs["relevanceLanguage"] == "en" else "KO123"
        return FakeRequest({"items": [{"id": {"videoId": video_id}}]})


class FakeYouTube:
    def __init__(self):
        self.search_resource = FakeSearch()

    def search(self):
        return self.search_resource


class YouTubeSearchConfigTests(unittest.TestCase):
    def test_seed_query_can_override_search_language_region_and_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.yaml"
            path.write_text(
                "max_results_per_query: 1\n"
                "max_results_per_group: 10\n"
                "region_code: KR\n"
                "relevance_language: ko\n"
                "seed_queries:\n"
                "  - { query: 테크 리뷰, group: tech, category: Tech }\n"
                "  - { query: tech review, group: tech_en, category: Tech, relevance_language: en, region_code: US }\n",
                encoding="utf-8",
            )
            config = load_config(path)
            fake_youtube = FakeYouTube()

            rows = collect_video_candidates(fake_youtube, config, sleep_sec=0)

            self.assertEqual(fake_youtube.search_resource.calls[0]["relevanceLanguage"], "ko")
            self.assertEqual(fake_youtube.search_resource.calls[0]["regionCode"], "KR")
            self.assertEqual(fake_youtube.search_resource.calls[1]["relevanceLanguage"], "en")
            self.assertEqual(fake_youtube.search_resource.calls[1]["regionCode"], "US")
            self.assertEqual(rows[0]["seed_category"], "Tech")
            self.assertEqual(rows[1]["search_language"], "en")


if __name__ == "__main__":
    unittest.main()
