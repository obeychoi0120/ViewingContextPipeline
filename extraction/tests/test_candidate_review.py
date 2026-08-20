from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.video_data_collection.candidate_review import (
    REVIEW_CSV_FIELDNAMES,
    YouTubeRateLimitError,
    accepted_candidate_rows,
    clean_caption_text,
    enrich_candidate,
    enrich_candidates,
    ensure_review_csv_schema,
    judge_candidate,
    merge_video_lists,
    write_searched_video_list,
)


class CandidateReviewTests(unittest.TestCase):
    def test_clean_caption_text_removes_vtt_timing_and_tags(self):
        raw = (
            "WEBVTT\n"
            "Kind: captions\n\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "<c>Hello</c> &amp; welcome\n\n"
            "00:00:01.000 --> 00:00:02.000\n"
            "Hello &amp; welcome\n"
        )

        self.assertEqual(clean_caption_text(raw), "Hello & welcome")

    def test_judge_candidate_accepts_supported_language_with_script(self):
        result = judge_candidate(
            {
                "title": "Tech review",
                "duration_sec": 600,
                "script_language": "en",
                "script_chars": 1000,
                "availability": "public",
                "live_status": "not_live",
            }
        )

        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["needs_visual_review"], "false")

    def test_judge_candidate_marks_missing_script_for_visual_review(self):
        result = judge_candidate(
            {
                "title": "Travel vlog",
                "duration_sec": 600,
                "language": "en",
                "script_chars": 0,
                "availability": "public",
                "live_status": "not_live",
            }
        )

        self.assertEqual(result["decision"], "needs_visual_review")
        self.assertEqual(result["needs_visual_review"], "true")

    def test_enrich_candidates_sleeps_between_candidates(self):
        candidates = [
            {"video_id": "one", "url": "https://www.youtube.com/watch?v=one"},
            {"video_id": "two", "url": "https://www.youtube.com/watch?v=two"},
            {"video_id": "three", "url": "https://www.youtube.com/watch?v=three"},
        ]

        with mock.patch(
            "src.video_data_collection.candidate_review.enrich_candidate",
            side_effect=lambda candidate, **_: candidate,
        ), mock.patch("src.video_data_collection.candidate_review.time.sleep") as sleep:
            rows = enrich_candidates(candidates, script_dir="scripts", sleep_sec=1.5)

        self.assertEqual([row["video_id"] for row in rows], ["one", "two", "three"])
        self.assertEqual(sleep.call_args_list, [mock.call(1.5), mock.call(1.5)])

    def test_enrich_candidates_resumes_from_existing_output_rows(self):
        candidates = [
            {"video_id": "done", "url": "https://www.youtube.com/watch?v=done"},
            {"video_id": "new", "url": "https://www.youtube.com/watch?v=new"},
        ]
        existing_rows = [
            {
                "video_id": "done",
                "url": "https://www.youtube.com/watch?v=done",
                "decision": "accept",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reviewed.csv"
            with mock.patch(
                "src.video_data_collection.candidate_review.enrich_candidate",
                side_effect=lambda candidate, **_: {**candidate, "decision": "accept"},
            ) as enrich:
                rows = enrich_candidates(
                    candidates,
                    script_dir="scripts",
                    existing_rows=existing_rows,
                    output_path=output,
                )

            self.assertEqual(enrich.call_count, 1)
            self.assertEqual([row["video_id"] for row in rows], ["done", "new"])
            written = output.read_text(encoding="utf-8-sig")
            self.assertIn("done", written)
            self.assertIn("new", written)
            self.assertEqual(len(written.splitlines()), 3)

    def test_enrich_candidates_retries_existing_metadata_script_errors(self):
        candidates = [
            {"video_id": "retry", "url": "https://www.youtube.com/watch?v=retry"},
        ]
        existing_rows = [
            {
                "video_id": "retry",
                "url": "https://www.youtube.com/watch?v=retry",
                "decision": "needs_visual_review",
                "reasons": "metadata_or_script_error:URLError",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reviewed.csv"
            with mock.patch(
                "src.video_data_collection.candidate_review.enrich_candidate",
                side_effect=lambda candidate, **_: {**candidate, "decision": "accept", "reasons": "metadata_script_ok"},
            ) as enrich:
                rows = enrich_candidates(
                    candidates,
                    script_dir="scripts",
                    existing_rows=existing_rows,
                    output_path=output,
                )

            self.assertEqual(enrich.call_count, 1)
            self.assertEqual([row["decision"] for row in rows], ["needs_visual_review", "accept"])
            written = output.read_text(encoding="utf-8-sig")
            self.assertIn("metadata_or_script_error:URLError", written)
            self.assertIn("metadata_script_ok", written)

    def test_ensure_review_csv_schema_rewrites_existing_dynamic_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reviewed.csv"
            output.write_text("video_id,url,decision,custom\none,u,accept,x\n", encoding="utf-8")

            ensure_review_csv_schema(output)

            header = output.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
            self.assertEqual(header, REVIEW_CSV_FIELDNAMES)
            self.assertIn("one,u", output.read_text(encoding="utf-8-sig"))

    def test_enrich_candidates_stops_on_youtube_rate_limit_without_writing_current_row(self):
        candidates = [
            {"video_id": "ok", "url": "https://www.youtube.com/watch?v=ok"},
            {"video_id": "limited", "url": "https://www.youtube.com/watch?v=limited"},
            {"video_id": "later", "url": "https://www.youtube.com/watch?v=later"},
        ]

        def fake_enrich(candidate, **_):
            if candidate["video_id"] == "limited":
                raise YouTubeRateLimitError("YouTube rate limit detected")
            return {**candidate, "decision": "accept"}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reviewed.csv"
            with mock.patch(
                "src.video_data_collection.candidate_review.enrich_candidate",
                side_effect=fake_enrich,
            ):
                with self.assertRaises(YouTubeRateLimitError):
                    enrich_candidates(candidates, script_dir="scripts", output_path=output)

            written = output.read_text(encoding="utf-8-sig")
            self.assertIn("ok", written)
            self.assertNotIn("limited", written)
            self.assertNotIn("later", written)

    def test_enrich_candidate_raises_on_youtube_rate_limit_message(self):
        message = (
            "ERROR: [youtube] JzJw5qM2brE: This content isn't available, try again later. "
            "The current session has been rate-limited by YouTube for up to an hour."
        )
        candidate = {
            "video_id": "limited",
            "url": "https://www.youtube.com/watch?v=limited",
        }

        with mock.patch(
            "src.video_data_collection.candidate_review.extract_video_info",
            side_effect=RuntimeError(message),
        ):
            with self.assertRaises(YouTubeRateLimitError):
                enrich_candidate(candidate, script_dir="scripts")

    def test_accepted_candidate_rows_deduplicates_manual_and_preserves_category(self):
        rows = [
            {
                "decision": "accept",
                "url": "https://www.youtube.com/watch?v=manual001",
                "video_id": "manual001",
                "seed_category": "Tech",
            },
            {
                "decision": "accept",
                "url": "https://www.youtube.com/watch?v=newtech001",
                "video_id": "newtech001",
                "seed_category": "Tech",
            },
            {
                "decision": "needs_visual_review",
                "url": "https://www.youtube.com/watch?v=review001",
                "video_id": "review001",
                "seed_category": "Game",
            },
        ]

        accepted = accepted_candidate_rows(
            rows,
            excluded_video_ids={"manual001"},
            excluded_urls={"https://www.youtube.com/watch?v=manual001"},
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["category"], "Tech")
        self.assertEqual(accepted[0]["video_id"], "newtech001")

    def test_write_searched_video_list_writes_accept_rows_as_auto_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "video_list_searched.txt"
            rows = [
                {
                    "decision": "accept",
                    "url": "https://www.youtube.com/watch?v=newtech001",
                    "video_id": "newtech001",
                    "seed_category": "Tech",
                },
                {
                    "decision": "reject",
                    "url": "https://www.youtube.com/watch?v=reject001",
                    "video_id": "reject001",
                    "seed_category": "Tech",
                },
                {
                    "decision": "needs_visual_review",
                    "url": "https://www.youtube.com/watch?v=review001",
                    "video_id": "review001",
                    "seed_category": "Game",
                },
                {
                    "decision": "accept",
                    "url": "https://www.youtube.com/watch?v=newtech001",
                    "video_id": "newtech001",
                    "seed_category": "Tech",
                },
                {
                    "decision": "accept",
                    "url": "https://www.youtube.com/watch?v=newgame001",
                    "video_id": "newgame001",
                    "seed_category": "Game",
                },
            ]

            result = write_searched_video_list(rows, output)

            text = output.read_text(encoding="utf-8")
            self.assertEqual(result["searched_count"], 2)
            self.assertIn("Tech_Auto_001 https://www.youtube.com/watch?v=newtech001", text)
            self.assertIn("Game_Auto_001 https://www.youtube.com/watch?v=newgame001", text)
            self.assertNotIn("reject001", text)
            self.assertNotIn("review001", text)

    def test_merge_video_lists_merges_txt_lists_manual_then_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "video_list_manual.txt"
            searched = root / "video_list_searched.txt"
            output = root / "video_list_merged.txt"
            manual.write_text(
                "[Tech]\n"
                "Tech_Manual_001 https://www.youtube.com/watch?v=manual001\n"
                "Tech_Manual_002 https://www.youtube.com/watch?v=dupe001\n\n"
                "[Game]\n"
                "Game_Manual_001 https://www.youtube.com/watch?v=manual002\n",
                encoding="utf-8",
            )
            searched.write_text(
                "[Tech]\n"
                "Tech_Auto_001 https://www.youtube.com/watch?v=dupe001\n"
                "Tech_Auto_002 https://www.youtube.com/watch?v=newtech001\n\n"
                "[News]\n"
                "News_Auto_001 https://www.youtube.com/watch?v=newnews001\n",
                encoding="utf-8",
            )

            result = merge_video_lists(manual, searched, output)
            lines = output.read_text(encoding="utf-8").splitlines()

            self.assertEqual(result["manual_count"], 3)
            self.assertEqual(result["searched_count"], 3)
            self.assertEqual(result["searched_added"], 2)
            self.assertEqual(result["merged_count"], 5)
            self.assertLess(
                lines.index("Tech_Manual_002 https://www.youtube.com/watch?v=dupe001"),
                lines.index("Tech_Auto_002 https://www.youtube.com/watch?v=newtech001"),
            )
            self.assertNotIn("Tech_Auto_001 https://www.youtube.com/watch?v=dupe001", lines)
            self.assertLess(lines.index("[Game]"), lines.index("[News]"))

    def test_merge_video_lists_keeps_all_searched_when_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manual = root / "video_list_manual.txt"
            searched = root / "video_list_searched.txt"
            output = root / "video_list_merged.txt"
            manual.write_text(
                "[Tech]\n"
                "Tech_Manual_001 https://www.youtube.com/watch?v=manual001\n\n",
                encoding="utf-8",
            )
            searched.write_text(
                "[Tech]\n"
                "Tech_Auto_001 https://www.youtube.com/watch?v=newtech001\n\n"
                "[Game]\n"
                "Game_Auto_001 https://www.youtube.com/watch?v=newgame001\n",
                encoding="utf-8",
            )

            result = merge_video_lists(manual, searched, output)

            self.assertEqual(result["manual_count"], 1)
            self.assertEqual(result["searched_added"], 2)
            self.assertEqual(result["merged_count"], 3)
            self.assertIn("Tech_Auto_001 https://www.youtube.com/watch?v=newtech001", output.read_text(encoding="utf-8"))
            self.assertIn("Game_Auto_001 https://www.youtube.com/watch?v=newgame001", output.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
