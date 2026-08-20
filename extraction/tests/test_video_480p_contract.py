from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.video_data_collection import video_processor
from src.video_data_collection.raw_pipeline import ensure_canonical_480p_video


class Video480pContractTests(unittest.TestCase):
    def test_480p_source_is_copied_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")

            with mock.patch.object(video_processor, "get_video_height", return_value=480):
                video_processor.ensure_480p_video(source, canonical)

            self.assertEqual(canonical.read_bytes(), b"source")
            self.assertTrue(source.exists())

    def test_higher_resolution_source_uses_resize_and_validates_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")

            def fake_resize(_source: Path, output: Path) -> None:
                Path(output).write_bytes(b"resized")

            with mock.patch.object(video_processor, "get_video_height", side_effect=[1080, 480]), mock.patch.object(
                video_processor,
                "resize_to_480p",
                side_effect=fake_resize,
            ) as resize:
                video_processor.ensure_480p_video(source, canonical)

            temporary = canonical.with_name(f".{canonical.stem}.480p_tmp{canonical.suffix}")
            resize.assert_called_once_with(source, temporary)
            self.assertEqual(canonical.read_bytes(), b"resized")

    def test_invalid_canonical_is_repaired_when_original_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            canonical = Path(tmp) / "video_480p.mp4"
            canonical.write_bytes(b"640p")

            def fake_resize(_source: Path, output: Path) -> None:
                Path(output).write_bytes(b"480p")

            with mock.patch.object(video_processor, "get_video_height", side_effect=[640, 640, 480]), mock.patch.object(
                video_processor,
                "resize_to_480p",
                side_effect=fake_resize,
            ) as resize:
                video_processor.ensure_480p_video(Path(tmp) / "video.mp4", canonical)

            temporary = canonical.with_name(f".{canonical.stem}.480p_tmp{canonical.suffix}")
            resize.assert_called_once_with(canonical, temporary)
            self.assertEqual(canonical.read_bytes(), b"480p")
            self.assertFalse(temporary.exists())

    def test_invalid_result_is_removed_and_source_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")

            def fake_resize(_source: Path, output: Path) -> None:
                Path(output).write_bytes(b"invalid")

            with mock.patch.object(video_processor, "get_video_height", side_effect=[1080, 720]), mock.patch.object(
                video_processor,
                "resize_to_480p",
                side_effect=fake_resize,
            ):
                with self.assertRaisesRegex(RuntimeError, "height=720"):
                    video_processor.ensure_480p_video(source, canonical)

            self.assertTrue(source.exists())
            self.assertFalse(canonical.exists())

    def test_empty_result_is_removed_and_source_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")

            def fake_resize(_source: Path, output: Path) -> None:
                Path(output).touch()

            with mock.patch.object(video_processor, "get_video_height", return_value=1080), mock.patch.object(
                video_processor,
                "resize_to_480p",
                side_effect=fake_resize,
            ):
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    video_processor.ensure_480p_video(source, canonical)

            self.assertTrue(source.exists())
            self.assertFalse(canonical.exists())

    def test_source_is_deleted_only_after_canonicalization_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")
            canonical.write_bytes(b"canonical")
            processor = mock.Mock()

            ensure_canonical_480p_video(processor, source, canonical)

            processor.ensure_480p_video.assert_called_once_with(source, canonical)
            self.assertFalse(source.exists())
            self.assertTrue(canonical.exists())

    def test_source_is_preserved_when_canonicalization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "video.mp4"
            canonical = Path(tmp) / "video_480p.mp4"
            source.write_bytes(b"source")
            processor = mock.Mock()
            processor.ensure_480p_video.side_effect = RuntimeError("failed")

            with self.assertRaisesRegex(RuntimeError, "failed"):
                ensure_canonical_480p_video(processor, source, canonical)

            self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()
