from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest import mock

from src.scene_context_extraction.graph_core.prompt import SCENE_EXTRACTION_PROMPT, USER_MESSAGE
from src.scene_context_extraction.gauss.api import (
    GaussApiClient,
    GaussApiError,
    GaussApiGeneration,
    api_workers_from_config,
)
from src.scene_context_extraction.gauss import cli as gauss_cli
from src.scene_context_extraction.gauss.extractor import (
    extract_scene_graph,
    run_gauss_api_inference,
)
from src.scene_context_extraction.gauss.pipeline import (
    GaussApiWorkerPool,
    SceneContextJob,
    gauss_failure_path,
    gauss_context_path,
    gauss_scene_context_path,
    run_scene_context_job,
)


class JsonResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def make_client(**overrides) -> GaussApiClient:
    settings = {
        "base_url": (
            "https://ondevice-aistudio.com/prompt/"
            "GaussA-Gemma-4-E2B-v0.3/api/v1/"
        ),
        "model": "GaussA-Gemma-4-E2B-v0.3",
        "image_url_template": (
            "https://images.example.test/{mode}/resized_keyframes/"
            "{content_id}/{filename}"
        ),
        "timeout_seconds": 10,
        "max_retries": 2,
        "retry_base_seconds": 0.25,
    }
    settings.update(overrides)
    return GaussApiClient(**settings)


class GaussApiClientTests(unittest.TestCase):
    def test_gauss_network_config_sets_process_environment(self) -> None:
        with tempfile.NamedTemporaryFile() as cert_file, mock.patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "https://existing-proxy.example.test",
                "NO_PROXY": "existing.example.test",
            },
            clear=False,
        ):
            gauss_cli.apply_gauss_network_config(
                {
                    "API_PROXY_URL": "http://168.219.61.252:8080",
                    "API_SSL_CERT_FILE": cert_file.name,
                    "API_NO_PROXY": "localhost,127.0.0.1",
                }
            )

            for key in (
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "https_proxy",
                "http_proxy",
            ):
                self.assertEqual(
                    os.environ[key],
                    "http://168.219.61.252:8080",
                )
            self.assertEqual(os.environ["SSL_CERT_FILE"], cert_file.name)
            self.assertEqual(
                os.environ["NO_PROXY"],
                "localhost,127.0.0.1",
            )
            self.assertEqual(os.environ["no_proxy"], os.environ["NO_PROXY"])

    def test_gauss_network_config_validates_before_mutating(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HTTPS_PROXY": "https://existing-proxy.example.test"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "API_SSL_CERT_FILE"):
                gauss_cli.apply_gauss_network_config(
                    {
                        "API_PROXY_URL": "http://168.219.61.252:8080",
                        "API_SSL_CERT_FILE": "/missing/DigitalCity.crt",
                    }
                )
            self.assertEqual(
                os.environ["HTTPS_PROXY"],
                "https://existing-proxy.example.test",
            )

    def test_config_and_template_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "API_BASE_URL"):
            make_client(base_url="ftp://example.test/api/v1")
        with self.assertRaisesRegex(ValueError, "missing required placeholders"):
            make_client(
                image_url_template="https://images.example.test/{filename}"
            )
        with self.assertRaisesRegex(ValueError, "unsupported placeholders"):
            make_client(
                image_url_template=(
                    "https://images.example.test/{content_id}/"
                    "{filename}/{token}"
                )
            )
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            make_client(
                image_url_template="file:///{content_id}/{filename}"
            )
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            make_client(
                image_url_template=(
                    "https://<image-host>/{content_id}/{filename}"
                )
            )
        with self.assertRaisesRegex(ValueError, "API_WORKERS"):
            api_workers_from_config({"API_WORKERS": 0})

    def test_configured_ca_is_passed_to_urlopen(self) -> None:
        ssl_context = object()
        with tempfile.NamedTemporaryFile() as cert_file, mock.patch(
            "src.scene_context_extraction.gauss.api.ssl.create_default_context",
            return_value=ssl_context,
        ) as create_context:
            client = make_client(ssl_cert_file=cert_file.name)

        create_context.assert_called_once_with(cafile=cert_file.name)
        response = JsonResponse(
            {"data": [{"id": "GaussA-Gemma-4-E2B-v0.3"}]}
        )
        with mock.patch(
            "src.scene_context_extraction.gauss.api.urlopen",
            return_value=response,
        ) as urlopen_mock:
            client.ensure_ready()
        self.assertIs(urlopen_mock.call_args.kwargs["context"], ssl_context)

    def test_image_urls_keep_count_order_and_expand_fields(self) -> None:
        client = make_client()
        image_paths = [
            "C:/frames/0000.png",
            "C:/frames/0015.png",
            "C:/frames/0030.png",
            "C:/frames/0045.png",
        ]

        self.assertEqual(
            client.image_urls(
                image_paths[:1],
                mode="fixed_15s",
                content_id="content one",
            ),
            [
                "https://images.example.test/fixed_15s/resized_keyframes/"
                "content%20one/0000.png"
            ],
        )
        self.assertEqual(
            client.image_urls(
                image_paths,
                mode="fixed_15s",
                content_id="content",
            ),
            [
                "https://images.example.test/fixed_15s/resized_keyframes/"
                f"content/{timestamp}.png"
                for timestamp in ("0000", "0015", "0030", "0045")
            ],
        )
        with self.assertRaisesRegex(ValueError, "at least one image"):
            client.image_urls([], mode="fixed_15s", content_id="content")
        with self.assertRaisesRegex(ValueError, "at most 4 images"):
            client.image_urls(
                image_paths + ["C:/frames/0060.png"],
                mode="fixed_15s",
                content_id="content",
            )

    def test_request_payload_uses_image_urls_and_email_options(self) -> None:
        client = make_client()
        response = {
            "choices": [
                {
                    "message": {"content": "{}"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": 17},
        }
        with mock.patch.object(
            client,
            "_request_json",
            return_value=response,
        ) as request_json:
            generation = client.generate(
                [
                    "https://images.example.test/fixed_15s/content/0000.png",
                    "https://images.example.test/fixed_15s/content/0015.png",
                ],
                system_prompt=SCENE_EXTRACTION_PROMPT,
                user_message=USER_MESSAGE,
                max_tokens=1536,
                temperature=0.0,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
            )

        self.assertEqual(generation.generated_tokens, 17)
        self.assertEqual(generation.finish_reason, "length")
        method, path, payload = request_json.call_args.args
        self.assertEqual((method, path), ("POST", "chat/completions"))
        self.assertEqual(payload["model"], "GaussA-Gemma-4-E2B-v0.3")
        self.assertEqual(
            payload["messages"][0],
            {"role": "system", "content": SCENE_EXTRACTION_PROMPT},
        )
        self.assertEqual(payload["messages"][1]["content"][-1]["text"], USER_MESSAGE)
        self.assertEqual(
            [item["type"] for item in payload["messages"][1]["content"]],
            ["image_url", "image_url", "text"],
        )
        self.assertEqual(payload["max_tokens"], 1536)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 0.95)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["repetition_penalty"], 1.0)
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("base64", json.dumps(payload).lower())

    def test_usage_is_optional_and_malformed_choice_is_not_retried(self) -> None:
        client = make_client()
        with mock.patch.object(
            client,
            "_request_json",
            return_value={
                "choices": [
                    {"message": {"content": "{}"}, "finish_reason": "stop"}
                ]
            },
        ):
            generation = client.generate(
                ["https://images.example.test/content/0000.png"],
                system_prompt="system",
                user_message="user",
                max_tokens=1,
                temperature=0,
                top_p=1,
                top_k=1,
                repetition_penalty=1,
            )
        self.assertEqual(generation.generated_tokens, 0)

        with mock.patch.object(
            client,
            "_request_json",
            return_value={"choices": []},
        ) as request_json:
            with self.assertRaisesRegex(GaussApiError, "at least one choice"):
                client.generate(
                    ["https://images.example.test/content/0000.png"],
                    system_prompt="system",
                    user_message="user",
                    max_tokens=1,
                    temperature=0,
                    top_p=1,
                    top_k=1,
                    repetition_penalty=1,
                )
        request_json.assert_called_once()

    def test_readiness_requires_configured_model(self) -> None:
        client = make_client()
        with mock.patch.object(
            client,
            "_request_json",
            return_value={"data": [{"id": "GaussA-Gemma-4-E2B-v0.3"}]},
        ):
            client.ensure_ready()
        with mock.patch.object(
            client,
            "_request_json",
            return_value={"data": [{"id": "another-model"}]},
        ):
            with self.assertRaisesRegex(GaussApiError, "is not available"):
                client.ensure_ready()

    def test_retries_timeout_429_and_503_but_not_400(self) -> None:
        success = JsonResponse(
            {"data": [{"id": "GaussA-Gemma-4-E2B-v0.3"}]}
        )
        for failure in (
            TimeoutError("timed out"),
            HTTPError("url", 429, "rate limited", {"Retry-After": "2"}, None),
            HTTPError("url", 503, "unavailable", {}, None),
        ):
            with self.subTest(failure=repr(failure)):
                client = make_client()
                with mock.patch(
                    "src.scene_context_extraction.gauss.api.urlopen",
                    side_effect=[failure, success],
                ) as urlopen_mock, mock.patch(
                    "src.scene_context_extraction.gauss.api.sleep"
                ) as sleep_mock:
                    client.ensure_ready()
                self.assertEqual(urlopen_mock.call_count, 2)
                sleep_mock.assert_called_once()
                if isinstance(failure, HTTPError) and failure.code == 429:
                    sleep_mock.assert_called_once_with(2.0)

        client = make_client()
        bad_request = HTTPError(
            "url",
            400,
            "bad request",
            {"Content-Type": "application/json"},
            BytesIO(
                json.dumps(
                    {
                        "error": {
                            "message": (
                                "image fetch failed for "
                                "https://images.example.test/private/0000.png"
                            )
                        }
                    }
                ).encode("utf-8")
            ),
        )
        with mock.patch(
            "src.scene_context_extraction.gauss.api.urlopen",
            side_effect=bad_request,
        ) as urlopen_mock, mock.patch(
            "src.scene_context_extraction.gauss.api.sleep"
        ) as sleep_mock:
            with self.assertRaisesRegex(
                GaussApiError,
                "HTTP 400.*image fetch failed for <redacted-url>",
            ):
                client.ensure_ready()
        urlopen_mock.assert_called_once()
        sleep_mock.assert_not_called()

    def test_transport_errors_include_actionable_details(self) -> None:
        client = make_client(max_retries=0)
        not_found = HTTPError(
            "url",
            404,
            "Not Found",
            {"Content-Type": "text/html; charset=utf-8"},
            None,
        )
        with mock.patch(
            "src.scene_context_extraction.gauss.api.urlopen",
            side_effect=not_found,
        ):
            with self.assertRaisesRegex(
                GaussApiError,
                r"HTTP 404 Not Found \(Content-Type: text/html; charset=utf-8\)",
            ):
                client.ensure_ready()

        connection_error = URLError("connection timed out")
        with mock.patch(
            "src.scene_context_extraction.gauss.api.urlopen",
            side_effect=connection_error,
        ):
            with self.assertRaisesRegex(
                GaussApiError,
                "after 1 attempts: connection timed out",
            ):
                client.ensure_ready()


class GaussExtractorTests(unittest.TestCase):
    def test_json_output_failure_does_not_repeat_model_call(self) -> None:
        client = make_client()
        with mock.patch.object(
            client,
            "generate",
            return_value=GaussApiGeneration(
                text="not-json",
                generated_tokens=5,
                finish_reason="stop",
                generation_seconds=0.1,
            ),
        ) as generate:
            observation, warnings = extract_scene_graph(
                client,
                ["C:/frames/0000.png"],
                "content",
                {},
            )
        self.assertIsNone(observation)
        self.assertIn("json_repair_failed", warnings[0])
        generate.assert_called_once()

    def test_one_to_four_images_are_sent_without_padding(self) -> None:
        client = make_client()
        response = GaussApiGeneration(
            text="{}",
            generated_tokens=3,
            finish_reason="length",
            generation_seconds=0.5,
        )
        for image_count in range(1, 5):
            with self.subTest(image_count=image_count), mock.patch.object(
                client,
                "generate",
                return_value=response,
            ) as generate:
                result = run_gauss_api_inference(
                    client,
                    [f"C:/frames/{index:04d}.png" for index in range(image_count)],
                    "content",
                    {
                        "max_new_tokens": 1536,
                        "shot_interval": "shot_wise",
                    },
                )
            sent_urls = generate.call_args.args[0]
            self.assertEqual(len(sent_urls), image_count)
            self.assertEqual(
                [Path(url).name for url in sent_urls],
                [f"{index:04d}.png" for index in range(image_count)],
            )
            self.assertTrue(all("/shot_wise/" in url for url in sent_urls))
            self.assertTrue(result.reached_max_tokens)

    def test_zero_and_five_images_are_scene_failures(self) -> None:
        client = make_client()
        observation, warnings = extract_scene_graph(
            client,
            [],
            "content",
            {},
        )
        self.assertIsNone(observation)
        self.assertIn("No keyframe", warnings[0])

        with mock.patch.object(client, "generate") as generate:
            observation, warnings = extract_scene_graph(
                client,
                [f"C:/frames/{index:04d}.png" for index in range(5)],
                "content",
                {},
            )
        self.assertIsNone(observation)
        self.assertIn("at most 4 images", warnings[0])
        generate.assert_not_called()


class GaussPipelineTests(unittest.TestCase):
    def test_gauss_cli_rejects_gpus(self) -> None:
        with mock.patch.object(
            gauss_cli,
            "parse_args",
            return_value=argparse.Namespace(
                manifest="contracts/manifest.csv",
                limit=None,
                force=False,
                gpus="0",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "API_WORKERS"):
                gauss_cli.main()

    def test_paths_use_gauss_specific_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ,
            {"OUTPUT_SAVE_PATH": tmp},
            clear=False,
        ):
            job = SceneContextJob(
                content_id="demo",
                ref_jsonl="ref.jsonl",
                scene_context_jsonl=str(gauss_scene_context_path("demo")),
                frames_dir="frames",
                timestamp_json="timestamps.json",
            )
            self.assertEqual(
                gauss_scene_context_path("demo").parent.name,
                "scene_context_graph_gaussa_gemma4_e2b_v0_3",
            )
            self.assertEqual(
                gauss_failure_path(job).parent.name,
                "scene_context_graph_gaussa_gemma4_e2b_v0_3",
            )
            self.assertEqual(
                gauss_context_path(job).parent.name,
                "video_context_graph_gaussa_gemma4_e2b_v0_3",
            )

            shot_wise_job = SceneContextJob(
                content_id="demo",
                ref_jsonl="ref.jsonl",
                scene_context_jsonl=str(
                    gauss_scene_context_path("demo", "shot_wise")
                ),
                frames_dir="frames",
                timestamp_json="timestamps.json",
                shot_interval="shot_wise",
            )
            self.assertEqual(
                gauss_scene_context_path("demo", "shot_wise").parent.parent.name,
                "shot_wise",
            )
            self.assertEqual(gauss_failure_path(shot_wise_job).parent.parent.name, "shot_wise")
            self.assertEqual(gauss_context_path(shot_wise_job).parent.parent.name, "shot_wise")

    def test_worker_pool_is_bounded_and_returns_input_order(self) -> None:
        client = make_client()
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        active = 0
        max_active = 0

        def run_task(task, _client, _config):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            if task["scene_order"] < 2:
                barrier.wait(timeout=1)
            time.sleep(0.01 * (4 - task["scene_order"]))
            with lock:
                active -= 1
            return {**task, "summary": {"failed": 0, "warnings": 0}}

        tasks = [
            {"task_id": f"demo:{index}", "scene_order": index}
            for index in range(4)
        ]
        with mock.patch(
            "src.scene_context_extraction.gauss.pipeline._run_gauss_api_task",
            side_effect=run_task,
        ):
            with GaussApiWorkerPool(client, 2, {}) as pool:
                results = pool.run_tasks(tasks)

        self.assertEqual(max_active, 2)
        self.assertEqual(
            [result["task_id"] for result in results],
            [task["task_id"] for task in tasks],
        )

    def test_scene_failure_isolated_output_sorted_and_resume_skips_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "asset" / "fixed_15s" / "resized_keyframes" / "demo"
            frames.mkdir(parents=True)
            for timestamp in (0, 15, 30):
                (frames / f"{timestamp:04d}.png").write_bytes(b"not-read")
            ref_jsonl = root / "demo_ref.jsonl"
            scenes = [
                {"scene_idx": index, "timeline": [{"timestamp": timestamp}]}
                for index, timestamp in enumerate((0, 15, 30))
            ]
            ref_jsonl.write_text(
                "".join(json.dumps(scene) + "\n" for scene in scenes),
                encoding="utf-8",
            )
            timestamps = root / "timestamps.json"
            timestamps.write_text(
                json.dumps(
                    [
                        {"keyframe_timestamps": [timestamp]}
                        for timestamp in (0, 15, 30)
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"OUTPUT_SAVE_PATH": str(root)},
                clear=False,
            ):
                job = SceneContextJob(
                    content_id="demo",
                    ref_jsonl=str(ref_jsonl),
                    scene_context_jsonl=str(gauss_scene_context_path("demo")),
                    frames_dir=str(frames),
                    timestamp_json=str(timestamps),
                )
                client = make_client()

                def generate(image_urls, **_kwargs):
                    filename = Path(image_urls[0]).name
                    if filename == "0015.png":
                        raise GaussApiError("HTTP 503 after retries")
                    if filename == "0000.png":
                        time.sleep(0.03)
                    return GaussApiGeneration(
                        text="{}",
                        generated_tokens=2,
                        finish_reason="stop",
                        generation_seconds=0.1,
                    )

                with mock.patch.object(
                    client,
                    "generate",
                    side_effect=generate,
                ) as generate_mock:
                    with GaussApiWorkerPool(client, 3, {}) as pool:
                        summary = run_scene_context_job(job, pool)

                records = [
                    json.loads(line)
                    for line in Path(job.scene_context_jsonl)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                ]
                failures = [
                    json.loads(line)
                    for line in gauss_failure_path(job)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                ]
                self.assertEqual(summary["success"], 2)
                self.assertEqual(summary["failed"], 1)
                self.assertEqual(
                    [record["scene_idx"] for record in records],
                    [0, 2],
                )
                self.assertEqual(
                    [record["scene_idx"] for record in failures],
                    [1],
                )
                self.assertTrue(gauss_context_path(job).exists())
                self.assertEqual(generate_mock.call_count, 3)

                with mock.patch.object(client, "generate") as resumed_generate:
                    with GaussApiWorkerPool(client, 1, {}) as pool:
                        resumed_summary = run_scene_context_job(job, pool)
                self.assertEqual(resumed_summary["success"], 2)
                self.assertEqual(resumed_summary["failed"], 1)
                resumed_generate.assert_not_called()

                forced_response = GaussApiGeneration(
                    text="{}",
                    generated_tokens=2,
                    finish_reason="stop",
                    generation_seconds=0.1,
                )
                with mock.patch.object(
                    client,
                    "generate",
                    return_value=forced_response,
                ) as forced_generate:
                    with GaussApiWorkerPool(client, 1, {}) as pool:
                        forced_summary = run_scene_context_job(
                            job,
                            pool,
                            force=True,
                        )
                self.assertEqual(forced_summary["success"], 3)
                self.assertEqual(forced_summary["failed"], 0)
                self.assertEqual(forced_generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
