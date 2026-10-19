"""Small source data and external-runtime doubles; production contracts stay real."""

from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import yaml

from artifact_io import read_json, read_jsonl, write_json
from arm_registry import generation_registry
from extraction.qwen_config import generation_settings
from extraction.scene_storage import write_scene_records
from pipeline_runtime import RunContext
from preparation.cohort import prepare_full_cohort
from validation.leave_one_out_data import EventTable
from visual_sampling import timestamp_stem

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["meta", "graph_qwen3vl_meta"]
GRAPH_TEXT = """[Context]
medium: live_action
format: demonstration
topics: food preparation
[Entities]
p1: person; wearing apron
f1: tomato
t1: knife
l1: kitchen
[Actions]
p1 - cutting - f1; t1; l1
"""
NAMED_GRAPH_TEXT = """[Context]
medium: live_action
format: demonstration
topics: food preparation
[Entities]
person 1; wearing apron
tomato 2
knife
kitchen
[Actions]
person 1 - cutting - tomato 2; knife; kitchen
"""


def graph():
    return {
        "medium": "live_action",
        "format": "demonstration",
        "topics": ["food preparation"],
        "entities": [
            {"id": "p1", "name": "person", "attributes": ["wearing apron"]},
            {"id": "f1", "name": "tomato", "attributes": []},
            {"id": "t1", "name": "knife", "attributes": []},
            {"id": "l1", "name": "kitchen", "attributes": []},
        ],
        "actions": [
            {"actor": "p1", "action": "cutting", "target": "f1", "tool": "t1", "place": "l1"}
        ],
    }


def scene_row(cid, index=0, *, value=None, warnings=None):
    return {
        "content_id": cid,
        "scene_idx": index,
        "tokens": 19,
        "warning": warnings or [],
        "scene_graph": graph() if value is None else value,
    }


def failure_row(cid, index=0):
    return {
        "content_id": cid,
        "scene_idx": index,
        "tokens": 4,
        "error": "interrupted generation",
        "raw_output": "unfinished",
    }


def events(lengths=(5, 6, 13)):
    rows = []
    for user, length in enumerate(lengths, 1):
        for position in range(length):
            index = len(rows)
            rows.append(
                dict(
                    event_id=index,
                    user_id=str(user),
                    item_id=str(index + 1),
                    timestamp=1_700_000_000_000 + position * 1000,
                )
            )
    return rows


def make_context(root, lengths=(5, 6, 13)):
    root.mkdir(parents=True, exist_ok=True)
    config = deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    rows = events(lengths)
    data = root / "data"
    videos = data / "videos"
    videos.mkdir(parents=True)
    with (data / "pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user", "item", "timestamp"])
        writer.writerows([r["user_id"], r["item_id"], r["timestamp"]] for r in rows)
    (data / "titles.csv").write_text(
        "".join(f"{r['item_id']},TITLE_SECRET_{r['item_id']}\n" for r in rows)
    )
    (data / "supplement.csv").write_text('item,title\n1,"Supplement title"\n')
    for row in rows:
        (videos / f"{row['item_id']}.mp4").write_bytes(b"decoder-double input")
    config["data"] = {
        "pairs_csv": str(data / "pairs.csv"),
        "pairs_tsv": str(data / "absent.tsv"),
        "titles_csv": str(data / "titles.csv"),
        "titles_supplement_csv": str(data / "supplement.csv"),
        "videos_dir": str(videos),
    }
    for name, kind, architecture in (
        ("qwen3vl", "qwen3_vl", "Qwen3VLForConditionalGeneration"),
        ("qwen3.5", "qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("bge", "bert", "BertModel"),
    ):
        path = root / "models" / name
        path.mkdir(parents=True)
        write_json(
            path / "config.json",
            {"model_type": kind, "architectures": [architecture], "hidden_size": 1024},
        )
        config["models"][name] = str(path)
    config["models"]["gemini"].update(project_id="offline-project")
    config["artifacts_root"] = str(root / "artifacts")
    settings = config["validation"]
    settings["cohort"].update(
        user_count=len(lengths), interaction_count=len(rows), item_count=len(rows)
    )
    settings["model"].update(max_epochs=2, batch_size=8, dropout=0.0, deterministic=True)
    settings["evaluation"]["bootstrap_samples"] = 32
    settings["graph_execution"].update(feature_cache_mb=0, checkpoint="never")
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    prompts = root / "prompts"
    prompts.mkdir()
    for name in (
        "scene_graph_v9.md",
        "summary_graph_v9.md",
        "scene_description_v3.md",
        "summary_description_v5.md",
    ):
        (prompts / name).write_text((ROOT / "prompts" / name).read_text())
    return RunContext.load("regression", root=root)


def ready_cohort(context):
    context.initialize()
    prepare_full_cohort(context)
    return context.require_ready_cohort()


class FakeEncoder:
    def __init__(self, settings=None, **kwargs):
        self.dim = settings.embedding_dim if settings else 1024
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        result = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            value = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            result.append(value / np.linalg.norm(value))
        self.last_truncated_flags = [False] * len(texts)
        self.last_truncation = {"text_count": len(texts), "truncated_count": 0}
        return np.asarray(result, dtype=np.float32).reshape(len(texts), self.dim)


class FakeGenerator:
    def __init__(self, text=GRAPH_TEXT, *, finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason
        self.tasks = []

    @contextmanager
    def __call__(self, *, runtime=None, **kwargs):
        def generate(tasks, on_task_complete):
            outputs = {}
            for task in tasks:
                self.tasks.append(task)
                text = self.text(task) if callable(self.text) else self.text
                if runtime:
                    runtime.current_result = {
                        "output_tokens": 19,
                        "finish_reason": self.finish_reason,
                    }
                try:
                    on_task_complete(task.task_id, text)
                finally:
                    if runtime:
                        runtime.current_result = None
                outputs[task.task_id] = text
            return outputs

        yield generate


def decode_frames(source, timestamps, output_folder, image_size):
    directory = Path(output_folder)
    directory.mkdir(parents=True, exist_ok=True)
    for timestamp in timestamps:
        Image.new("RGB", tuple(image_size), (30, 80, 120)).save(
            directory / f"{timestamp_stem(timestamp)}.png"
        )
    return {"reused_frames": 0, "extracted_frames": len(timestamps)}


def install_doubles(monkeypatch):
    monkeypatch.setattr("validation.features.BGETextEncoder", FakeEncoder)
    monkeypatch.setattr("preparation.video.media.probe_duration", lambda path: 32.0)
    monkeypatch.setattr("preparation.video.fixed30.extract_resized_keyframes", decode_frames)


def extract_scenes(context, monkeypatch):
    from preparation.input_data import prepare_input_data
    from extraction.steps import extract_graph_scenes

    ready_cohort(context)
    install_doubles(monkeypatch)
    prepare_input_data(context)
    generator = FakeGenerator()
    monkeypatch.setattr("extraction.steps.qwen_generator", generator)
    extract_graph_scenes(
        context, model="qwen3vl", schema="prompts/scene_graph_v9.md", arm="graph_qwen3vl"
    )
    return generator


def prepare_text(context, monkeypatch):
    from extraction.steps import summarize
    from validation.steps import embed_representations

    generator = extract_scenes(context, monkeypatch)
    summary = FakeGenerator("A person cuts a tomato with a knife in a kitchen.")
    monkeypatch.setattr("extraction.steps.qwen_generator", summary)
    summarize(context, model="qwen3vl", schema="prompts/summary_graph_v9.md", arm="graph_qwen3vl")
    embed_representations(context, target=TARGETS)
    return generator, summary


def train_text(context, monkeypatch):
    from validation.steps import run_recommendation

    prepare_text(context, monkeypatch)
    return run_recommendation(context, target=TARGETS)


def summary_options(context, catalog=None):
    return dict(
        arm=generation_registry(context.config)["graph_qwen3vl"],
        schema=context.root / "prompts" / "summary_graph_v9.md",
        catalog=catalog or context.require_ready_cohort()["catalog"],
        provenance={
            "uses_title": False,
            "scene_arm": "graph_qwen3vl",
            "representation": "graph",
            "summary_model": "qwen3vl",
            "model": {"files_signature": "fake"},
            "prompt_hash": "fake",
            "settings": {"backend": "vllm", "generation": {"model": "qwen3vl"}},
        },
        generation=generation_settings("qwen3vl", representation="graph", summary=True),
        model="qwen3vl",
        force=False,
    )


def graph_store(context, *, counts=(2, 1), arm_name="graph_qwen3vl_meta"):
    from arm_registry import registry
    from validation.graph_inputs import build_input

    catalog = [{"item_id": str(i + 1), "content_id": f"video_{i}"} for i in range(len(counts))]
    cohort = {
        "catalog": catalog,
        "metadata_titles": [{**r, "title": f"title {i}"} for i, r in enumerate(catalog)],
    }
    for item, count in zip(catalog, counts, strict=True):
        records = []
        for index in range(count):
            value = graph()
            if index % 2:
                value["actions"][0].update(actor="f1", target="p1", tool=None, place="unknown")
            records.append(scene_row(item["content_id"], index, value=value))
        write_scene_records(
            context.scene_arm_dir("graph_qwen3vl") / f"{item['content_id']}.jsonl", records
        )
    texts, arrays, stats = build_input(context, cohort, registry(context.config)[arm_name])
    encoder = FakeEncoder()
    arrays["features"] = encoder.encode(texts)
    arrays["features"][0] = 0
    return arrays, cohort, texts, stats


def simple_model(item_count=24, dim=8, *, arm="graph", features=None):
    from validation.model import SASRec

    if features is None:
        features = np.random.default_rng(1).normal(size=(item_count, 4)).astype(np.float32)
    return SASRec(item_count, 10, dim, 1, 2, 0.0, arm=arm, item_features=features)


def selected_table(context):
    from validation.selection import load_validation_cohort

    cohort = load_validation_cohort(context)
    return cohort, EventTable(cohort["events"]), cohort["plan"]["splits"][0]


def result_identity(context, cohort, seed=42, arm="meta"):
    from validation.representation_provenance import recommendation_identity
    from validation.selection import training_signature
    from validation.steps import validation_config

    return dict(
        run_id=context.run_id,
        protocol="leave_one_out",
        seed=seed,
        arm=arm,
        split_hash=cohort["plan"]["splits"][0]["split_hash"],
        deterministic=True,
        training_input_hash=training_signature(context, cohort, validation_config(context)),
        **recommendation_identity(context, arm),
    )


def refresh_checksums(directory):
    from validation.shared_cache import checksum

    complete = read_json(directory / "complete.json")
    complete["checksums"] = {name: checksum(directory / name) for name in complete["checksums"]}
    write_json(directory / "complete.json", complete)


def assert_run_grid(context, *, graph_mode=False):
    from validation.leave_one_out_recommendation import combination_complete

    cohort, table, split = selected_table(context)
    total = 0
    for seed in (42, 43, 44):
        for arm in TARGETS:
            directory = context.recommendations_dir / f"seed_{seed}" / arm
            identity = result_identity(context, cohort, seed, arm)
            assert combination_complete(directory, identity, 3)
            training = read_json(directory / "training.json")
            assert training["checkpoint_policy"] == "best_validation_hr10"
            assert len(training["train"]) == 2
            rows = read_jsonl(directory / "per_event_metrics.jsonl")
            assert {r["event_id"] for r in rows} == set(split["phases"]["test"]["event_ids"])
            assert all(r["candidate_count"] == len(table.items) for r in rows)
            assert all(np.isfinite(r["NDCG@10"]) for r in rows)
            if graph_mode:
                import torch
                from validation.graph_model import new_graph_model
                from validation.steps import validation_config

                vectors = np.load(directory / "catalog_vectors.npy")
                assert vectors.shape == (len(table.items), 512)
                assert np.isfinite(vectors).all()
                restored = new_graph_model(context, validation_config(context), arm, "cpu")
                checkpoint = torch.load(
                    directory / "sasrec.pt", map_location="cpu", weights_only=True
                )
                restored.load_state_dict(checkpoint["state_dict"])
                restored.eval()
                with torch.no_grad():
                    np.testing.assert_allclose(
                        vectors, restored.catalog_vectors().numpy(), rtol=1e-5, atol=1e-6
                    )
            total += len(rows)
    assert total == 18


def quiet_progress():
    return SimpleNamespace(
        total=1,
        fp=io.StringIO(),
        complete=lambda **kw: None,
        begin_pass=lambda *a, **kw: None,
        update_stats=lambda *a: None,
        write_log=lambda *a: None,
    )
