import json

from conftest import config_data
from vc_validation.config import ExperimentConfig
from vc_validation.experiment import ARMS, build_report


def test_report_ready_requires_nine_checkpoints_three_arms_and_complete_profiles(tmp_path) -> None:
    config = ExperimentConfig.model_validate(config_data(tmp_path, users=2))
    representations = config.output_dir / "representations"
    representations.mkdir(parents=True)
    (representations / "item_index.json").write_text('{"1": 0, "2": 1}', encoding="utf-8")
    (representations / "representation_manifest.json").write_text('{"graph_completeness": 1.0, "desc_completeness": 1.0, "failure_count": 0}', encoding="utf-8")
    run_manifests = []
    for seed in config.model.seeds:
        for arm in ARMS:
            checkpoint = tmp_path / f"{seed}-{arm}.pt"
            checkpoint.write_bytes(b"checkpoint")
            run_manifests.append({"seed": seed, "arm": arm, "checkpoint": str(checkpoint)})
    rows = []
    for seed in config.model.seeds:
        for user in ("u1", "u2"):
            for arm in ARMS:
                rows.append({"seed": seed, "user_id": user, "arm": arm, "history_stratum": "5-9", "target_frequency_bucket": "low", "rank": 1, "top_items": [0, 1], **{f"{metric}@{cutoff}": 1.0 for cutoff in (4, 8, 10, 20) for metric in ("HR", "NDCG")}})
    report = build_report(config, rows, run_manifests)
    ready = json.loads((config.output_dir / "experiment/report_ready.json").read_text(encoding="utf-8"))
    assert report["result"] == "GRAPH_NON_INFERIOR_TO_DESC"
    assert ready["report_ready"] is True
    assert ready["arm_count"] == 3

    (representations / "representation_manifest.json").write_text('{"graph_completeness": 1.0, "desc_completeness": 1.0, "failure_count": 1}', encoding="utf-8")
    build_report(config, rows, run_manifests)
    not_ready = json.loads((config.output_dir / "experiment/report_ready.json").read_text(encoding="utf-8"))
    assert not_ready["report_ready"] is False
