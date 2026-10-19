from pathlib import Path
import yaml


def config_data(root: Path, *, users: int = 2) -> dict:
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "config.yaml").read_text())
    settings = config["validation"]
    settings["cohort"]["user_count"] = users
    return {"schema_version": "validation-config/v5", "run_id": "test",
            "dataset": {key: root / name for key, name in {
                "pairs_csv": "pairs.csv", "pairs_tsv": "pairs.tsv", "videos_dir": "videos",
                "titles_csv": "titles.csv"}.items()},
            **settings, "encoder": {**settings["encoder"], "model_path": root / "bge"},
            "output_dir": root / "output"}
