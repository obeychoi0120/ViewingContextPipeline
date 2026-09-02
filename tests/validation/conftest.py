from pathlib import Path


def config_data(root: Path, *, users: int = 2) -> dict:
    return {
        "schema_version": "validation-config/v2",
        "run_id": "test",
        "dataset": {
            "pairs_tsv": root / "pairs.tsv",
            "videos_dir": root / "videos",
            "titles_csv": root / "titles.csv",
        },
        "cohort": {
            "user_count": users,
            "seed": 42,
            "min_sequence_length": 5,
            "max_sequence_length": 13,
            "history_strata": [5, 10, 20, 50],
        },
        "encoder": {
            "model_path": root / "bge",
            "embedding_dim": 1024,
            "max_length": 512,
            "batch_size": 2,
        },
        "model": {
            "max_sequence_length": 10,
            "embedding_dim": 512,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "batch_size": 256,
            "max_epochs": 2,
            "patience": 1,
            "learning_rate": 0.001,
            "popularity_power": 1.0,
            "seeds": [42, 43, 44],
        },
        "evaluation": {
            "cutoffs": [4, 8, 10, 20],
            "primary_cutoff": 10,
            "bootstrap_samples": 20,
            "non_inferiority_margin": 0.05,
            "familywise_alpha": 0.05,
            "multiple_comparison_correction": "bonferroni",
            "min_scene_coverage": 0.95,
            "max_arm_coverage_gap": 0.05,
        },
        "output_dir": root / "output",
    }
