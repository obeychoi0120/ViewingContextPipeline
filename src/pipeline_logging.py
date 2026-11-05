"""Console-only settings for each selected step."""

from __future__ import annotations
import json
from pathlib import Path
from arm_registry import generation_registry, generated_arm, select_arms
from pipeline_runtime import CONFIG_PATH


def step_settings(context, step, **options):
    values = {
        "run_id": context.run_id,
        "dataset": context.dataset,
        "config_file": context.root / CONFIG_PATH,
        "force": options.get("force", False),
    }
    if step.startswith("extract-"):
        kind = "graph" if "graph" in step else "description"
        source = options.get("model")
        arm = (
            generation_registry(context.config)[options["arm"]]
            if options.get("arm")
            else generated_arm(context.config, kind, source)
        )
        source = arm.model
        values.update(
            arm=arm.name,
            model=source,
            model_settings=context.config["models"][source],
            source=source,
            prompt=context.prompt_path(options["schema"]),
            output_dir=context.extraction_dir(arm.representation, source, "scenes"),
            max_new_tokens=context.config["extraction"][kind][source]["scene_max_new_tokens"],
        )
    else:
        values["output_dir"] = {
            "prepare-cohort": context.cohort_dir,
            "prepare-input-data": context.keyframes_dir,
            "embed-representations": context.representations_dir,
            "prepare-module": context.representations_dir,
            "run-recommendation": context.recommendations_dir,
            "run-diagnosis": context.diagnosis_dir,
        }[step]
        if step in {"embed-representations", "prepare-module", "run-recommendation", "run-diagnosis"}:
            values["selected_arms"] = list(select_arms(context.config, options.get("target")))
    values.update({key: value for key, value in options.items() if key not in values})
    return values


def log_step_start(context, step, **options):
    print(f"[STEP] {step}", flush=True)
    for key, value in step_settings(context, step, **options).items():
        print(
            f"  {key}: {(str(value) if isinstance(value, (str, Path)) else json.dumps(value))}",
            flush=True,
        )
