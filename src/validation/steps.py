"""Explicit feature/module preparation, recommendation training, and diagnosis commands."""

import json

from pipeline_logging import log_step_start
from rec_experiments import load_experiments
from validation.config import build_validation_config
from validation.recommendation_contracts import resolve_target_arms


def validation_config(context):
    load_experiments(context)
    return build_validation_config(
        run_id=context.run_id,
        dataset={key: context.path("data", key) for key in context.data},
        settings=context.config["validation"],
        model_path=context.path("models", "bge"),
        output_dir=context.run_root,
    )



def embed_representations(context, *, force=False, target=None):
    from validation.graph_inputs import prepare

    load_experiments(context)
    resolve_target_arms(target, config=context.config)
    log_step_start(context, "embed-representations", force=force, target=target)
    return prepare(context, target=target, force=force)


def prepare_module(context, *, force=False, target=None):
    from arm_registry import feature_source, registry
    from validation.scene_segments import attach

    load_experiments(context)
    selected = resolve_target_arms(target, config=context.config)
    arms = registry(context.config)
    log_step_start(context, "prepare-module", force=force, target=target)
    statistics = {}
    prepared = {}
    for name in selected:
        if "segments" not in arms[name].extensions:
            continue
        source = feature_source(arms[name])
        if source not in prepared:
            prepared[source] = attach(context, name)
        statistics[name] = {**prepared[source], "arm": name}
        print(json.dumps(statistics[name], indent=1))
    return {"stage": "prepare-module", "selected_arms": list(selected), "statistics": statistics}


def run_recommendation(
    context, *, force=False, workers_per_gpu=1, target=None,
):
    from validation.leave_one_out_recommendation import run_leave_one_out

    load_experiments(context)
    resolve_target_arms(target, config=context.config)
    log_step_start(context, "run-recommendation", force=force, target=target,
                   workers_per_gpu=workers_per_gpu,
                   deterministic=validation_config(context).model.deterministic)
    context.initialize()
    return run_leave_one_out(
        context, force=force, workers_per_gpu=workers_per_gpu, target=target,
    )


def run_diagnosis(context, *, force=False, target=None, compare_run_id=None):
    from validation.leave_one_out_diagnosis import diagnose

    load_experiments(context)
    target = resolve_target_arms(target, config=context.config)
    log_step_start(context, "run-diagnosis", force=force, target=target,
                   compare_run_id=compare_run_id)
    return diagnose(context, target=target, compare_run_id=compare_run_id)


STEP_HANDLERS = {
    "embed-representations": embed_representations,
    "prepare-module": prepare_module,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
