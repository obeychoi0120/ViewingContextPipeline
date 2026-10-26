"""Explicit feature preparation, recommendation training, and diagnosis commands."""

from pipeline_logging import log_step_start
from validation.config import build_validation_config
from validation.recommendation_contracts import resolve_target_arms


def validation_config(context):
    return build_validation_config(
        run_id=context.run_id,
        dataset={key: context.path("data", key) for key in context.config["data"]},
        settings=context.config["validation"],
        model_path=context.path("models", "bge"),
        output_dir=context.run_root,
    )



def embed_representations(context, *, force=False, target=None):
    from validation.graph_inputs import prepare

    resolve_target_arms(target, config=context.config)
    log_step_start(context, "embed-representations", force=force, target=target)
    return prepare(context, target=target, force=force)


def run_recommendation(
    context, *, force=False, workers_per_gpu=1, target=None,
    profile_every=None, profile_operators=False,
):
    from validation.leave_one_out_recommendation import run_leave_one_out

    resolve_target_arms(target, config=context.config)
    log_step_start(context, "run-recommendation", force=force, target=target,
                   workers_per_gpu=workers_per_gpu, profile_every=profile_every,
                   profile_operators=profile_operators)
    context.initialize()
    return run_leave_one_out(
        context, force=force, workers_per_gpu=workers_per_gpu, target=target,
        profile_every=profile_every, profile_operators=profile_operators,
    )


def run_diagnosis(context, *, force=False, target=None, compare_run_id=None):
    from validation.leave_one_out_diagnosis import diagnose

    resolve_target_arms(target, config=context.config)
    log_step_start(context, "run-diagnosis", force=force, target=target,
                   compare_run_id=compare_run_id)
    return diagnose(context, target=target, compare_run_id=compare_run_id)


STEP_HANDLERS = {
    "embed-representations": embed_representations,
    "run-recommendation": run_recommendation,
    "run-diagnosis": run_diagnosis,
}
