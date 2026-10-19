from __future__ import annotations

from pipeline_logging import log_step_start
from preparation.input_data import prepare_input_data


def prepare_cohort_step(context, *, force=False, plan_only=False):
    from validation.rolling_data import prepare_full_cohort

    log_step_start(context, "prepare-cohort", force=force, plan_only=plan_only)
    context.initialize()
    return prepare_full_cohort(context, plan_only=plan_only)


STEP_HANDLERS = {
    "prepare-cohort": prepare_cohort_step,
    "prepare-input-data": prepare_input_data,
}
