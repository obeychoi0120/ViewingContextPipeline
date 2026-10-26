"""Verify source files, frozen BGE features, catalog and recorded checksums."""

from validation.graph_inputs import verify
from validation.selection import load_validation_cohort


def verify_representations(context, cohort=None, *, arms=None):
    if arms is None:
        raise ValueError("explicit arms are required for representation verification")
    return verify(context, cohort or load_validation_cohort(context), arms)
