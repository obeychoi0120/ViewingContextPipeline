"""Generation contracts beside compact scenes, without changing legacy payloads."""

from artifact_io import atomic_write_json, fingerprint, read_json
from model_provenance import canonical
from extraction.errors import ExtractionStepError


def contract_identity(provenance):
    return fingerprint(canonical(provenance))


def prepare_contract(directory, provenance, *, force):
    marker = directory / "generation.json"
    existing = read_json(marker) if marker.is_file() else None
    changed = existing is None or contract_identity(existing) != contract_identity(provenance)
    has_outputs = any(directory.glob("*.jsonl")) or any((directory / "failures").glob("*.jsonl"))
    if changed and has_outputs and not force:
        raise ExtractionStepError("MiniCPM generation conditions changed; rerun extraction with --force")
    if changed and force:
        # Clear the previous generation before publishing a new identity so an
        # interrupted force run cannot mix old scenes with new processor modes.
        for path in [*directory.glob("*.jsonl"), *(directory / "failures").glob("*.jsonl")]:
            path.unlink()
        (directory / "runtime.json").unlink(missing_ok=True)
    if changed:
        atomic_write_json(marker, provenance, durable=True)


def current_contract(context, arm):
    from extraction.steps import prompt_provenance
    from arm_registry import generation_registry

    source = generation_registry(context.config)[arm.scene_arm]
    marker = context.scene_arm_dir(source.name) / "generation.json"
    if not marker.is_file():
        raise ExtractionStepError("MiniCPM generation provenance missing; run extract-graph-scenes")
    stored = read_json(marker)
    expected = prompt_provenance(context, stored["prompt_path"], source)
    if contract_identity(stored) != contract_identity(expected):
        raise ExtractionStepError("MiniCPM generation conditions changed; rerun extraction with --force")
    return stored


def record_runtime(directory, event):
    """Retain engine versions across resumptions of the same generation."""
    path = directory / "runtime.json"
    document = read_json(path) if path.is_file() else {
        "generation_hash": contract_identity(read_json(directory / "generation.json")),
        "executions": {},
    }
    document["executions"][event["execution_id"]] = event["engines"]
    atomic_write_json(path, document, durable=True)


def check_summary_input(context, arm, cid, provenance):
    from extraction.summary_executor import summary_scene_input
    from extraction.step_support import minimal_graph_records

    current_contract(context, arm)
    source = context.scene_arm_dir(arm.scene_arm) / f"{cid}.jsonl"
    _, current = summary_scene_input(source, minimal_graph_records, True, {})
    if current["scene_input_hash"] != provenance.get("scene_input_hash"):
        raise ExtractionStepError("MiniCPM Scene input changed; rerun summarize with --force")
