"""English titles accompany scene inputs for every extractor and summarizer."""

from unittest.mock import patch

import pytest

import extraction.steps as steps
from extraction.descriptions import description_summary_prompt
from extraction.semantic_graph import graph_summary_prompt
from extraction.recovery import fingerprint
from pipeline_runtime import read_json


@pytest.mark.parametrize("build,records", [
    (description_summary_prompt, [{"schema_version": "scene-description/v2",
                                  "scene_idx": 0, "description": "A literal {english_title}."}]),
    (graph_summary_prompt, [{"scene_idx": 0, "graph": {"context": ["{english_title}"]}}]),
])
@pytest.mark.parametrize("title", ['A "title"\nwith {scenes} and {english_title} 한글', "", "  "])
def test_title_rendering_preserves_input_literals(build, records, title):
    observations = build("{scenes}", records)
    displayed = title.strip() or "(unavailable)"
    expected = f"English Title: {displayed}\n\nScene observations:\n{observations}"
    assert build("English Title: {english_title}\n\nScene observations:\n{scenes}",
                 records, english_title=title) == expected
    assert build("{scenes}", records, english_title=title) == (
        f"English Title: {displayed}\n\n{observations}"
    )


@pytest.mark.parametrize("representation", ["description", "graph"])
@pytest.mark.parametrize("source", ["qwen", "gemini"])
@pytest.mark.parametrize("model", ["qwen", "gemini"])
def test_titles_reach_every_summary_and_input_hash(
    ready_context, fake_models, representation, source, model
):
    context = ready_context
    getattr(steps, f"extract_{representation}_scenes")(
        context, model=source,
        schema=f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md",
    )
    cohort = context.require_ready_cohort()
    titles = {str(row["content_id"]): row["title"] for row in cohort["metadata_titles"]}
    # Lookup must use content identity, not row order.
    original_loader = context.require_ready_cohort
    shuffled = {**cohort, "metadata_titles": list(reversed(cohort["metadata_titles"]))}
    run = getattr(steps, f"summarize_{representation}")
    kwargs = dict(source=source, model=model, schema=f"prompts/{representation}_summary_v4.md")
    fake_models.clear()
    with patch.object(type(context), "require_ready_cohort", return_value=shuffled):
        run(context, **kwargs)
    tasks = [task for batch in fake_models for task in batch]
    assert {task.task_id for task in tasks} == set(titles)
    before = {}
    for task in tasks:
        title = titles[task.task_id]
        header = f"English Title: {title or '(unavailable)'}\n\nScene observations:"
        assert header in task.prompt
        assert task.image_paths == ()
        doc = read_json(context.summary_dir(representation, source, model) / f"{task.task_id}.json")
        prov = doc["provenance"]
        assert prov["english_title"] == title
        assert prov["input_hash"] == fingerprint({k: v for k, v in prov.items() if k != "input_hash"})
        before[task.task_id] = prov
    changed = original_loader()
    changed["metadata_titles"][0]["title"] = "Changed title"
    with patch.object(type(context), "require_ready_cohort", return_value=changed):
        run(context, force=True, **kwargs)
    cid = str(changed["metadata_titles"][0]["content_id"])
    after = read_json(context.summary_dir(representation, source, model) / f"{cid}.json")["provenance"]
    assert after["input_hash"] != before[cid]["input_hash"]
    assert after["scene_input_hash"] == before[cid]["scene_input_hash"]


@pytest.mark.parametrize("representation", ["description", "graph"])
@pytest.mark.parametrize("source", ["qwen", "gemini"])
def test_empty_terminal_fallback_includes_title(ready_context, fake_models, representation, source):
    from extraction.failures import FailureLog

    context = ready_context
    getattr(steps, f"extract_{representation}_scenes")(
        context, model=source,
        schema=f"prompts/{representation}_scene_v{'3' if representation == 'graph' else '2'}.md",
    )
    directory = context.summary_dir(representation, source, "qwen")
    titles = context.require_ready_cohort()["metadata_titles"]
    failures = FailureLog(directory)
    for row in titles:
        failures.record(str(row["content_id"]), None, "empty", "",
                        summary_model="qwen", repetition_penalty=2.0)
    fake_models.clear()
    getattr(steps, f"summarize_{representation}")(
        context, source=source, model="qwen", schema=f"prompts/{representation}_summary_v4.md",
    )
    assert not fake_models
    for row in titles:
        doc = read_json(directory / f"{row['content_id']}.json")
        assert doc["status"] == "raw_fallback"
        assert doc["text"].startswith(f"English Title: {row['title'] or '(unavailable)'}\n\n")
        assert doc["provenance"]["english_title"] == row["title"]
