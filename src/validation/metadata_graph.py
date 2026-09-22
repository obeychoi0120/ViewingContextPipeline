"""V1 title + Qwen graph-summary text fusion, before the existing BGE encoder."""

BRANCH = "graph_qwen_metadata"
ARM = "SASRec_METADATA_GRAPH_QWEN"
TARGET = "METADATA_GRAPH_QWEN"
FUSION_SOURCES = {BRANCH: "graph_qwen"}


def combined_text(title, visual_summary):
    if not isinstance(title, str) or not isinstance(visual_summary, str) or not visual_summary.strip():
        raise ValueError("Metadata fusion requires a title string and a nonempty visual summary")
    return f"Title: {title.strip()}\n\nVisual context:\n{visual_summary.strip()}"


def visual_dependencies(branches):
    if branches is None:
        return None
    return {FUSION_SOURCES.get(branch, branch) for branch in branches}
