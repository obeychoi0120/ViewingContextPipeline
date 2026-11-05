"""Render the initial description/graph/graph-activity/graph-role-activity examples as matching 3840 x 2160 architecture PNGs.

Run: python docs/design/generate_graph_arm_diagrams.py
Requires python-pptx, PyMuPDF and LibreOffice, as do the existing generators.
Only PNGs are retained; editable slides and PDFs are temporary render inputs.
Implemented architecture: common Residual Scene MLP,
activity hypergraph (A), and optional Role Fusion (B). Labels describe the initial recommendation YAML. Asset filenames remain stable.
No model, extraction, embedding or training is executed.
"""

from pathlib import Path
import shutil
import tempfile

from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt

from generate_main_diagram import Diagram, render


INK, MUTED = "19324F", "62768D"
TEAL, TEAL_FILL = "008B91", "EFF9F8"
PURPLE, PURPLE_FILL = "7139BE", "F3EEFC"
GREEN, GREEN_FILL = "28835D", "EFF8F1"
GRAY, GRAY_FILL = "9AA7B5", "EFF2F6"

ARMS = {
    "d": "Description + Mean",
    "g0": "Graph Encoding + Mean",
    "g1": "Graph + Activity Hypergraph",
    "g2": "Graph + Hypergraph + Role Fusion",
}


LABELS = {"d": "description", "g0": "graph", "g1": "graph-activity", "g2": "graph-role-activity"}


def card(d, name, x, y, w, h, title, *lines, ink=TEAL, fill="FFFFFF", size=16,
         detail_size=13):
    d.box(name, x, y, w, h, fill, ink)
    shape = d.text(name + " text", x + 0.07, y + 0.03, w - 0.14, h - 0.06,
                   [(title, size, INK, True),
                    *((line, detail_size, INK, False) for line in lines)])
    for paragraph in shape.text_frame.paragraphs:
        paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.0
    return shape


def wire(d, name, points, ink=TEAL, *, dashed=False):
    start = len(d.slide.shapes)
    d.wire(name, points, ink)
    for shape in list(d.slide.shapes)[start:]:
        # Remove the inherited Office connector style (including its shadow).
        style = shape._element.find(
            "{http://schemas.openxmlformats.org/presentationml/2006/main}style"
        )
        if style is not None:
            shape._element.remove(style)
        if dashed:
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH


def heading(d, name, x, y, w, text, ink=TEAL):
    d.label(name, x, y, w, 0.32, text, 18, ink, True, PP_ALIGN.LEFT)


def step(d, name, x, y, w, h, title, dimensions, note=None, **style):
    """Put tensor or transformation dimensions in brackets on every processing step."""
    lines = [f"[{dimensions}]"]
    if note:
        lines.append(note)
    return card(d, name, x, y, w, h, title, *lines, **style)


def scene_encoder(d, role_fusion):
    """Graph features with optional B, followed by the common Residual Scene MLP."""
    d.box("Scene encoder panel", 0.55, 2.0, 13.85, 8.50, TEAL_FILL, "B9DFDD")
    heading(d, "Scene encoder heading", 0.85, 2.18, 13.1,
            "1  |  GRAPH SCENE ENCODER")
    columns = (("Title", 0.95, ""), ("Entities", 4.25, "E × "),
               ("Actions", 7.55, "A × "), ("Context", 10.85, ""))
    for name, x, count in columns:
        card(d, name, x, 2.95, 2.95, 0.56, name, size=20)
        step(d, name + " BGE", x, 3.86, 2.95, 0.76, "BGE (frozen)", count + "1024",
             ink=GRAY, fill=GRAY_FILL, size=18, detail_size=16)
        step(d, name + " projection", x, 5.00, 2.95, 0.76, "Linear Projection",
             f"{count}1024 → {count}128", size=18, detail_size=14)
        wire(d, name + " input", [(x + 1.475, 3.51), (x + 1.475, 3.86)], MUTED)
        wire(d, name + " encoding", [(x + 1.475, 4.62), (x + 1.475, 5.00)])

    if role_fusion:
        step(d, "Role fusion", 4.25, 6.14, 6.25, 0.96, "B | Entity → Action Role Fusion",
             "A × 640 → A × 128", "Residual fusion · actor / target / tool / place",
             size=19, detail_size=15)
        wire(d, "Entity roles", [(5.725, 5.76), (5.725, 6.14)])
        wire(d, "Action roles", [(9.025, 5.76), (9.025, 6.14)])
        wire(d, "Entity bypass", [(5.725, 5.95), (4.03, 5.95), (4.03, 7.76), (4.25, 7.76)])
        d.junction("Entity fork", 5.725, 5.95, TEAL)
    else:
        wire(d, "Entity to mean", [(5.725, 5.76), (5.725, 7.43)])
        wire(d, "Action to mean", [(9.025, 5.76), (9.025, 7.43)])
    step(d, "Entity mean", 4.25, 7.43, 2.95, 0.66, "Entity Mean", "E × 128 → 128",
         size=17, detail_size=14)
    step(d, "Action mean", 7.55, 7.43, 2.95, 0.66, "Action Mean", "A × 128 → 128",
         size=17, detail_size=14)
    if role_fusion:
        wire(d, "Fused actions", [(9.025, 7.10), (9.025, 7.43)])
    step(d, "Content X", 4.25, 8.40, 9.55, 0.62,
         "Content X: Entity / Action / Context Concat", "S × 384", size=18,
         detail_size=15)
    for name, x, start in (("Entity", 5.725, 8.09), ("Action", 9.025, 8.09),
                            ("Context", 12.325, 5.76)):
        wire(d, name + " content", [(x, start), (x, 8.40)])

    step(d, "Scene concat", 0.95, 9.45, 6.05, 0.72,
         "Scene Concat: Title + Content", "S × 512", size=18, detail_size=16)
    wire(d, "Content to scene", [(5.725, 9.02), (5.725, 9.45)])
    wire(d, "Title to scene", [(2.425, 5.76), (2.425, 9.45)])
    scene_mlp(d)


def description_encoder(d):
    d.box("Scene encoder panel", 0.55, 2.0, 13.85, 8.50, TEAL_FILL, "B9DFDD")
    heading(d, "Scene encoder heading", 0.85, 2.18, 13.1,
            "1  |  DESCRIPTION SCENE ENCODER")
    for name, x, width in (("Title", 1.2, 128), ("Scene Description", 8.0, 384)):
        step(d, name + " input", x, 2.95, 4.6, 0.90, name, "one text",
             "Each scene encoded separately" if width == 384 else "Shared across a video's scenes",
             size=20, detail_size=16)
        step(d, name + " BGE", x, 4.35, 4.6, 0.90, "BGE (frozen)", "1024",
             ink=GRAY, fill=GRAY_FILL, size=20, detail_size=18)
        step(d, name + " projection", x, 5.80, 4.6, 0.90,
             "Description X: Projection" if width == 384 else "Linear Projection", f"1024 → {width}",
             size=20, detail_size=18)
        wire(d, name + " to BGE", [(x + 2.3, 3.85), (x + 2.3, 4.35)])
        wire(d, name + " to projection", [(x + 2.3, 5.25), (x + 2.3, 5.80)])
    step(d, "Description concat", 0.95, 9.45, 6.05, 0.72,
         "Scene Concat: Title + Description", "S × 512", size=18, detail_size=16)
    wire(d, "Title to concat", [(3.5, 6.70), (3.5, 9.45)])
    wire(d, "Description to concat", [(10.3, 6.70), (10.3, 8.70),
                                      (5.725, 8.70), (5.725, 9.45)])
    scene_mlp(d)


def scene_mlp(d):
    step(d, "Scene MLP", 7.55, 9.45, 6.25, 0.72,
         "Scene H: Residual Scene MLP", "S × 512 → S × 512", size=18, detail_size=16)
    wire(d, "Scene concat to MLP", [(7.0, 9.81), (7.55, 9.81)])


def output(d, note):
    step(d, "Video output", 15.35, 9.33, 7.60, 0.96, "Video Vector", "512", note,
         ink=INK, fill="E8EFF6", size=19, detail_size=14)


def scene_input(d):
    step(d, "Scene MLP output", 15.35, 3.12, 7.60, 0.78,
         "Scene H | Residual Scene MLP Output", "S × 512",
         "Independent scene encoding before readout", ink=PURPLE,
         fill=PURPLE_FILL, size=18, detail_size=14)
    wire(d, "Scene MLP to aggregation", [(13.80, 9.81), (14.45, 9.81),
                                        (14.45, 3.51), (15.35, 3.51)])


def aggregation_heading(d, title, ink, *, segments=False):
    heading(d, "Aggregation heading", 15.15, 2.18, 8.0, "2  |  " + title, ink)
    if segments:
        d.label("Aggregation notes", 15.35, 2.67, 7.60, 0.25,
                "Both segment gates init: 0", 13, ink)


def mean_readout(d):
    aggregation_heading(d, "MEAN READOUT", PURPLE)
    scene_input(d)
    step(d, "Scene mean readout", 15.35, 6.10, 7.60, 1.05,
         "Mean Pooling", "S × 512 → 512", "Equal weight per valid scene",
         ink=PURPLE, fill=PURPLE_FILL, size=22, detail_size=18)
    wire(d, "Scenes to mean", [(19.15, 3.90), (19.15, 6.10)], PURPLE)
    wire(d, "Mean to output", [(19.15, 7.15), (19.15, 9.33)], PURPLE)
    output(d, "Mean over independently encoded scenes")


def segment_units(d):
    scene_input(d)
    step(d, "Prepare segments", 15.35, 4.26, 3.50, 0.90,
         "prepare-module", "M × 2", "Same activity key in adjacent scenes",
         ink=GREEN, fill=GREEN_FILL, size=16, detail_size=12)
    step(d, "Scene to segments", 19.45, 4.26, 3.50, 0.90,
         "Scene → Segment Encoding", "K × 514 → K × 512",
         "Scene means + log(size), size / S",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=12)
    wire(d, "Scenes to segment encoding", [(19.15, 3.90), (19.15, 4.06),
                                           (21.20, 4.06), (21.20, 4.26)], PURPLE)
    wire(d, "Fixed incidence input", [(18.85, 4.71), (19.45, 4.71)], GREEN, dashed=True)


def segment_feedback(d):
    step(d, "Segments to scenes", 15.35, 5.75, 3.50, 1.00,
         "Segment → Scene Feedback", "S × 1024 → S × 512",
         "Gated residual · incident segments",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=12)
    wire(d, "Segment feedback", [(21.20, 5.16), (21.20, 5.45),
                                 (17.10, 5.45), (17.10, 5.75)], GREEN)
    wire(d, "Scene residual input", [(16.15, 3.90), (16.15, 4.06),
                                     (15.10, 4.06), (15.10, 6.25), (15.35, 6.25)], PURPLE)


def segment_readout(d):
    step(d, "Segment readout", 19.45, 5.75, 3.50, 1.00,
         "Segment Readout", "K × 512 → 512", "Mean → Linear → learned gate",
         ink=GREEN, fill=GREEN_FILL, size=16, detail_size=13)
    wire(d, "Segments to readout", [(21.20, 5.45), (21.20, 5.75)], GREEN)
    d.junction("Segment readout fork", 21.20, 5.45, GREEN)
    wire(d, "Segment contribution to output", [(21.20, 6.75), (21.20, 9.33)], GREEN)


def hypergraph_readout(d):
    aggregation_heading(d, "A: ACTIVITY HYPERGRAPH", GREEN, segments=True)
    segment_units(d)
    segment_feedback(d)
    segment_readout(d)
    step(d, "Updated scene mean", 15.35, 7.30, 3.50, 0.90,
         "Updated Scene Mean", "S × 512 → 512", "Mean over segment-updated scenes",
         ink=PURPLE, fill=PURPLE_FILL, size=16, detail_size=12)
    wire(d, "Updated scenes to mean", [(17.10, 6.75), (17.10, 7.30)], GREEN)
    wire(d, "Scene mean to output", [(17.10, 8.20), (17.10, 9.33)], PURPLE)
    output(d, "Scene + segment readouts")


def recommendation(d):
    d.box("Recommendation panel", 0.55, 10.95, 22.90, 1.95, "F5F7FB", "BCCDDD")
    heading(d, "Recommendation heading", 0.85, 11.10, 12.0,
            "3  |  SHARED SASREC RECOMMENDER", INK)
    d.label("Recommender inputs note", 11.80, 11.15, 6.60, 0.24,
            "History / candidate vectors: from video readout", 12, MUTED)
    card(d, "Candidate input", 19.05, 11.15, 4.00, 0.38,
         "Candidate Video [512]", ink=INK, fill="E8EFF6", size=15)
    wire(d, "Video to candidate", [(21.05, 10.29), (21.05, 11.15)], INK)
    specs = ((0.95, 4.2, "History Video Vectors", "L × 512", "Chronological history; L ≤ 10"),
             (5.65, 4.2, "SASRec", "L × 512 → L × 512", "2 blocks · 2 heads"),
             (10.35, 5.5, "User Readout + MLP", "L × 512 → 512", "Last valid position · GELU + LN"),
             (19.05, 4.0, "Dot Product → Ranking", "512, 512 → 1", "One score per candidate"))
    for i, (x, width, title, dimensions, note) in enumerate(specs):
        step(d, f"Recommendation {i}", x, 11.92, width, 0.80, title, dimensions, note,
             ink=INK, fill="FFFFFF", size=16, detail_size=14)
        if i:
            prev = specs[i - 1]
            wire(d, f"Recommendation flow {i}",
                 [(prev[0] + prev[1], 12.32), (x, 12.32)], MUTED)
    wire(d, "Candidate to scoring", [(21.05, 11.53), (21.05, 11.92)], INK)
    d.label("Training note", 0.65, 13.08, 22.70, 0.25,
            "BGE frozen; other blocks train jointly  |  S: scenes · E / A: entities / actions per scene · "
            "K: segments · M: memberships · L: history length  |  Brackets: feature shapes / dimensions",
            12, MUTED, align=PP_ALIGN.LEFT)


def build(arm):
    d = Diagram()
    d.label("Diagram title", 0.60, 0.28, 22.80, 0.60,
            f"{LABELS[arm]}  |  {ARMS[arm]}", 32, INK, True, PP_ALIGN.LEFT)
    d.label("Diagram subtitle", 0.65, 1.01, 22.7, 0.31,
            "INITIAL YAML EXAMPLE  |  Common scene encoder: Residual Scene MLP  |  "
            "Video representation [512] → SASRec", 17, MUTED, align=PP_ALIGN.LEFT)
    for name, x, width, ink, fill in (("Frozen BGE", 0.65, 2.55, GRAY, GRAY_FILL),
                                    ("Scene encoder + MLP", 3.40, 4.5, TEAL, TEAL_FILL),
                                    ("Mean readout", 8.1, 3.2, PURPLE, PURPLE_FILL),
                                    ("A: activity hypergraph", 11.5, 4.5, GREEN, GREEN_FILL)):
        if name.startswith("A:") and arm not in ("g1", "g2"):
            continue
        card(d, "Legend " + name, x, 1.49, width, 0.36, name, ink=ink, fill=fill, size=14)
    d.label("Legend fixed rule note", 17.0, 1.49, 6.25, 0.32,
            "Solid: feature flow; dashed: fixed incidence" if arm in ("g1", "g2") else
            "Solid arrows: feature flow", 12, MUTED)
    d.box("Aggregation panel", 14.85, 2.0, 8.60, 8.50, "FBFCFE", "D6E0E9")
    if arm == "d":
        description_encoder(d)
    else:
        scene_encoder(d, role_fusion=arm == "g2")
    if arm in ("g1", "g2"):
        hypergraph_readout(d)
    else:
        mean_readout(d)
    recommendation(d)
    for shape in d.slide.shapes:
        if shape.name.endswith(" panel"):
            shape.adjustments[0] = 0.035
    validate_bounds(d)
    return d.presentation


def validate_bounds(d):
    """Fail before rendering if a block or text box escapes the canvas."""
    for shape in d.slide.shapes:
        if (shape.left < 0 or shape.top < 0
                or shape.left + shape.width > d.presentation.slide_width
                or shape.top + shape.height > d.presentation.slide_height):
            raise ValueError(f"shape outside canvas: {shape.name}")


def main():
    directory = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="graph-arm-diagrams-") as work:
        for arm in ARMS:
            intermediate = Path(work) / f"recsys_diagram_{arm}.pptx"
            build(arm).save(intermediate)
            render(intermediate, expected_size=(3840, 2160))
            target = directory / intermediate.with_suffix(".png").name
            shutil.copyfile(intermediate.with_suffix(".png"), target)
            print(f"Created {target.name} (3840 × 2160)")


if __name__ == "__main__":
    main()
