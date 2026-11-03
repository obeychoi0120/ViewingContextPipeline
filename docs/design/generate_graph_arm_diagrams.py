"""Render the D/G0/G1/G2 suite as matching 3840 x 2160 architecture PNGs.

Run: python docs/design/generate_graph_arm_diagrams.py
Requires python-pptx, PyMuPDF and LibreOffice, as do the existing generators.
Only PNGs are retained; editable slides and PDFs are temporary render inputs.
Implemented architecture: shared Residual Scene MLP, optional Role Fusion (A),
and activity-segment hypergraph (B). Names follow the runtime Arm registry.
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
    "g1": "Graph + Role Fusion (A) + Mean",
    "g2": "Graph + Role Fusion (A) + Hypergraph (B)",
}


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
    """Graph features with optional A, followed by the common Residual Scene MLP."""
    d.box("Scene encoder panel", 0.55, 2.0, 13.85, 5.78, TEAL_FILL, "B9DFDD")
    heading(d, "Scene encoder heading", 0.85, 2.13, 13.1,
            "1  |  GRAPH SCENE ENCODER")
    columns = (("Title", 0.95, ""), ("Entities", 4.25, "E × "),
               ("Actions", 7.55, "A × "), ("Context", 10.85, ""))
    for name, x, count in columns:
        card(d, name, x, 2.69, 2.95, 0.43, name, size=17)
        step(d, name + " BGE", x, 3.34, 2.95, 0.57, "BGE (frozen)", count + "1024",
             ink=GRAY, fill=GRAY_FILL, size=15)
        step(d, name + " projection", x, 4.13, 2.95, 0.57, "Linear Projection",
             f"{count}1024 → {count}128", size=15, detail_size=12)
        wire(d, name + " input", [(x + 1.475, 3.12), (x + 1.475, 3.34)], MUTED)
        wire(d, name + " encoding", [(x + 1.475, 3.91), (x + 1.475, 4.13)])

    if role_fusion:
        step(d, "Role fusion", 4.25, 4.95, 6.25, 0.72, "A | Entity → Action Role Fusion",
             "A × 640 → A × 128", "Residual fusion · actor / target / tool / place",
             size=15, detail_size=12)
        wire(d, "Entity roles", [(5.725, 4.70), (5.725, 4.95)])
        wire(d, "Action roles", [(9.025, 4.70), (9.025, 4.95)])
        wire(d, "Entity bypass", [(5.725, 4.82), (4.03, 4.82), (4.03, 6.06), (4.25, 6.06)])
        d.junction("Entity fork", 5.725, 4.82, TEAL)
    else:
        wire(d, "Entity to mean", [(5.725, 4.70), (5.725, 5.84)])
        wire(d, "Action to mean", [(9.025, 4.70), (9.025, 5.84)])
    step(d, "Entity mean", 4.25, 5.84, 2.95, 0.48, "Entity Mean", "E × 128 → 128",
         size=14, detail_size=12)
    step(d, "Action mean", 7.55, 5.84, 2.95, 0.48, "Action Mean", "A × 128 → 128",
         size=14, detail_size=12)
    if role_fusion:
        wire(d, "Fused actions", [(9.025, 5.67), (9.025, 5.84)])
    step(d, "Content X", 4.25, 6.48, 9.55, 0.48,
         "Content X: Entity / Action / Context Concat", "S × 384", size=14,
         detail_size=12)
    for name, x, start in (("Entity", 5.725, 6.32), ("Action", 9.025, 6.32),
                            ("Context", 12.325, 4.70)):
        wire(d, name + " content", [(x, start), (x, 6.48)])

    step(d, "Scene concat", 0.95, 7.15, 6.05, 0.48,
         "Scene Concat: Title + Content", "S × 512", size=14, detail_size=12)
    wire(d, "Content to scene", [(5.725, 6.96), (5.725, 7.15)])
    wire(d, "Title to scene", [(2.425, 4.70), (2.425, 7.15)])
    step(d, "Scene MLP", 7.55, 7.15, 6.25, 0.48,
         "Scene H: Residual Scene MLP", "S × 512 → S × 512", size=14, detail_size=12)
    wire(d, "Scene concat to MLP", [(7.0, 7.39), (7.55, 7.39)])


def description_encoder(d):
    d.box("Scene encoder panel", 0.55, 2.0, 13.85, 5.78, TEAL_FILL, "B9DFDD")
    heading(d, "Scene encoder heading", 0.85, 2.13, 13.1,
            "1  |  DESCRIPTION SCENE ENCODER")
    for name, x, width in (("Title", 1.2, 128), ("Scene Description", 8.0, 384)):
        step(d, name + " input", x, 2.9, 4.6, 0.8, name, "one text",
             "Each scene encoded separately" if width == 384 else "Shared across a video's scenes")
        step(d, name + " BGE", x, 4.0, 4.6, 0.7, "BGE (frozen)", "1024",
             ink=GRAY, fill=GRAY_FILL)
        step(d, name + " projection", x, 5.0, 4.6, 0.7, "Linear Projection", f"1024 → {width}")
        wire(d, name + " to BGE", [(x + 2.3, 3.7), (x + 2.3, 4.0)])
        wire(d, name + " to projection", [(x + 2.3, 4.7), (x + 2.3, 5.0)])
        wire(d, name + " to concat", [(x + 2.3, 5.7), (x + 2.3, 6.25)])
    step(d, "Description concat", 1.2, 6.25, 11.4, 0.5,
         "Scene Concat: Title + Description", "S × 512")
    step(d, "Scene MLP", 1.2, 7.12, 11.4, 0.5,
         "Scene H: Residual Scene MLP", "S × 512 → S × 512")
    wire(d, "Concat to MLP", [(6.9, 6.75), (6.9, 7.12)])


def design_notes(d, arm):
    d.box("Design panel", 14.75, 2.0, 8.70, 5.78, "F5F7FB", "BCCDDD")
    heading(d, "Design heading", 15.05, 2.13, 8.1, "SHARED ENCODING CONTRACT", INK)
    card(d, "Residual equation", 15.15, 2.85, 7.90, 0.9,
         "Residual Scene MLP", "H = X + ReLU(Linear(X))", "512 → 512; applied independently per scene",
         size=20, detail_size=16)
    card(d, "Fallback", 15.15, 4.05, 7.90, 0.9,
         "No valid scenes: title fallback", "SceneMLP([projected title; zeros(384)])",
         "No title and no valid scenes: zero vector", size=18, detail_size=14)
    modules = {"d": "A: absent   |   B: absent", "g0": "A: absent   |   B: absent",
               "g1": "A: Role Fusion   |   B: absent", "g2": "A: Role Fusion   |   B: activity hypergraph"}
    card(d, "Modules", 15.15, 5.25, 7.90, 0.8, "Modules in this Arm", modules[arm],
         size=18, detail_size=15)
    note = {"d": "D vs G0: input representation / encoding",
            "g0": "G0 vs G1: incremental Role Fusion effect",
            "g1": "G1 vs G2: incremental B module effect",
            "g2": "B combines scene feedback and segment readout"}[arm]
    card(d, "Interpretation", 15.15, 6.45, 7.90, 0.8, "Comparison", note,
         "Shared scene MLP and SASRec in D / G0 / G1 / G2", size=18, detail_size=14)


def output(d, note, *, x=19.85, y=10.04, w=3.20):
    step(d, "Video output", x, y, w, 0.80, "Video Vector", "512", note,
         ink=INK, fill="E8EFF6", size=15, detail_size=12)


def scene_input(d, x=0.95, y=8.83):
    step(d, "Scene input", x, y, 4.20, 0.80,
         "Scene H", "S × 512", "From Residual Scene MLP", ink=PURPLE,
         fill=PURPLE_FILL, size=15, detail_size=12)


def aggregation_heading(d, title, ink, *, segments=False):
    heading(d, "Aggregation heading", 0.85, 8.25, 16.0, "2  |  " + title, ink)
    if segments:
        d.label("Aggregation notes", 17.0, 8.28, 6.0, 0.22,
                "Both segment gates init: 0", 12, ink)


def mean_readout(d):
    aggregation_heading(d, "MEAN OVER VALID SCENES", PURPLE)
    scene_input(d, y=9.40)
    step(d, "Scene mean readout", 8.20, 9.40, 6.10, 0.80,
         "Mean Pooling", "S × 512 → 512", "Equal weight per valid scene",
         ink=PURPLE, fill=PURPLE_FILL, size=18, detail_size=14)
    wire(d, "Scenes to mean", [(5.15, 9.80), (8.20, 9.80)], PURPLE)
    wire(d, "Mean to output", [(14.30, 9.80), (19.40, 9.80)], PURPLE)
    output(d, "Mean over Scene MLP outputs", x=19.40, y=9.40, w=3.65)


def segment_units(d):
    scene_input(d)
    step(d, "Prepare segments", 0.95, 10.04, 4.20, 0.80,
         "prepare-module", "M × 2", "Same activity key in adjacent scenes",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=12)
    step(d, "Scene to segments", 5.65, 8.83, 5.0, 0.80,
         "Scene → Segment Encoding", "K × 514 → K × 512",
         "Scene means + log(size), size / S",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=13)
    wire(d, "Scenes to segment encoding", [(5.15, 9.23), (5.65, 9.23)], PURPLE)
    wire(d, "Fixed incidence input", [(5.15, 10.44), (5.40, 10.44),
                                         (5.40, 9.43), (5.65, 9.43)], GREEN, dashed=True)


def segment_feedback(d, *, w=7.15):
    step(d, "Segments to scenes", 11.15, 8.83, w, 0.80,
         "Segment → Scene Feedback", "S × 1024 → S × 512",
         "Gated residual update from incident segments",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=12)
    wire(d, "Segment feedback", [(10.65, 9.23), (11.15, 9.23)], GREEN)


def segment_readout(d):
    step(d, "Segment readout", 5.65, 10.04, 5.0, 0.80,
         "Segment Readout", "K × 512 → 512", "Mean → Linear → learned gate",
         ink=GREEN, fill=GREEN_FILL, size=15, detail_size=13)
    wire(d, "Segments to readout", [(8.15, 9.63), (8.15, 10.04)], GREEN)
    # One unobstructed skip path below the processing row carries the contribution.
    wire(d, "Segment contribution to output", [(8.15, 10.84), (8.15, 11.09),
                                                  (21.45, 11.09), (21.45, 10.84)], GREEN)


def hypergraph_readout(d):
    aggregation_heading(d, "B | ACTIVITY HYPERGRAPH + SEGMENT CONTRIBUTION", GREEN, segments=True)
    segment_units(d)
    segment_feedback(d)
    segment_readout(d)
    step(d, "Updated scene mean", 11.15, 10.04, 7.15, 0.80,
         "Updated Scene Mean", "S × 512 → 512", "Mean over segment-updated scenes",
         ink=PURPLE, fill=PURPLE_FILL, size=16, detail_size=13)
    wire(d, "Updated scenes to mean", [(14.725, 9.63), (14.725, 10.04)], GREEN)
    wire(d, "Scene mean to output", [(18.30, 10.44), (19.85, 10.44)], PURPLE)
    output(d, "Scene + segment readouts")


def recommendation(d):
    d.box("Recommendation panel", 0.55, 11.55, 22.90, 1.40, "F5F7FB", "BCCDDD")
    heading(d, "Recommendation heading", 0.85, 11.65, 12.0,
            "3  |  SHARED SASREC RECOMMENDER", INK)
    d.label("Recommender inputs note", 11.80, 11.67, 6.60, 0.24,
            "History / candidate vectors: from video readout", 12, MUTED)
    card(d, "Candidate input", 19.05, 11.65, 4.00, 0.33,
         "Candidate Video [512]", ink=INK, fill="E8EFF6", size=13)
    specs = ((0.95, 4.2, "History Video Vectors", "L × 512", "Chronological history; L ≤ 10"),
             (5.65, 4.2, "SASRec", "L × 512 → L × 512", "2 blocks · 2 heads"),
             (10.35, 5.5, "User Readout + MLP", "L × 512 → 512", "Last valid position · GELU + LN"),
             (19.05, 4.0, "Dot Product → Ranking", "512, 512 → 1", "One score per candidate"))
    for i, (x, width, title, dimensions, note) in enumerate(specs):
        step(d, f"Recommendation {i}", x, 12.14, width, 0.65, title, dimensions, note,
             ink=INK, fill="FFFFFF", size=14, detail_size=12)
        if i:
            prev = specs[i - 1]
            wire(d, f"Recommendation flow {i}",
                 [(prev[0] + prev[1], 12.465), (x, 12.465)], MUTED)
    wire(d, "Candidate to scoring", [(21.05, 11.98), (21.05, 12.14)], INK)
    d.label("Training note", 0.65, 13.08, 22.70, 0.25,
            "BGE frozen; other blocks train jointly  |  S: scenes · E / A: entities / actions per scene · "
            "K: segments · M: memberships · L: history length  |  Brackets: feature shapes / dimensions",
            12, MUTED, align=PP_ALIGN.LEFT)


def build(arm):
    d = Diagram()
    d.label("Diagram title", 0.60, 0.28, 22.80, 0.60,
            f"{arm.upper()}  |  {ARMS[arm]}", 32, INK, True, PP_ALIGN.LEFT)
    d.label("Diagram subtitle", 0.65, 1.01, 22.7, 0.31,
            "FIVE-ARM EXPERIMENT  |  Residual Scene MLP retained in every scene encoder  |  "
            "Video representation [512] → SASRec", 17, MUTED, align=PP_ALIGN.LEFT)
    for name, x, width, ink, fill in (("Frozen BGE", 0.65, 2.55, GRAY, GRAY_FILL),
                                    ("Scene encoder + MLP", 3.40, 4.5, TEAL, TEAL_FILL),
                                    ("Mean readout", 8.1, 3.2, PURPLE, PURPLE_FILL),
                                    ("B: activity hypergraph", 11.5, 4.5, GREEN, GREEN_FILL)):
        if name.startswith("B:") and arm != "g2":
            continue
        card(d, "Legend " + name, x, 1.49, width, 0.32, name, ink=ink, fill=fill, size=12)
    d.label("Legend fixed rule note", 17.0, 1.49, 6.25, 0.32,
            "Named inputs link panels; dashed: fixed incidence" if arm == "g2" else
            "Named inputs link panels; solid: feature flow", 11, MUTED)
    d.box("Aggregation panel", 0.55, 8.10, 22.90, 3.15, "FBFCFE", "D6E0E9")
    if arm == "d":
        description_encoder(d)
    else:
        scene_encoder(d, role_fusion=arm in ("g1", "g2"))
    design_notes(d, arm)
    if arm == "g2":
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
