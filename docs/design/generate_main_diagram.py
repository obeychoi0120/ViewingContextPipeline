"""Build the editable pipeline overview and render its PNG from the PPTX.

Run: python docs/design/generate_main_diagram.py
Requires python-pptx, PyMuPDF and LibreOffice. No recommendation model is run.
The overview includes item representations and the shared recommendation flow.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pymupdf
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


FONT = "Liberation Sans"
NAVY = "183450"
MUTED = "536B82"
GRAY = "687581"
BLUE = "2C66B7"
TEAL = "00898D"
WHITE = "FFFFFF"


def color(value):
    return RGBColor.from_string(value)


class Diagram:
    def __init__(self, presentation=None):
        if presentation is not None:
            self.presentation = presentation
            self.slide = presentation.slides[0]
            return
        self.presentation = Presentation()
        self.presentation.slide_width = Inches(24)
        self.presentation.slide_height = Inches(13.5)
        self.slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = color(WHITE)

    def box(self, name, x, y, w, h, fill=WHITE, stroke="CCD7E2"):
        shape = self.slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.name = name
        style = shape._element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}style")
        if style is not None:
            shape._element.remove(style)
        shape.adjustments[0] = 0.12
        shape.fill.solid()
        shape.fill.fore_color.rgb = color(fill)
        shape.line.color.rgb = color(stroke)
        shape.line.width = Pt(1.2)
        return shape

    def text(self, name, x, y, w, h, lines, align=PP_ALIGN.CENTER):
        shape = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        shape.name = name
        frame = shape.text_frame
        frame.clear()
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = Inches(0.025)
        frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for index, (value, size, ink, bold) in enumerate(lines):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.alignment = align
            paragraph.space_before = Pt(0)
            paragraph.space_after = Pt(4 if index < len(lines) - 1 else 0)
            run = paragraph.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color(ink)
        return shape

    def label(self, name, x, y, w, h, value, size=22, ink=MUTED, bold=False,
              align=PP_ALIGN.CENTER):
        return self.text(name, x, y, w, h, [(value, size, ink, bold)], align=align)

    def node(self, name, x, y, w, h, title, subtitle=None, ink=NAVY,
             fill=WHITE, size=24, subtitle_size=22):
        self.box(name, x, y, w, h, fill, ink)
        lines = [(title, size, ink, True)]
        if subtitle:
            lines.append((subtitle, subtitle_size, MUTED, False))
        return self.text(name + " text", x + 0.10, y + 0.06, w - 0.20, h - 0.12, lines)

    def wire(self, name, points, ink=MUTED, arrow=True):
        for index, (start, end) in enumerate(zip(points, points[1:])):
            line = self.slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT,
                Inches(start[0]), Inches(start[1]), Inches(end[0]), Inches(end[1]),
            )
            line.name = f"{name} segment {index + 1}"
            line.line.color.rgb = color(ink)
            line.line.width = Pt(1.8)
            properties = line.line._get_or_add_ln()
            if arrow and index == len(points) - 2:
                tail = OxmlElement("a:tailEnd")
                tail.set("type", "triangle")
                tail.set("w", "sm")
                tail.set("len", "sm")
                properties.append(tail)

    def junction(self, name, x, y, ink=MUTED):
        dot = self.slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x - 0.035), Inches(y - 0.035),
            Inches(0.07), Inches(0.07),
        )
        dot.name = name
        dot.fill.solid()
        dot.fill.fore_color.rgb = color(ink)
        dot.line.fill.background()


def build():
    d = Diagram()
    d.label("Title", 0.6, 0.30, 22.8, 0.60, "ViewingContextPipeline", 38, NAVY,
            True, PP_ALIGN.LEFT)
    d.label("Datasets", 0.6, 0.99, 22.8, 0.32,
            "MicroLens-100K / M³L-10M  |  Title + visual observations for recommendation", 22,
            MUTED, align=PP_ALIGN.LEFT)
    d.box("Experiment configuration", 0.6, 1.50, 22.8, 0.48, "E7EDF5", NAVY)
    d.label("Configuration text", 0.75, 1.53, 22.5, 0.40,
            "rec_experiments_config.yaml: label + source + modules + readout  |  "
            "Titles always included  |  One model per experiment", 20, NAVY)

    for x, w, text in ((0.6, 4.5, "VISUAL OBSERVATIONS"),
                       (5.6, 11.5, "SOURCE-SPECIFIC VIDEO REPRESENTATIONS"),
                       (18.2, 5.2, "TRAIN / EVALUATE")):
        d.label(text, x, 2.20, w, 0.35, text, 24, NAVY, True, PP_ALIGN.LEFT)
    d.box("Visual observations panel", 0.6, 2.70, 4.5, 8.65, "F6F8FB")
    d.box("Metadata panel", 5.6, 2.70, 11.5, 1.70, "F5F6F8")
    d.box("Description panel", 5.6, 4.75, 11.5, 2.70, "F0F5FF", "C6D9F5")
    d.box("Graph panel", 5.6, 7.80, 11.5, 3.55, "EFF9F8", "BADEDC")
    d.box("Recommendation panel", 18.2, 2.70, 5.2, 8.65, "E7EDF5", NAVY)

    d.node("Video", 1.0, 3.30, 3.7, 0.80, "Video")
    d.node("Keyframes", 1.0, 4.60, 3.7, 1.00, "Window sampling",
           "Ordered keyframes", size=22, subtitle_size=20)
    d.node("Extraction", 1.0, 6.00, 3.7, 0.90, "Scene extraction",
           "Qwen3-VL / Gemini", size=22, subtitle_size=19)
    d.node("Descriptions", 1.0, 7.45, 3.7, 1.00, "Scene Descriptions",
           "desc_qwen3vl / desc_gemini", ink=BLUE, size=22, subtitle_size=14)
    d.node("Graphs", 1.0, 9.05, 3.7, 1.00, "Scene Graphs",
           "graph_qwen3vl / graph_gemini", ink=TEAL, size=22, subtitle_size=14)
    d.wire("Video to keyframes", [(2.85, 4.10), (2.85, 4.60)])
    d.wire("Keyframes to extraction", [(2.85, 5.60), (2.85, 6.00)])
    d.wire("Extraction to descriptions", [(2.85, 6.90), (2.85, 7.45)], BLUE)
    d.wire("Extraction to graphs", [(2.85, 6.90), (2.85, 7.15), (0.82, 7.15),
                                     (0.82, 9.55), (1.0, 9.55)], TEAL)
    d.label("Visual scope", 0.9, 10.53, 3.9, 0.32,
            "No titles or audio in extraction", 16, MUTED)

    d.label("Metadata source", 5.9, 2.90, 10.9, 0.27, "source: metadata", 19, GRAY,
            True, PP_ALIGN.LEFT)
    d.node("Metadata title", 5.9, 3.30, 2.75, 0.85, "Title", ink=GRAY, size=22)
    d.node("Metadata BGE", 9.0, 3.30, 2.7, 0.85, "BGE [frozen]", "1,024", ink=GRAY,
           size=20, subtitle_size=18)
    d.node("Metadata model", 12.05, 3.30, 4.75, 0.85, "Projection + Residual MLP",
           "Video [512]", ink=GRAY, size=20, subtitle_size=18)
    d.wire("Metadata title to BGE", [(8.65, 3.725), (9.0, 3.725)], GRAY)
    d.wire("Metadata BGE to model", [(11.7, 3.725), (12.05, 3.725)], GRAY)

    d.label("Description sources", 5.9, 4.98, 10.9, 0.32, "DESCRIPTION SOURCES", 22,
            BLUE, True, PP_ALIGN.LEFT)
    d.node("Description inputs", 5.9, 5.75, 2.75, 1.00, "Title + Description",
           "Each valid scene", ink=BLUE, size=18, subtitle_size=16)
    d.node("Description BGE", 9.0, 5.75, 2.7, 1.00, "BGE [frozen]",
           "Per text: 1,024", ink=BLUE, size=20, subtitle_size=17)
    d.box("Description encoder", 12.05, 5.40, 4.75, 1.70, "E2EEFF", BLUE)
    d.text("Description encoder text", 12.15, 5.47, 4.55, 1.56,
           [("Projections + Scene MLP", 22, BLUE, True),
            ("Title [128] + Description [384]", 17, MUTED, False),
            ("Residual scene representation [512]", 17, MUTED, False),
            ("Mean → Video [512]", 21, BLUE, True)])
    d.wire("Descriptions to inputs", [(4.7, 7.95), (5.30, 7.95),
                                         (5.30, 6.25), (5.9, 6.25)], BLUE)
    d.wire("Description inputs to BGE", [(8.65, 6.25), (9.0, 6.25)], BLUE)
    d.wire("Description BGE to model", [(11.7, 6.25), (12.05, 6.25)], BLUE)

    d.label("Graph sources", 5.9, 8.04, 10.9, 0.32, "GRAPH SOURCES", 22,
            TEAL, True, PP_ALIGN.LEFT)
    d.node("Graph inputs", 5.9, 9.05, 2.75, 1.00, "Title + Graph",
           "Entities / Actions / Context", ink=TEAL, size=21, subtitle_size=13)
    d.node("Graph BGE", 9.0, 9.05, 2.7, 1.00, "BGE [frozen]",
           "Per text: 1,024", ink=TEAL, size=20, subtitle_size=17)
    d.box("Graph encoder", 12.05, 8.55, 4.75, 2.25, "DFF2F0", TEAL)
    d.text("Graph encoder text", 12.15, 8.62, 4.55, 2.11,
           [("Graph scene encoder", 23, TEAL, True),
            ("role_fusion (optional)", 19, MUTED, False),
            ("→ Residual Scene MLP [512]", 19, TEAL, True),
            ("→ activity_hypergraph (optional)", 17, MUTED, False),
            ("→ Mean + optional segment term [512]", 17, TEAL, True)])
    d.wire("Graphs to inputs", [(4.7, 9.55), (5.9, 9.55)], TEAL)
    d.wire("Graph inputs to BGE", [(8.65, 9.55), (9.0, 9.55)], TEAL)
    d.wire("Graph BGE to model", [(11.7, 9.55), (12.05, 9.55)], TEAL)

    # Alternative models supply their own history and candidate vectors.
    for y, ink in ((3.725, GRAY), (6.25, BLUE), (9.55, TEAL)):
        d.wire(f"Video vectors {y}", [(16.8, y), (17.65, y)], ink, arrow=False)
        d.junction(f"Video junction {y}", 17.65, y, NAVY)
    d.wire("Alternative model bus", [(17.65, 3.725), (17.65, 9.55)], NAVY, arrow=False)
    d.node("History vectors", 18.65, 3.35, 4.3, 1.00, "History item vectors",
           "Past interactions", size=23, subtitle_size=20)
    d.node("User model", 18.65, 4.95, 4.3, 1.10, "SASRec + User MLP",
           "User [512]", size=23, subtitle_size=20)
    d.node("Scoring", 18.65, 7.10, 4.3, 1.10, "Full-catalog scores",
           "User · Candidate", size=23, subtitle_size=20)
    d.box("Evaluation", 18.65, 9.10, 4.3, 1.45, WHITE, NAVY)
    d.text("Evaluation text", 18.75, 9.18, 4.1, 1.29,
           [("Validation / Test", 24, NAVY, True),
            ("HR@10 checkpoint selection", 19, MUTED, False),
            ("NDCG@10 / HR@10 reporting", 19, MUTED, False)])
    d.wire("Vectors to history", [(17.65, 3.85), (18.65, 3.85)], NAVY)
    d.wire("History to SASRec", [(20.8, 4.35), (20.8, 4.95)], NAVY)
    d.wire("User to scoring", [(20.8, 6.05), (20.8, 7.10)], NAVY)
    d.wire("Candidate vectors to scoring", [(17.65, 7.65), (18.65, 7.65)], NAVY)
    d.wire("Scoring to evaluation", [(20.8, 8.20), (20.8, 9.10)], NAVY)

    d.box("Training boundary", 0.6, 11.85, 22.8, 0.92, "F6F8FB")
    d.text("Training boundary text", 0.8, 11.94, 22.4, 0.74,
           [("Frozen: BGE features   |   Learnable per experiment: projections, scene encoder, "
             "selected modules, SASRec and User MLP", 20, NAVY, True),
            ("readout: mean   |   Role Fusion before Scene MLP; activity Hypergraph after Scene MLP   |   "
             "Shared source features; independent seed × experiment training", 17, MUTED, False)])
    d.label("Scope note", 0.65, 12.97, 22.7, 0.25,
            "Source identifiers select observations. Module choices select the graph encoder. "
            "Labels name outputs. This overview describes the implemented data flow.", 15, MUTED,
            align=PP_ALIGN.LEFT)
    d.slide.notes_slide.notes_text_frame.text = (
        "Recommendation definitions come from rec_experiments_config.yaml. "
        "Metadata requires titles only. Description uses each valid scene paragraph directly. "
        "Graph uses scene-local entities, actions, context and optional role_fusion and activity_hypergraph. "
        "Titles are always included in recommendation and excluded from extraction. "
        "BGE is frozen; trainable video encoders and SASRec train jointly per seed and experiment. "
        "The representation paths are alternative models, not concatenated final outputs. "
        "Activity segments are prepared from the source graph bundle before training. "
        "Validation HR@10 selects the checkpoint; test NDCG@10/HR@10 are reported."
    )
    for shape in d.slide.shapes:
        if (shape.left < 0 or shape.top < 0
                or shape.left + shape.width > d.presentation.slide_width
                or shape.top + shape.height > d.presentation.slide_height):
            raise ValueError(f"shape outside canvas: {shape.name}")
    return d.presentation


def render(output, *, expected_size=(3840, 2160)):
    environment = os.environ.copy()
    program = Path(shutil.which("soffice")).resolve().parent
    environment["LD_LIBRARY_PATH"] = str(program) + ":" + environment.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="main-diagram-render-") as work:
        work = Path(work)
        subprocess.run([
            "soffice", f"-env:UserInstallation={work.joinpath('lo-profile').as_uri()}",
            "--headless", "--convert-to", "pdf", "--outdir", str(work), str(output),
        ], check=True, env=environment)
        with pymupdf.open(work / output.with_suffix(".pdf").name) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(160 / 72, 160 / 72), alpha=False)
            assert (pixmap.width, pixmap.height) == expected_size
            pixmap.save(output.with_suffix(".png"))


def main():
    output = Path(__file__).resolve().parent / "main_diagram.pptx"
    build().save(output)
    render(output)


if __name__ == "__main__":
    main()
