"""Generate the two HGVRec paper figures as editable PPTX, vector PDF and 600-dpi PNG.

Run from any directory: python /path/to/docs/design/generate_hgvrec_figures.py
Requires python-pptx, PyMuPDF and LibreOffice; no model or dataset is loaded.
All coordinates and font sizes are physical points at the intended ACM print width.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pymupdf
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "WWW2027" / "figures"
FONT = "Liberation Sans"
INK, MUTED, RULE = "20262B", "606970", "C9CDD0"
TEAL, AMBER = "187C80", "A76718"
PALE_TEAL, PALE_AMBER, FROZEN = "F2F8F8", "FBF6EE", "F3F4F5"
WHITE = "FFFFFF"


def rgb(value):
    return RGBColor.from_string(value)


class Figure:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.presentation = Presentation()
        self.presentation.slide_width = Pt(width)
        self.presentation.slide_height = Pt(height)
        self.slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = rgb(WHITE)

    @staticmethod
    def clean(shape):
        # Remove Office theme styles, including inherited shadows and end markers.
        style = shape._element.find(
            "{http://schemas.openxmlformats.org/presentationml/2006/main}style"
        )
        if style is not None:
            shape._element.remove(style)
        return shape

    def text(self, text, x, y, w, h=12, *, size=8.5, bold=False, ink=INK,
             align=PP_ALIGN.CENTER, background=None):
        if size < 8.2:
            raise ValueError("Paper labels must remain at least 8 pt after inclusion")
        shape = self.slide.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
        shape.name = text
        if background:
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(background)
        frame = shape.text_frame
        frame.clear()
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = Pt(0)
        frame.margin_top = frame.margin_bottom = Pt(0)
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, value in enumerate(text.split("\n")):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.alignment = align
            paragraph.space_before = paragraph.space_after = Pt(0)
            paragraph.line_spacing = 1.0
            run = paragraph.add_run()
            run.text = value
            run.font.name = FONT
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = rgb(ink)
        return shape

    def shape(self, x, y, w, h, *, ink=RULE, fill=WHITE, oval=False, dashed=False):
        kind = MSO_SHAPE.OVAL if oval else MSO_SHAPE.ROUNDED_RECTANGLE
        shape = self.clean(self.slide.shapes.add_shape(kind, Pt(x), Pt(y), Pt(w), Pt(h)))
        if not oval:
            shape.adjustments[0] = 0.05
        if fill:
            shape.fill.solid()
            shape.fill.fore_color.rgb = rgb(fill)
        else:
            shape.fill.background()
        shape.line.color.rgb = rgb(ink)
        shape.line.width = Pt(0.65)
        if dashed:
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        return shape

    def box(self, text, x, y, w, h, *, ink=INK, fill=WHITE, size=9, bold=False):
        self.shape(x, y, w, h, ink=ink, fill=fill)
        self.text(text, x + 2, y, w - 4, h, ink=ink, size=size, bold=bold)

    def wire(self, points, *, ink=MUTED, dashed=False, arrow=True):
        for i, (start, end) in enumerate(zip(points, points[1:])):
            shape = self.clean(self.slide.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT, Pt(start[0]), Pt(start[1]), Pt(end[0]), Pt(end[1])
            ))
            shape.name = "Fixed structure" if dashed else "Feature flow"
            shape.line.color.rgb = rgb(ink)
            shape.line.width = Pt(0.65)
            if dashed:
                shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            if arrow and i == len(points) - 2:
                marker = OxmlElement("a:tailEnd")
                marker.set("type", "triangle")
                marker.set("w", "sm")
                marker.set("len", "sm")
                shape.line._get_or_add_ln().append(marker)

    def circle(self, label, x, y, *, radius=8, ink=INK, fill=WHITE):
        self.shape(x - radius, y - radius, radius * 2, radius * 2,
                   ink=ink, fill=fill, oval=True)
        self.text(label, x - radius, y - radius, radius * 2, radius * 2, size=8.5, ink=ink)

    def hypergraph(self, centers, y):
        """An illustrative cooking video: cut, cut+talk, talk, cut+talk, cut."""
        first, second, third, fourth, fifth = centers
        self.shape(first - 17, y - 17, second - first + 34, 34,
                   oval=True, ink=TEAL, fill=None)
        self.shape(second - 19, y - 19, fourth - second + 38, 42,
                   oval=True, ink=TEAL, fill=None, dashed=True)
        self.shape(fourth - 17, y - 17, fifth - fourth + 34, 34,
                   oval=True, ink=TEAL, fill=None)
        self.text("cut₁", (first + second) / 2 - 14, y - 25, 28, 10,
                  ink=TEAL, background=WHITE)
        self.text("talk", third - 14, y + 16, 28, 11, ink=TEAL, background=WHITE)
        self.text("cut₂", (fourth + fifth) / 2 - 14, y - 25, 28, 10,
                  ink=TEAL, background=WHITE)
        for i, x in enumerate(centers, start=1):
            self.circle(f"S{i}", x, y, ink=TEAL)

    def vector(self, x, y, w=30, h=7, *, ink=TEAL):
        """Schematic feature vector; cell shades do not encode measured values."""
        for i in range(6):
            self.shape(x + i * w / 6, y, w / 6 - 0.5, h, ink=ink,
                       fill=ink if i % 3 == 0 else WHITE)

    def table(self, title, x, y, w, rows, *, ink=INK, fill=FROZEN, row_height=13):
        self.shape(x, y, w, 16 + row_height * len(rows), ink=ink)
        self.shape(x, y, w, 16, ink=ink, fill=fill)
        self.text(title, x + 2, y, w - 4, 16, bold=True, size=9, ink=ink)
        for i, text in enumerate(rows):
            self.text(text, x + 3, y + 16 + i * row_height, w - 6, row_height,
                      size=8.5, align=PP_ALIGN.LEFT)

    def scene(self, x, y, w, h, *, cut=True, talk=False):
        """Editable pictogram of a person cutting a carrot and/or talking."""
        self.shape(x, y, w, h, ink=RULE)
        self.shape(x + w * 0.21, y + h * 0.12, w * 0.15, h * 0.20,
                   oval=True, ink=INK, fill=FROZEN)
        self.wire([(x + w * 0.28, y + h * 0.32),
                   (x + w * 0.28, y + h * 0.77)], ink=INK, arrow=False)
        self.wire([(x + w * 0.13, y + h * 0.84),
                   (x + w * 0.89, y + h * 0.84)], ink=INK, arrow=False)
        if cut:
            def polygon(points, *, ink, fill):
                vertices = [(Pt(x + w * px), Pt(y + h * py)) for px, py in points]
                builder = self.slide.shapes.build_freeform(*vertices[0])
                builder.add_line_segments(vertices[1:], close=True)
                shape = self.clean(builder.convert_to_shape())
                shape.fill.solid()
                shape.fill.fore_color.rgb = rgb(fill)
                shape.line.color.rgb = rgb(ink)
                shape.line.width = Pt(0.45)

            # Tapered carrot root, crown and three leaves remain editable vectors.
            polygon(((0.47, 0.79), (0.77, 0.69), (0.82, 0.72),
                     (0.83, 0.77), (0.79, 0.81)), ink=AMBER, fill=AMBER)
            for tip in ((0.84, 0.60), (0.91, 0.65), (0.92, 0.76)):
                self.wire([(x + w * 0.81, y + h * 0.74),
                           (x + w * tip[0], y + h * tip[1])], ink=TEAL, arrow=False)
            for px, py in ((0.62, 0.76), (0.74, 0.73)):
                self.wire([(x + w * px, y + h * py),
                           (x + w * (px + 0.015), y + h * (py + 0.035))],
                          ink=PALE_AMBER, arrow=False)
            # A broad steel blade and separate dark handle make the knife legible.
            polygon(((0.65, 0.52), (0.72, 0.49), (0.79, 0.70),
                     (0.72, 0.74)), ink=INK, fill="DCE2E6")
            polygon(((0.60, 0.39), (0.65, 0.37), (0.70, 0.52),
                     (0.65, 0.55)), ink=INK, fill=INK)
            self.wire([(x + w * 0.28, y + h * 0.45),
                       (x + w * 0.49, y + h * 0.59),
                       (x + w * 0.64, y + h * 0.46)], ink=INK, arrow=False)
        if talk:
            self.shape(x + w * 0.48, y + h * 0.10, w * 0.39, h * 0.22,
                       ink=TEAL, fill=PALE_TEAL)
            for offset in (0.57, 0.67, 0.77):
                self.shape(x + w * offset, y + h * 0.20, 0.7, 0.7,
                           oval=True, ink=TEAL, fill=TEAL)

    def save(self, path):
        for shape in self.slide.shapes:
            if (shape.left < 0 or shape.top < 0
                    or shape.left + shape.width > self.presentation.slide_width
                    or shape.top + shape.height > self.presentation.slide_height):
                raise ValueError(f"Shape outside canvas: {shape.name}")
        self.presentation.save(path)


def motivation():
    f = Figure(242, 201)
    f.shape(0, 0, 242, 26, ink=RULE, fill=FROZEN)
    f.text("Task: Represent a video for\nnext-video recommendation", 3, 1, 236, 24,
           size=9.5, bold=True)
    f.wire([(3, 31), (239, 31)], ink=MUTED)
    actions = ("cut", "cut+talk", "talk", "cut+talk", "cut")
    for i, (x, action) in enumerate(zip((1, 49, 97, 145, 193), actions), start=1):
        f.scene(x, 37, 47, 29, cut="cut" in action, talk="talk" in action)
        f.text(f"S{i}", x, 68, 47, 11, size=8.5, bold=True)
        f.text(action, x, 79, 47, 12, size=8.5)
    f.table("Scene facts", 0, 101, 72,
            ("person, carrot", "knife, kitchen", "cut, talk", "role links", "(per scene)"),
            row_height=13)
    f.table("REAF", 85, 101, 72,
            ("cut + roles", "actor: person", "target: carrot", "tool: knife", "place: kitchen"),
            ink=AMBER, fill=PALE_AMBER, row_height=13)
    f.table("ASHA", 170, 101, 72, ("", "", "", "", ""),
            ink=TEAL, fill=PALE_TEAL, row_height=13)
    for label, y, dashed in (("cut: S1–S2", 122, False), ("talk: S2–S4", 143, True),
                             ("cut: S4–S5", 164, False)):
        f.shape(173, y, 66, 15, ink=TEAL, fill=PALE_TEAL, dashed=dashed)
        f.text(label, 175, y, 62, 15, size=8.5, ink=TEAL)
    f.wire([(74, 145), (83, 145)], ink=INK)
    f.wire([(159, 145), (168, 145)], ink=INK)
    f.text("Role-aware actions + action continuity", 1, 188, 240, 12,
           size=9, bold=True)
    return f


def framework():
    f = Figure(506, 160)
    # Left to right: frozen extraction, REAF inside a scene, ASHA across scenes,
    # then the recommender. Gray fill marks frozen models; colors mark trained modules.
    for x, w, title, color, fill in (
        (0, 116, "1. Scene graphs", INK, FROZEN),
        (122, 124, "2. REAF: role fusion", AMBER, PALE_AMBER),
        (252, 168, "3. ASHA: action segments", TEAL, PALE_TEAL),
        (426, 80, "4. Recommender", INK, FROZEN),
    ):
        f.shape(x, 0, w, 158, ink=RULE, fill=None)
        f.shape(x, 0, w, 16, ink=color, fill=fill)
        f.text(title, x + 2, 0, w - 4, 16, size=9, bold=True, ink=color)

    # 1. One sampled scene becomes a small role-labelled graph.
    f.scene(6, 21, 46, 30, cut=True)
    f.wire([(52, 36), (60, 36)])
    f.box("MLLM\n(frozen)", 60, 22, 50, 28, fill=FROZEN, size=8.5)
    f.wire([(85, 50), (85, 58)])
    f.shape(2, 58, 112, 68, ink=RULE, fill=None)
    for x0, y0, x1, y1 in ((44, 74, 48, 84), (72, 74, 68, 84),
                           (44, 110, 48, 98), (72, 110, 68, 98)):
        f.wire([(x0, y0), (x1, y1)], ink=AMBER, arrow=False)
    for name, x, y in (("person", 6, 61), ("carrot", 66, 61),
                       ("knife", 6, 110), ("kitchen", 66, 110)):
        f.box(name, x, y, 44, 13, size=8.5)
    f.box("cut", 40, 84, 36, 14, size=8.5, bold=True)
    for role, x, y, align in (("actor", 6, 75, PP_ALIGN.LEFT),
                              ("target", 80, 75, PP_ALIGN.RIGHT),
                              ("tool", 6, 99, PP_ALIGN.LEFT),
                              ("place", 80, 99, PP_ALIGN.RIGHT)):
        f.text(role, x, y, 30, 10, size=8.2, ink=AMBER, align=align)
    f.text("context: demonstration", 6, 128, 106, 11, size=8.2, ink=MUTED,
           align=PP_ALIGN.LEFT)
    f.box("title: Vegetable prep", 6, 141, 104, 14, size=8.5)

    # 2. Role-linked entity vectors are fused into the action vector.
    f.text("frozen BGE + projections", 126, 18, 116, 11, size=8.2, ink=MUTED)
    rows = (("action: cut", INK), ("actor: person", AMBER), ("target: carrot", AMBER),
            ("tool: knife", AMBER), ("place: kitchen", AMBER))
    for i, (label, color) in enumerate(rows):
        y = 31 + i * 11
        f.shape(130, y + 1, 14, 9, ink=color, fill=color)
        f.text(label, 149, y, 92, 11, size=8.2, align=PP_ALIGN.LEFT)
    f.wire([(114, 92), (119, 92), (119, 53), (127, 53)])
    f.wire([(137, 86), (137, 92)], ink=AMBER)
    f.box("fuse roles into action", 126, 92, 116, 14, ink=AMBER, fill=PALE_AMBER,
          size=8.5)
    f.text("pool, concat", 128, 108, 56, 11, size=8.2, ink=MUTED,
           align=PP_ALIGN.LEFT)
    f.wire([(194, 106), (194, 121)], ink=AMBER)
    for label, x, w, color in (("Title", 126, 24, MUTED), ("Ent", 150, 28, MUTED),
                               ("Act′", 178, 32, AMBER), ("Ctx", 210, 32, MUTED)):
        f.box(label, x, 121, w, 14, ink=color,
              fill=PALE_AMBER if color == AMBER else WHITE, size=8.2)
    f.wire([(110, 148), (119, 148), (119, 128), (126, 128)])
    f.wire([(184, 135), (184, 140)])
    f.box("residual scene MLP", 126, 140, 116, 14, size=8.5)

    # 3. Scene vectors are nodes; contiguous runs of one action are hyperedges.
    f.text("scene nodes, segment hyperedges", 256, 18, 160, 11, size=8.2, ink=MUTED)
    centers = (280, 308, 336, 364, 392)
    f.wire([(242, 147), (248, 147), (248, 38), (271, 38)], ink=TEAL)
    for x in centers:
        f.wire([(x, 46), (x, 96)], ink=RULE, arrow=False)
    for label, x, y, w, dashed in (("cut₁", 270, 52, 48, False),
                                   ("talk", 298, 67, 76, True),
                                   ("cut₂", 354, 82, 48, False)):
        f.shape(x, y, w, 12, ink=TEAL, fill=PALE_TEAL, dashed=dashed)
        f.text(label, x + 2, y, w - 4, 12, size=8.2, ink=TEAL)
    for i, x in enumerate(centers, start=1):
        f.circle(f"S{i}", x, 38, radius=8.5, ink=TEAL)
    f.box("scene → segment\nmean + size", 256, 100, 78, 24, ink=TEAL,
          fill=PALE_TEAL, size=8.2)
    f.box("segment → scene\ngated residual", 338, 100, 78, 24, ink=TEAL,
          fill=PALE_TEAL, size=8.2)
    f.wire([(295, 124), (295, 132)], ink=TEAL)
    f.wire([(377, 124), (377, 128), (318, 128), (318, 132)], ink=TEAL)
    f.box("video readout", 256, 132, 74, 16, ink=TEAL, fill=PALE_TEAL, size=8.5)
    f.wire([(330, 140), (336, 140)])
    f.box("video embedding", 336, 132, 80, 16, size=8.5, bold=True)

    # 4. The same video encoder embeds history items and candidates.
    f.text("history videos", 430, 18, 72, 11, size=8.2, ink=MUTED)
    for x in (432, 455, 478):
        f.vector(x, 31, 18, 6, ink=MUTED)
    f.wire([(466, 38), (466, 44)])
    f.box("SASRec", 430, 44, 72, 14, size=8.5)
    f.wire([(466, 58), (466, 64)])
    f.box("user MLP", 430, 64, 72, 14, size=8.5)
    f.wire([(466, 78), (466, 90)])
    f.box("dot-product\nscore", 430, 90, 72, 24, size=8.5)
    f.wire([(416, 140), (423, 140), (423, 102), (430, 102)])
    f.wire([(466, 114), (466, 120)])
    f.box("next-video\nranking", 430, 120, 72, 24, size=8.5, bold=True)
    return f


def render(path, width, height):
    program = shutil.which("soffice")
    if program is None:
        raise RuntimeError("LibreOffice (soffice) is required to export the editable figures")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (
        str(Path(program).resolve().parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    with tempfile.TemporaryDirectory(prefix="hgvrec-figure-render-") as work:
        profile = Path(work) / "profile"
        subprocess.run([
            program, f"-env:UserInstallation={profile.as_uri()}", "--headless",
            "--convert-to", "pdf", "--outdir", str(path.parent), str(path),
        ], env=environment, check=True, capture_output=True, text=True, timeout=60)
    with pymupdf.open(path.with_suffix(".pdf")) as pdf:
        if len(pdf) != 1:
            raise ValueError(f"Expected one figure page: {path}")
        page = pdf[0]
        if abs(page.rect.width - width) > 0.1 or abs(page.rect.height - height) > 0.1:
            raise ValueError(f"Unexpected PDF print dimensions: {page.rect}")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["size"] < 8.1:
                        raise ValueError(f"Unexpected font shrinkage: {span['text']}")
                    if not page.rect.contains(pymupdf.Rect(span["bbox"])):
                        raise ValueError(f"Rendered text outside canvas: {span['text']}")
        page.get_pixmap(dpi=600, alpha=False).save(path.with_suffix(".png"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("hgvrec_motivation", motivation), ("hgvrec_framework", framework)):
        figure = build()
        path = OUTPUT / f"{name}.pptx"
        figure.save(path)
        render(path, figure.width, figure.height)
        print(f"Generated {name}.{{pptx,pdf,png}}")


if __name__ == "__main__":
    main()
