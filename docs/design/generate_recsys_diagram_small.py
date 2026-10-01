"""Derive an editable small-model diagram and render its PNG via LibreOffice.

Run from any directory with python-pptx, PyMuPDF and LibreOffice installed.
The original large-model PPTX/PNG are preserved.
"""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import pymupdf
from pptx import Presentation
from pptx.util import Inches


def main():
    directory = Path(__file__).resolve().parent
    presentation = Presentation(directory / "recsys_diagram.pptx")
    slide = presentation.slides[0]
    shapes = list(slide.shapes)
    for shape in shapes:
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.text = run.text.replace("1,024", "384").replace("BGE large", "BGE small")
    shapes[1].text_frame.paragraphs[0].runs[0].text = (
        "BGE small-en-v1.5  /  Baseline: Title 512  /  Text & Graph: Title 128 + Video 384"
        "  /  Final LayerNorm: 512  /  Frozen BGE: 384"
    )
    # Remove the Text summary projection and its incoming connector. The existing
    # blue wire now runs directly from the summary's BGE output to the concat.
    for shape in (shapes[36], shapes[38]):
        element = shape._element
        element.getparent().remove(element)
    shapes[39].top = Inches(4.4)
    shapes[39].height = Inches(3.3)
    shapes[60].text_frame.paragraphs[0].runs[0].text = "Shared node/context projection: 384 → 128"
    shapes[61].text_frame.paragraphs[0].runs[0].text = "Shared projection*"
    shapes[26].text_frame.paragraphs[0].runs[0].text = (
        "E / A = entities / actions; S = valid scenes  |  *Node/context projection weights shared"
        "  |  Text summary: direct 384  |  No-title: 128 zeros"
    )
    output = directory / "recsys_diagram_small.pptx"
    presentation.save(output)
    environment = os.environ.copy()
    program = Path(shutil.which("soffice")).resolve().parent
    environment["LD_LIBRARY_PATH"] = str(program) + ":" + environment.get("LD_LIBRARY_PATH", "")
    with tempfile.TemporaryDirectory(prefix="recsys-small-render-") as work:
        work = Path(work)
        subprocess.run([
            "soffice", f"-env:UserInstallation={work.joinpath('lo-profile').as_uri()}",
            "--headless", "--convert-to", "pdf", "--outdir", str(work), str(output),
        ], check=True, env=environment)
        with pymupdf.open(work / "recsys_diagram_small.pdf") as pdf:
            assert len(pdf) == 1
            pdf[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(
                directory / "recsys_diagram_small.png"
            )


if __name__ == "__main__":
    main()
