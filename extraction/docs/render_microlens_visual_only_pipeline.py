from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1800, 1050
BACKGROUND = "#F7F9FC"
INK = "#172033"
MUTED = "#62708A"
GRAPH = "#E2F5EA"
GRAPH_BORDER = "#27875A"
DESC = "#E4EEFF"
DESC_BORDER = "#3A6FD8"
SHARED = "#FFF3D6"
SHARED_BORDER = "#C48917"
DOWNSTREAM = "#F0E9FF"
DOWNSTREAM_BORDER = "#7552B8"
PARKED = "#ECEFF4"
PARKED_BORDER = "#9AA5B5"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


TITLE = font(38, True)
SUBTITLE = font(21)
BOX_TITLE = font(24, True)
BOX_TEXT = font(18)
SMALL = font(16)
LANE = font(22, True)


def rounded_box(draw: ImageDraw.ImageDraw, bounds: tuple[int, int, int, int], title: str, body: str, fill: str, border: str, width: int = 3) -> None:
    draw.rounded_rectangle(bounds, radius=18, fill=fill, outline=border, width=width)
    left, top, right, _ = bounds
    draw.text(((left + right) / 2, top + 19), title, font=BOX_TITLE, fill=INK, anchor="ma")
    wrapped = "\n".join(textwrap.wrap(body, width=max(20, int((right - left) / 10.5))))
    draw.multiline_text(((left + right) / 2, top + 60), wrapped, font=BOX_TEXT, fill=MUTED, anchor="ma", align="center", spacing=6)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = MUTED, width: int = 4) -> None:
    draw.line([start, end], fill=color, width=width)
    x2, y2 = end
    x1, y1 = start
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - direction * 14, y2 - 9), (x2 - direction * 14, y2 + 9)]
    else:
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 9, y2 - direction * 14), (x2 + 9, y2 - direction * 14)]
    draw.polygon(points, fill=color)


def routed_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int = 4) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    arrow(draw, points[-2], points[-1], color, width)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((70, 52), "MicroLens Visual-Only Graph vs Description Pipeline", font=TITLE, fill=INK)
    draw.text((72, 105), "Same 30-second scene evidence; OCR, ASR and metadata are excluded from both context arms", font=SUBTITLE, fill=MUTED)

    rounded_box(draw, (70, 165, 410, 305), "Cohort Selection", "vce_selection.jsonl\ninteraction pairs + existing MP4", SHARED, SHARED_BORDER)
    rounded_box(draw, (530, 165, 1010, 305), "Shared Visual Evidence", "fixed_30s scenes · keyframes at 5/15/25s · image SHA-256 and size · one evidence fingerprint", SHARED, SHARED_BORDER)
    arrow(draw, (410, 235), (530, 235), SHARED_BORDER)

    rounded_box(draw, (1160, 165, 1715, 305), "Excluded from Experiment Track", "OCR · ASR · title · genre · category\nLegacy code remains available but is not invoked", PARKED, PARKED_BORDER)
    draw.line([(1010, 235), (1100, 235)], fill=PARKED_BORDER, width=3)
    draw.line([(1100, 215), (1100, 255)], fill=PARKED_BORDER, width=5)
    draw.line([(1120, 215), (1120, 255)], fill=PARKED_BORDER, width=5)

    draw.text((70, 354), "Arm 1 · Graph", font=LANE, fill=GRAPH_BORDER)
    graph_boxes = [
        ((70, 390, 385, 545), "SC_graph", "Qwen3-VL visual Context Graph per scene"),
        ((455, 390, 770, 545), "VC_graph", "Deterministic aggregate across complete scenes"),
        ((840, 390, 1210, 545), "Canonical Serializer", "Fixed English field order; no additional MLLM call"),
        ((1280, 390, 1640, 545), "VP_graph", "Complete text profile + shared evidence fingerprint"),
    ]
    for bounds, title, body in graph_boxes:
        rounded_box(draw, bounds, title, body, GRAPH, GRAPH_BORDER)
    for first, second in zip(graph_boxes, graph_boxes[1:]):
        arrow(draw, (first[0][2], 468), (second[0][0], 468), GRAPH_BORDER)
    routed_arrow(draw, [(750, 305), (750, 340), (230, 340), (230, 390)], SHARED_BORDER)

    draw.text((70, 600), "Arm 2 · Description", font=LANE, fill=DESC_BORDER)
    desc_boxes = [
        ((70, 635, 470, 790), "Scene Descriptions", "Qwen3-VL detailed visual description from the same keyframes"),
        ((570, 635, 1040, 790), "Text-Only Summary", "Chronological descriptions only · 150–300 English words"),
        ((1140, 635, 1540, 790), "VP_desc", "Complete text profile + shared evidence fingerprint"),
    ]
    for bounds, title, body in desc_boxes:
        rounded_box(draw, bounds, title, body, DESC, DESC_BORDER)
    for first, second in zip(desc_boxes, desc_boxes[1:]):
        arrow(draw, (first[0][2], 713), (second[0][0], 713), DESC_BORDER)
    routed_arrow(draw, [(790, 305), (790, 575), (270, 575), (270, 635)], SHARED_BORDER)

    draw.line([(60, 810), (1740, 810)], fill="#CDD5E1", width=3)
    draw.text((70, 825), "Downstream · ViewingContextValidation", font=LANE, fill=DOWNSTREAM_BORDER)

    rounded_box(draw, (70, 900, 490, 1015), "BGE Graph", "bge-large-en-v1.5 · local-only · 1024D · L2 normalized", DOWNSTREAM, DOWNSTREAM_BORDER)
    rounded_box(draw, (580, 900, 1000, 1015), "BGE Description", "Same encoder contract and item order", DOWNSTREAM, DOWNSTREAM_BORDER)
    rounded_box(draw, (1090, 900, 1715, 1015), "Independent SASRec Arms", "ID baseline · Graph · Description · primary NDCG@10 non-inferiority", DOWNSTREAM, DOWNSTREAM_BORDER)
    routed_arrow(draw, [(1640, 468), (1690, 468), (1690, 865), (280, 865), (280, 900)], GRAPH_BORDER, 3)
    routed_arrow(draw, [(1540, 713), (1650, 713), (1650, 890), (790, 890), (790, 900)], DESC_BORDER, 3)
    arrow(draw, (490, 958), (1090, 958), DOWNSTREAM_BORDER)
    arrow(draw, (1000, 958), (1090, 958), DOWNSTREAM_BORDER)

    output = Path(__file__).with_name("microlens_visual_only_pipeline.png")
    image.save(output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
