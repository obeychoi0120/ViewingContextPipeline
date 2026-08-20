from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2400, 1320
BG = "#F7F9FC"
INK = "#172033"
MUTED = "#61708A"
LINE = "#AAB5C5"
SHARED = ("#FFF3D6", "#C48917")
GRAPH = ("#E2F5EA", "#27875A")
DESC = ("#E4EEFF", "#3A6FD8")
ID = ("#F1F3F6", "#657187")
MODEL = ("#F0E9FF", "#7552B8")
EVAL = ("#FFE9E5", "#C6543F")
PARKED = ("#ECEFF4", "#9AA5B5")


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


TITLE = load_font(40, True)
SUBTITLE = load_font(21)
STAGE = load_font(22, True)
LANE = load_font(21, True)
BOX_TITLE = load_font(21, True)
BOX_TEXT = load_font(16)
SMALL = load_font(15)


def box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    title: str,
    body: str,
    palette: tuple[str, str],
    *,
    wrap: int = 28,
    title_size: ImageFont.ImageFont = BOX_TITLE,
) -> None:
    left, top, right, bottom = bounds
    fill, border = palette
    draw.rounded_rectangle(bounds, radius=16, fill=fill, outline=border, width=3)
    draw.text(((left + right) / 2, top + 17), title, font=title_size, fill=INK, anchor="ma")
    wrapped = "\n".join(textwrap.wrap(body, width=wrap))
    draw.multiline_text(
        ((left + right) / 2, top + 52),
        wrapped,
        font=BOX_TEXT,
        fill=MUTED,
        anchor="ma",
        align="center",
        spacing=6,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 4,
) -> None:
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        head = [(x2, y2), (x2 - direction * 14, y2 - 9), (x2 - direction * 14, y2 + 9)]
    else:
        direction = 1 if y2 > y1 else -1
        head = [(x2, y2), (x2 - 9, y2 - direction * 14), (x2 + 9, y2 - direction * 14)]
    draw.polygon(head, fill=color)


def stage(draw: ImageDraw.ImageDraw, x: int, number: str, label: str, color: str) -> None:
    draw.ellipse((x, 152, x + 40, 192), fill=color)
    draw.text((x + 20, 172), number, font=SMALL, fill="white", anchor="mm")
    draw.text((x + 54, 172), label, font=STAGE, fill=INK, anchor="lm")


def lane(draw: ImageDraw.ImageDraw, top: int, label: str, color: str) -> None:
    draw.rounded_rectangle((45, top, 1990, top + 205), radius=20, fill="#FFFFFF", outline="#DEE4ED", width=2)
    draw.text((70, top + 24), label, font=LANE, fill=color)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((55, 45), "MicroLens + SASRec Viewing Context Validation", font=TITLE, fill=INK)
    draw.text((57, 99), "Same cohort and protocol · three independently trained recommendation arms", font=SUBTITLE, fill=MUTED)

    stage(draw, 55, "1", "Data & Split", SHARED[1])
    stage(draw, 520, "2", "Item Representation", GRAPH[1])
    stage(draw, 1450, "3", "SASRec", MODEL[1])
    stage(draw, 2040, "4", "Evaluation", EVAL[1])

    # Shared data contract.
    box(draw, (55, 215, 365, 330), "MicroLens Data", "Interaction pairs + existing MP4", SHARED, wrap=30)
    box(draw, (425, 215, 735, 330), "Eligible Cohort", "Pairs ∩ valid videos · deterministic selection", SHARED, wrap=31)
    box(draw, (795, 215, 1105, 330), "Leave-Two-Out", "Train prefix · validation · test", SHARED, wrap=31)
    box(draw, (1165, 215, 1615, 330), "Controlled Inputs", "Same users, item index, catalog, splits and seeds", SHARED, wrap=45)
    arrow(draw, (365, 272), (425, 272), SHARED[1])
    arrow(draw, (735, 272), (795, 272), SHARED[1])
    arrow(draw, (1105, 272), (1165, 272), SHARED[1])

    draw.rounded_rectangle((55, 350, 1990, 395), radius=12, fill="#FFF9EA", outline=SHARED[1], width=2)
    draw.text((1022, 372), "Graph and Description share fixed_30s keyframes (+5/+15/+25s when present) and one visual-evidence fingerprint", font=BOX_TEXT, fill=MUTED, anchor="mm")

    # Horizontal arm lanes keep every arrow independent.
    lane(draw, 420, "ARM 3 · Non-contextual ID Baseline", ID[1])
    lane(draw, 650, "ARM 1 · Visual-only Graph", GRAPH[1])
    lane(draw, 880, "ARM 2 · Visual-only Description", DESC[1])

    # ID arm.
    box(draw, (90, 475, 330, 585), "Item IDs", "No context input", ID, wrap=22)
    box(draw, (510, 475, 800, 585), "ID Embedding", "Learned end-to-end", ID, wrap=27)
    box(draw, (1450, 460, 1740, 600), "SASRec ID", "Independent model and checkpoint", MODEL, wrap=28)
    box(draw, (1810, 475, 1960, 585), "Scores", "Full catalog", ID, wrap=14)
    arrow(draw, (330, 530), (510, 530), ID[1])
    arrow(draw, (800, 530), (1450, 530), ID[1])
    arrow(draw, (1740, 530), (1810, 530), ID[1])

    # Graph arm.
    box(draw, (90, 705, 310, 815), "Frames", "+ ontology", GRAPH, wrap=18)
    box(draw, (360, 705, 580, 815), "SC_graph", "Scene graphs", GRAPH, wrap=20)
    box(draw, (630, 705, 850, 815), "VC_graph", "Deterministic aggregate", GRAPH, wrap=22)
    box(draw, (900, 705, 1080, 815), "Serialize", "Canonical text", GRAPH, wrap=17)
    box(draw, (1130, 705, 1390, 815), "E_graph", "BGE · 1024D · frozen", GRAPH, wrap=25)
    box(draw, (1450, 690, 1740, 830), "SASRec Graph", "Projection + independent model", MODEL, wrap=28)
    box(draw, (1810, 705, 1960, 815), "Scores", "Full catalog", GRAPH, wrap=14)
    for left, right in [((310, 760), (360, 760)), ((580, 760), (630, 760)), ((850, 760), (900, 760)), ((1080, 760), (1130, 760)), ((1390, 760), (1450, 760)), ((1740, 760), (1810, 760))]:
        arrow(draw, left, right, GRAPH[1])

    # Description arm.
    box(draw, (90, 935, 310, 1045), "Frames", "+ description prompt", DESC, wrap=19)
    box(draw, (360, 935, 580, 1045), "SC_desc", "Scene descriptions", DESC, wrap=20)
    box(draw, (630, 935, 850, 1045), "Aggregate", "Chronological text", DESC, wrap=21)
    box(draw, (900, 935, 1080, 1045), "VC_desc", "Qwen text-only summary", DESC, wrap=18)
    box(draw, (1130, 935, 1390, 1045), "E_desc", "BGE · 1024D · frozen", DESC, wrap=25)
    box(draw, (1450, 920, 1740, 1060), "SASRec Description", "Projection + independent model", MODEL, wrap=28)
    box(draw, (1810, 935, 1960, 1045), "Scores", "Full catalog", DESC, wrap=14)
    for left, right in [((310, 990), (360, 990)), ((580, 990), (630, 990)), ((850, 990), (900, 990)), ((1080, 990), (1130, 990)), ((1390, 990), (1450, 990)), ((1740, 990), (1810, 990))]:
        arrow(draw, left, right, DESC[1])

    # Shared evaluation with three separate input ports.
    box(draw, (2040, 420, 2350, 615), "Ranking Metrics", "HR/NDCG@4/8/10/20\nPrimary: NDCG@10", EVAL, wrap=29)
    box(draw, (2040, 665, 2350, 865), "Non-Inferiority", "Graph vs Description\nPaired bootstrap 95% CI\nMargin: −5%", EVAL, wrap=29)
    box(draw, (2040, 915, 2350, 1085), "Final Report", "Secondary comparisons vs ID\n3 seeds · report_ready", EVAL, wrap=29)
    arrow(draw, (1960, 530), (2040, 530), ID[1])
    arrow(draw, (1960, 760), (2040, 760), GRAPH[1])
    arrow(draw, (1960, 990), (2040, 990), DESC[1])

    # Explicitly reserve VP terminology without adding another recommendation arm.
    box(draw, (55, 1135, 850, 1245), "Excluded in the current experiment", "Gemini multimodal Video Profile (VP): frames + ASR/OCR + metadata", PARKED, wrap=75)
    draw.text((920, 1175), "Claim boundary", font=LANE, fill=EVAL[1])
    draw.text((1090, 1177), "Next-item ranking under this fixed protocol; not semantic correctness, CTR or causal impact.", font=SUBTITLE, fill=MUTED, anchor="lm")

    output = Path(__file__).with_name("microlens_sasrec_recommendation_pipeline.png")
    image.save(output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
