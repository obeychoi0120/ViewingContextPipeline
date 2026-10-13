"""Draw the mean / gnn / attention scene-aggregation comparison figures.

Run: PYTHONPATH=src python docs/design/generate_scene_aggregation_comparison.py
Requires matplotlib, torch and a Noto Sans CJK font. No recommendation model is trained;
figure 4 evaluates the real adjacency/pooling modules on one synthetic five-scene video.
Mirrors src/validation/graph_model.py, scene_gcn.py, scene_attention.py,
scene_transformer.py and graph_context.py (mean v9, gnn v10, transformer v11, attention v12).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import torch

from validation.scene_attention import SceneAttentionGCN, SceneAttentionPool
from validation.scene_gcn import SceneGCN, normalized_similarity
from validation.scene_transformer import SceneTransformer
from validation.graph_context import SCENE_TRANSFORMER

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]

OUT = Path(__file__).parent
INK, MUTED, LINE, SURFACE = "#1f2a37", "#5b6675", "#c9ced4", "#fcfcfb"
# Mechanism colors (validated categorical slots 1-3) plus a neutral for "no learned weight".
FIXED, FIXED_BG = "#8a8f98", "#f1f2f4"
COSINE, COSINE_BG = "#2a78d6", "#e8f1fc"
ATTN, ATTN_BG = "#eb6834", "#fdeee7"
BLOCK, BLOCK_BG = "#1baf7a", "#e5f6ef"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def canvas(width, height, scale=10):
    fig = plt.figure(figsize=(width / scale, height / scale), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", *, fc="white", ec=LINE, size=11, color=INK, weight="normal",
        lw=1.3, ha="center", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, alpha=alpha))
    if text:
        tx = x + w / 2 if ha == "center" else x + 1.2
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color,
                weight=weight, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=1.4, rad=0.0, style="-|>", scale=13, z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=scale,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=z))


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(path)


# ---------------------------------------------------------------- synthetic video
def synthetic_video():
    """Five scenes: s0-s2 share one topic, s3-s4 another (cosine ~0.83 within, ~0.2 across)."""
    torch.manual_seed(0)
    unit = lambda v: torch.nn.functional.normalize(v, dim=-1)
    first = unit(torch.randn(384))
    second = unit(unit(torch.randn(384)) + 0.25 * first)
    features = torch.stack([v + 0.45 * unit(torch.randn(384)) for v in [first] * 3 + [second] * 2])
    scenes = torch.randn(5, 512) / math.sqrt(512)
    return features, scenes


@torch.no_grad()
def toy_weights():
    features, scenes = synthetic_video()
    unit = torch.nn.functional.normalize(features, dim=-1)
    cosine = unit @ unit.T
    gnn = normalized_similarity(features)
    torch.manual_seed(42)
    gnn_attn = SceneAttentionGCN(512, 384, 128).adjacency(features)
    pool = SceneAttentionPool(512, 128)  # score weight starts at 0, as in GraphSASRec
    videos = torch.zeros(5, dtype=torch.long)
    beta = pool.weights(scenes, videos, 1)
    entropy = -(gnn_attn * gnn_attn.log()).sum(-1) / math.log(5)
    return {"cosine": cosine, "gnn": gnn, "gnn_attn": gnn_attn, "beta": beta,
            "entropy": entropy}


def parameter_counts():
    count = lambda m: sum(p.numel() for p in m.parameters())
    transformer = SceneTransformer(512, dropout=0.1, position="none", attention="all_scenes",
                                   **SCENE_TRANSFORMER)
    return {"mean": 0, "attn_pool": count(SceneAttentionPool(512, 128)),
            "gnn": count(SceneGCN(512)), "gnn_attn": count(SceneAttentionGCN(512, 384, 128)),
            "transformer": count(transformer)}


# ---------------------------------------------------------------- figure 1: pipeline
def figure_pipeline(params):
    fig, ax = canvas(230, 118)
    ax.text(3, 113, "그림 1. 공통 파이프라인과 aggregation이 바꾸는 두 자리",
            fontsize=20, weight="bold", color=INK, va="center")
    ax.text(3, 108, "모든 방식은 장면 벡터 H까지 동일하고, ① 장면 간 교환과 ② 영상 readout 중 "
            "한 곳만 바꿉니다. SASRec·loss·입력 특징 캐시는 공유합니다.",
            fontsize=12.5, color=MUTED, va="center")

    top, h = 82, 16
    stages = [
        (3, 30, "장면 s의 Scene Graph\nEntity · Action · Context\n+ 영상 제목", "white", LINE),
        (38, 34, "장면 readout\nx_s = [ t ; ē ; ā′ ; c ]\n128 × 4 = 512", "white", LINE),
        (77, 30, "Scene MLP (1층)\nh_s = x_s + ReLU(W x_s)\n→ H  (S × 512)", "white", LINE),
        (112, 38, "① 장면 간 교환\nH → H′  (S × 512)\n장면끼리 정보를 섞음", "white", INK),
        (155, 34, "② 영상 readout\nH′ → v  (512)\n장면 S개 → 영상 1개", "white", INK),
        (194, 33, "SASRec\n최근 시청 v들 → 사용자 u\n점수 = u · v_cand", "white", LINE),
    ]
    for x, w, text, fc, ec in stages:
        box(ax, x, top, w, h, text, fc=fc, ec=ec, size=11.5, lw=2.2 if ec == INK else 1.3)
    for (x1, w1, *_), (x2, *_) in zip(stages, stages[1:]):
        arrow(ax, x1 + w1 + 0.4, top + h / 2, x2 - 0.4, top + h / 2, color=INK, lw=1.6)
    ax.text(131, top + h + 3.2, "교체 자리 ①", ha="center", fontsize=11.5, weight="bold",
            color=INK)
    ax.text(172, top + h + 3.2, "교체 자리 ②", ha="center", fontsize=11.5, weight="bold",
            color=INK)
    # Similarity features leave before the scene MLP and skip the title.
    arrow(ax, 55, top - 0.4, 120, top - 0.4, color=COSINE, lw=1.5, rad=0.28)
    ax.text(88, 70.2, "z_s = [ ē ; ā′ ; c ]  384  (제목 제외, Scene MLP 이전)\n"
            "→ gnn · gnn_attn의 이웃 가중치 입력", ha="center", va="center", fontsize=10.5,
            color=INK)

    # Slot table aligned under the pipeline columns.
    cols = {"name": (3, 30), "input": (38, 69), "slot1": (112, 38), "slot2": (155, 34),
            "start": (194, 33)}
    header_y = 60
    for key, label in [("name", "aggregation"), ("input", "가중치를 계산하는 입력"),
                       ("slot1", "① 장면 간 교환"), ("slot2", "② 영상 readout"),
                       ("start", "학습 시작점 · 추가 파라미터")]:
        x, w = cols[key]
        ax.text(x + w / 2, header_y, label, ha="center", va="center", fontsize=11.5,
                weight="bold", color=MUTED)
    ax.plot([3, 227], [header_y - 3, header_y - 3], color=LINE, lw=1)

    p = params
    rows = [
        ("mean  (A · 기준)", "없음 — 모든 장면 1/S",
         ("없음", FIXED, FIXED_BG), ("평균  (1/S) Σ h_s", FIXED, FIXED_BG),
         f"— · +{p['mean']:,}"),
        ("attn_pool  (B)", "h_s  (Scene MLP 출력, 제목 포함)",
         ("없음", FIXED, FIXED_BG), ("학습 attention 가중합  Σ β_s h_s", ATTN, ATTN_BG),
         f"mean과 동일 (w = 0) · +{p['attn_pool']:,}"),
        ("gnn  (C)", "z_s 384  (제목 제외)",
         ("cosine GCN 1층  (고정 규칙)", COSINE, COSINE_BG), ("평균", FIXED, FIXED_BG),
         f"— · +{p['gnn']:,}"),
        ("gnn_attn  (D)", "z_s 384  (제목 제외, gnn과 같음)",
         ("attention GCN 1층  (학습 q·k)", ATTN, ATTN_BG), ("평균", FIXED, FIXED_BG),
         f"거의 균등 attention · +{p['gnn_attn']:,}"),
        ("transformer ×3", "H  (+ 위치 P: _pos만)",
         ("Transformer 블록 1층  (2 heads + FFN)", BLOCK, BLOCK_BG), ("평균", FIXED, FIXED_BG),
         f"mean과 동일 (α = 0) · +{p['transformer']:,}"),
    ]
    y = 49
    for name, source, slot1, slot2, start in rows:
        x, w = cols["name"]
        ax.text(x + 1, y, name, va="center", fontsize=12, weight="bold", color=INK)
        x, w = cols["input"]
        ax.text(x + w / 2, y, source, ha="center", va="center", fontsize=11, color=INK)
        for key, (text, ec, fc) in (("slot1", slot1), ("slot2", slot2)):
            x, w = cols[key]
            box(ax, x + 0.6, y - 3.2, w - 1.2, 6.4, text, fc=fc, ec=ec, size=10.5, lw=1.6)
        x, w = cols["start"]
        ax.text(x + w / 2, y, start, ha="center", va="center", fontsize=10.5, color=INK)
        y -= 9.2

    legend = [(FIXED, FIXED_BG, "학습 가중치 없음 (고정 1/S 또는 연산 없음)"),
              (COSINE, COSINE_BG, "고정 규칙 이웃 가중치 (cosine)"),
              (ATTN, ATTN_BG, "학습되는 attention 가중치"),
              (BLOCK, BLOCK_BG, "Transformer 블록 (attention + FFN)")]
    x = 3
    for ec, fc, text in legend:
        box(ax, x, 2.2, 4, 3.2, fc=fc, ec=ec, lw=1.6)
        ax.text(x + 5.5, 3.8, text, va="center", fontsize=10.5, color=INK)
        x += 56
    save(fig, "scene_aggregation_pipeline.png")


# ---------------------------------------------------------------- figure 2: design map
def figure_design_map():
    fig, ax = canvas(170, 112)
    ax.text(3, 107, "그림 2. 비교 설계 지도 — 한 번에 한 가지만 바뀌도록 짝지은 비교",
            fontsize=19, weight="bold", color=INK, va="center")
    ax.text(3, 102, "세로 = ① 장면 간 교환 방식, 가로 = ② 영상 readout 방식. 화살표 한 개는 "
            "구조 차이 한 가지만 가지며, 마지막 화살표만 여러 요소가 함께 바뀝니다.", fontsize=12, color=MUTED, va="center")

    col_x = {"mean": 48, "attn": 115}
    col_w = 52
    row_y = {"none": 78, "cosine": 57, "attn": 36, "block": 12}
    row_h = {"none": 14, "cosine": 14, "attn": 14, "block": 18}
    ax.text(col_x["mean"] + col_w / 2, 95, "② readout: 평균", ha="center", fontsize=13,
            weight="bold", color=INK)
    ax.text(col_x["attn"] + col_w / 2, 95, "② readout: 학습 attention", ha="center",
            fontsize=13, weight="bold", color=INK)
    for key, label in [("none", "① 교환 없음"), ("cosine", "① cosine GCN"),
                       ("attn", "① attention GCN"), ("block", "① Transformer 블록")]:
        ax.text(42, row_y[key] + row_h[key] / 2, label, ha="right", va="center", fontsize=13,
                weight="bold", color=INK)

    def cell(col, row, title, detail, ec, fc, *, empty=False):
        x, y, h = col_x[col], row_y[row], row_h[row]
        box(ax, x, y, col_w, h, fc=fc, ec=ec, lw=1.0 if empty else 2.0,
            alpha=0.55 if empty else 1.0)
        ax.text(x + col_w / 2, y + h - 4, title, ha="center", va="center",
                fontsize=12.5 if not empty else 11, weight="bold",
                color=MUTED if empty else INK)
        ax.text(x + col_w / 2, y + (h - 4) / 2 - 0.6, detail, ha="center", va="center",
                fontsize=10.2, color=MUTED, linespacing=1.4)

    cell("mean", "none", "mean  (A)", "기준: 장면 평균", FIXED, FIXED_BG)
    cell("attn", "none", "attn_pool  (B)", "장면 섞지 않고\n영상 readout 가중치만 학습", ATTN,
         ATTN_BG)
    cell("mean", "cosine", "gnn  (C)", "z 기반 cosine 이웃 가중치\nH′ = H + ReLU(ÂHW + b)",
         COSINE, COSINE_BG)
    cell("attn", "cosine", "gnn + attn_pool", "후속 후보 (미구현)", LINE, "white", empty=True)
    cell("mean", "attn", "gnn_attn  (D)", "gnn에서 Â만 학습 attention으로\n나머지 조건 동일",
         ATTN, ATTN_BG)
    cell("attn", "attn", "—", "구현 없음", LINE, "white", empty=True)
    cell("mean", "block", "transformer · _pos · _local",
         "pre-LN · 2 heads · FFN 2048 · ReZero α\n_local: 자기 장면만 / _pos: 장면 순번 위치",
         BLOCK, BLOCK_BG)
    cell("attn", "block", "—", "구현 없음", LINE, "white", empty=True)

    def label(x, y, text, ha="left"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=10.8, color=INK, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=LINE, lw=0.8))

    # mean -> attn_pool
    arrow(ax, col_x["mean"] + col_w + 0.5, 85, col_x["attn"] - 0.5, 85, color=INK, lw=1.8)
    label((col_x["mean"] + col_w + col_x["attn"]) / 2, 89.4, "장면을 고르는 효과", ha="center")
    # mean -> gnn -> gnn_attn -> transformer, down the mean column
    cx = col_x["mean"] + col_w / 2
    for top, bottom, text in [(78, 71, "장면 간 정보 교환 효과"),
                              (57, 50, "이웃 가중치를 학습하는 효과"),
                              (36, 30, "입력 H·LN·FFN·2 heads·α 등 함께 변경")]:
        arrow(ax, cx, top - 0.4, cx, bottom + 0.4, color=INK, lw=1.8)
        label(cx + 2.5, (top + bottom) / 2, text)
    ax.text(3, 4.5, "A–D는 자문 메모 비교표의 라벨입니다. Transformer 절의 C0/C/D 라벨과는 "
            "다릅니다 (docs/graph_training.md).", fontsize=10.2, color=MUTED, va="center")
    ax.text(3, 1.6, "Transformer 내부 비교: transformer_local → transformer = 장면 간 상호작용, "
            "transformer → transformer_pos = 장면 순서 (파라미터 수 동일).", fontsize=10.2,
            color=MUTED, va="center")
    save(fig, "scene_aggregation_design_map.png")


# ---------------------------------------------------------------- figure 3: mechanisms
def pentagon(cx, cy, r):
    return [(cx + r * math.cos(math.pi / 2 + 2 * math.pi * k / 5),
             cy + r * math.sin(math.pi / 2 + 2 * math.pi * k / 5)) for k in range(5)]


def draw_nodes(ax, pos, color, *, radius=3.0, self_loops=False, center=None):
    for k, (x, y) in enumerate(pos):
        if self_loops:
            dx, dy = x - center[0], y - center[1]
            norm = math.hypot(dx, dy)
            ax.add_patch(Circle((x + dx / norm * (radius + 1.4), y + dy / norm * (radius + 1.4)),
                                1.4, fc="none", ec=color, lw=1.1, zorder=2))
        ax.add_patch(Circle((x, y), radius, fc="white", ec=color, lw=1.8, zorder=4))
        ax.text(x, y, f"s{k}", ha="center", va="center", fontsize=11, color=INK, zorder=5)


def exchange_view(ax, x, y, H, color, *, self_loops=False):
    """Scene graph for slot 1; returns node positions."""
    center = (x + 21, y + H - 31)
    pos = pentagon(*center, 10)
    draw_nodes(ax, pos, color, self_loops=self_loops, center=center)
    return pos, center


def readout_view(ax, x, y, weights, color, labels):
    """Row of scene vectors after slot 1, merged into the video vector v by slot 2."""
    row = [(x + 7 + 7 * k, y + 23) for k in range(5)]
    video = (x + 21, y + 7.5)
    for (sx, sy), weight, label in zip(row, weights, labels):
        ax.plot([sx, video[0]], [sy - 2.6, video[1] + 3.4], color=color, lw=0.6 + 10 * weight,
                alpha=0.8, solid_capstyle="round", zorder=1)
        ax.text(sx, sy + 4.4, label, ha="center", va="center", fontsize=10.5, color=INK)
    for k, (sx, sy) in enumerate(row):
        ax.add_patch(Circle((sx, sy), 2.6, fc="white", ec=MUTED, lw=1.4, zorder=4))
        ax.text(sx, sy, f"s{k}", ha="center", va="center", fontsize=9.5, color=INK, zorder=5)
    ax.add_patch(Circle(video, 3.4, fc=INK, ec=INK, zorder=4))
    ax.text(*video, "v", ha="center", va="center", fontsize=12.5, color="white",
            weight="bold", zorder=5)
    ax.text(x + 2, y + 31.5, "② readout 가중치", fontsize=10.5, color=MUTED, va="center",
            weight="bold")


def panel_frame(ax, x, y, w, h, title, subtitle, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.8, alpha=0.45))
    ax.text(x + 1.8, y + h - 3.6, title, fontsize=16, weight="bold", color=INK, va="center")
    ax.text(x + 1.8, y + h - 8.4, subtitle, fontsize=11.5, color=MUTED, va="center")
    ax.plot([x + 41.5, x + 41.5], [y + 3, y + h - 12], color=LINE, lw=0.8)


def panel_text(ax, x, y, H, formula, bullets):
    top = y + H - 13
    box(ax, x + 43, top - 19, 39, 19, formula, fc="white", ec=LINE, size=11.2, ha="left")
    ax.text(x + 43.4, top - 23, "\n".join("• " + b for b in bullets), fontsize=11.2,
            color=INK, va="top", linespacing=1.6)


def figure_mechanisms(toy, params):
    fig, ax = canvas(264, 196)
    ax.text(3, 191, "그림 3. 다섯 방식이 장면 가중치를 정하는 방법", fontsize=21,
            weight="bold", color=INK, va="center")
    ax.text(3, 185.5, "각 패널 위 = ① 장면 간 교환(이웃 가중치), 아래 = ② 영상 readout(장면 → v). "
            "선 굵기 = 가중치 크기. gnn 굵기는 그림 4의 합성 영상 계산값이고, attention 굵기는 "
            "학습 후 모습을 가정한 개념 예시입니다.", fontsize=11.8, color=MUTED, va="center")

    W, H = 84, 86
    slots = {"mean": (3, 96), "attn_pool": (90, 96), "gnn": (3, 4), "gnn_attn": (90, 4),
             "transformer": (177, 4)}
    uniform, fifth = [0.2] * 5, ["1/5"] * 5

    def no_exchange(x, y):
        exchange_view(ax, x, y, H, FIXED)
        ax.text(x + 21, y + H - 31, "연결 없음", ha="center", va="center", fontsize=11,
                color=MUTED)

    # mean
    x, y = slots["mean"]
    panel_frame(ax, x, y, W, H, "mean  (A · 기준)", "① 교환 없음 → ② 평균", FIXED, FIXED_BG)
    no_exchange(x, y)
    readout_view(ax, x, y, uniform, FIXED, fifth)
    panel_text(ax, x, y, H,
               "v = (1/S) Σ_s h_s\n\n유효 장면만 평균\n장면 0개 영상은 제목 fallback",
               ["가중치: 모든 장면 1/S, 학습 없음",
                "장면끼리 섞지 않음",
                f"추가 파라미터 {params['mean']} · 비용 O(S·d)",
                "내용과 무관하게 모든 장면을\n  같은 비중으로 합침"])

    # attn_pool
    x, y = slots["attn_pool"]
    panel_frame(ax, x, y, W, H, "attn_pool  (B)", "① 교환 없음 → ② 학습 attention 가중합",
                ATTN, ATTN_BG)
    no_exchange(x, y)
    beta = [0.10, 0.12, 0.45, 0.13, 0.20]
    readout_view(ax, x, y, beta, ATTN, [f"{b:.2f}" for b in beta])
    panel_text(ax, x, y, H,
               "e_s = wᵀ tanh(W₁ · unit_rms(h_s) + b₁)\nβ = softmax_s(e)   (영상 안에서)\n"
               "v = Σ_s β_s h_s\nW₁ 512→128,  w 128→1 (bias 없음)",
               ["가중치: 장면마다 점수 1개를 학습",
                "점수는 자기 h_s만 봄 → 장면끼리\n  섞지 않음 (softmax 분모만 공유)",
                f"추가 {params['attn_pool']:,} · 비용 O(S·d)",
                "w = 0 초기화 → 균등 가중치,\n  mean과 정확히 같은 함수로 시작"])

    # summary panel
    x, y = 177, 96
    panel_frame(ax, x, y, W, H, "핵심 차이 한눈에", "어느 자리의 가중치를, 무엇으로 정하나",
                LINE, "white")
    ax.plot([x + 41.5, x + 41.5], [y + 3, y + H - 12], color="white", lw=2)
    lines = [
        ("mean", "② 고정 1/S", FIXED),
        ("attn_pool", "② 장면별 점수를 학습", ATTN),
        ("gnn", "① 특징 z의 cosine 규칙\n    (규칙 자체에 학습 파라미터 없음)", COSINE),
        ("gnn_attn", "① 학습 q·k  (나머지는 gnn과 동일)", ATTN),
        ("transformer", "① 학습 attention 2 heads + FFN\n    (α로 출력 크기 조절)", BLOCK),
    ]
    ly = y + H - 18
    for name, text, color in lines:
        box(ax, x + 2, ly - 2.4, 3.6, 4.8, fc=color, ec=color, lw=1)
        ax.text(x + 8, ly, name, fontsize=12, weight="bold", color=INK, va="center")
        ax.text(x + 31, ly, text, fontsize=11, color=INK, va="center", linespacing=1.4)
        ly -= 11.5
    ax.text(x + 2, y + 6, "공통: 영상 사이 연결 없음 · 장면 1개 영상은 가중치 1\n"
            "장면 순서 무관 (transformer_pos만 순서 사용)", fontsize=10.5, color=MUTED,
            va="center", linespacing=1.5)

    # gnn: undirected edges from the synthetic adjacency
    x, y = slots["gnn"]
    panel_frame(ax, x, y, W, H, "gnn  (C)", "① cosine GCN 1층 → ② 평균", COSINE, COSINE_BG)
    pos, center = exchange_view(ax, x, y, H, COSINE, self_loops=True)
    adjacency = toy["gnn"]
    for i in range(5):
        for j in range(i + 1, 5):
            weight = float(adjacency[i, j])
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=COSINE,
                    lw=0.4 + 14 * weight, alpha=0.3 + 0.6 * min(1, weight / 0.3), zorder=1)
    readout_view(ax, x, y, uniform, FIXED, fifth)
    panel_text(ax, x, y, H,
               "A_ij = max(cos(z_i, z_j), 0),  A_ii = 1\nÂ = D^-½ A D^-½   (대칭 정규화)\n"
               "H′ = H + ReLU(Â H W + b)\nv = (1/S) Σ_s H′_s",
               ["이웃 가중치: 정해진 cosine 규칙,\n  Â 자체에는 학습 파라미터 없음",
                "z는 학습되는 projection 출력 →\n  간접적으로 변하고 gradient도 흐름",
                "대칭: i→j 비중 = j→i 비중",
                f"추가 {params['gnn']:,} (W) · 비용 O(S²·d)"])

    # gnn_attn: directed arrows into s0 only (row 0), illustrative learned weights
    x, y = slots["gnn_attn"]
    panel_frame(ax, x, y, W, H, "gnn_attn  (D)", "① 학습 attention GCN 1층 → ② 평균", ATTN,
                ATTN_BG)
    pos, center = exchange_view(ax, x, y, H, ATTN, self_loops=True)
    row = [0.15, 0.05, 0.10, 0.55, 0.15]
    for j in range(1, 5):
        arrow(ax, pos[j][0], pos[j][1], pos[0][0], pos[0][1], color=ATTN,
              lw=0.5 + 9 * row[j], rad=0.12, scale=10 + 12 * row[j], z=1)
    ax.text(x + 2, y + 39.5, "s0이 받는 가중치(행 0)만 표시", fontsize=10, color=MUTED,
            va="center")
    readout_view(ax, x, y, uniform, FIXED, fifth)
    panel_text(ax, x, y, H,
               "q_i = W_q · unit_rms(z_i)   (384→128)\nk_j = W_k · unit_rms(z_j)\n"
               "A_i· = softmax_j(q_iᵀ k_j / √128)\nH′ = H + ReLU(A H W + b) → 평균",
               ["이웃 가중치: 추천 loss로 학습",
                "비대칭 가능 (W_q ≠ W_k), 행 합 1,\n  self-loop도 softmax 안에서 경쟁",
                "입력 z·W·residual·1층·평균은 gnn과\n  같음 → checkpoint 차이 = q·k 두 행렬",
                f"추가 {params['gnn_attn']:,} · 비용 O(S²·d)"])

    # transformer: two heads into s0
    x, y = slots["transformer"]
    panel_frame(ax, x, y, W, H, "transformer ×3", "① Transformer 블록 1층 → ② 평균", BLOCK,
                BLOCK_BG)
    pos, center = exchange_view(ax, x, y, H, BLOCK)
    heads = ([0.20, 0.10, 0.45, 0.10, 0.15], [0.25, 0.30, 0.05, 0.25, 0.15])
    for head, rad, alpha in ((heads[0], 0.14, 0.9), (heads[1], -0.14, 0.45)):
        for j in range(1, 5):
            ax.add_patch(FancyArrowPatch(pos[j], pos[0], arrowstyle="-|>",
                                         mutation_scale=9 + 10 * head[j], color=BLOCK,
                                         lw=0.5 + 8 * head[j], alpha=alpha,
                                         connectionstyle=f"arc3,rad={rad}", zorder=1))
    ax.text(x + 2, y + 39.5, "진한/연한 선 = head 1/2, s0 행만", fontsize=10, color=MUTED,
            va="center")
    readout_view(ax, x, y, uniform, FIXED, fifth)
    panel_text(ax, x, y, H,
               "X = H + α₁ · MHA(LN₁(H + P))\nH′ = X + α₂ · FFN(LN₂(X + P))\n"
               "v = (1/S) Σ_s H′_s\n2 heads · FFN 2048 · α₁ = α₂ = 0 시작",
               ["입력 H (제목 포함), 기본 P = 0",
                "_pos: 장면 순번 sinusoid P\n  _local: 자기 장면만 attention",
                f"추가 {params['transformer']:,}\n  (세 변형 동일, gnn W의 약 12배)",
                "α = 0 → mean과 정확히 같게 시작"])
    save(fig, "scene_aggregation_mechanisms.png")


# ---------------------------------------------------------------- figure 4: toy weights
def figure_toy_weights(toy):
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", *BLUE_RAMP])
    vmax = 0.5
    fig = plt.figure(figsize=(19, 9.4), dpi=150, facecolor=SURFACE)
    fig.text(0.012, 0.965, "그림 4. 같은 합성 영상에서 실제 모듈로 계산한 가중치 (초기화 직후)",
             fontsize=17, weight="bold", color=INK, va="center")
    fig.text(0.012, 0.925, "합성 특징 z: s0–s2는 같은 주제(cosine 0.82–0.85), s3–s4는 다른 주제"
             "(0.83), 주제 간 0.17–0.23.  추천 학습은 하지 않았으므로 attention의 학습 후 "
             "가중치가 아닙니다.", fontsize=11.5, color=MUTED, va="center")
    columns = [
        ("mean", None, torch.full((5,), 0.2), "고정 1/S"),
        ("gnn", toy["gnn"], torch.full((5,), 0.2), "Â: 대칭, 같은 주제끼리 큼\n행 합 "
         f"{float(toy['gnn'].sum(-1).min()):.2f}–{float(toy['gnn'].sum(-1).max()):.2f}"),
        ("attn_pool", None, toy["beta"], "w = 0 초기화 → mean과 같은 1/S"),
        ("gnn_attn", toy["gnn_attn"], torch.full((5,), 0.2),
         "A: 비대칭, 행 합 1, 주제 구분 없음\n행 엔트로피/log S "
         f"{float(toy['entropy'].min()):.2f}–{float(toy['entropy'].max()):.2f}"),
    ]
    titles = {"mean": FIXED, "gnn": COSINE, "attn_pool": ATTN, "gnn_attn": ATTN}
    grid = fig.add_gridspec(2, 5, left=0.06, right=0.95, top=0.84, bottom=0.08,
                            height_ratios=[5, 1.25], width_ratios=[1, 1, 1, 1, 0.06],
                            hspace=0.55, wspace=0.28)
    labels = [f"s{k}" for k in range(5)]
    image = None
    for c, (name, matrix, readout, note) in enumerate(columns):
        top = fig.add_subplot(grid[0, c])
        top.set_facecolor(SURFACE)
        top.set_title(name, fontsize=14, weight="bold", color=INK, pad=24)
        top.add_patch(plt.Rectangle((0, 1.02), 1, 0.025, transform=top.transAxes,
                                    color=titles[name], clip_on=False))
        if matrix is None:
            # Same square frame as the heatmaps so titles and notes line up.
            top.set_xlim(-0.5, 4.5)
            top.set_ylim(4.5, -0.5)
            top.set_aspect("equal")
            top.set_xticks([])
            top.set_yticks([])
            for spine in top.spines.values():
                spine.set_color(LINE)
            top.text(0.5, 0.5, "① 장면 간 교환 없음\n(장면 벡터를 섞지 않음)", ha="center",
                     va="center", fontsize=12, color=MUTED, transform=top.transAxes)
        else:
            image = top.imshow(matrix.numpy(), cmap=cmap, vmin=0, vmax=vmax)
            for i in range(5):
                for j in range(5):
                    value = float(matrix[i, j])
                    top.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                             color="white" if value > 0.3 else INK)
            top.set_xticks(range(5), labels, fontsize=10)
            top.set_yticks(range(5), labels, fontsize=10)
            top.set_xticks([k + 0.5 for k in range(4)], minor=True)
            top.set_yticks([k + 0.5 for k in range(4)], minor=True)
            top.grid(which="minor", color=SURFACE, lw=2)
            top.tick_params(which="both", length=0, colors=INK)
            for spine in top.spines.values():
                spine.set_visible(False)
            top.set_xlabel("이웃 j (보내는 장면)", fontsize=10, color=MUTED)
            top.set_ylabel("장면 i (받는 장면)", fontsize=10, color=MUTED, labelpad=8)
        top.text(0.5, -0.2, note, ha="center", va="top",
                 fontsize=10.5, color=INK, transform=top.transAxes, linespacing=1.4)

        bottom = fig.add_subplot(grid[1, c])
        values = readout.numpy()[None, :]
        bottom.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for j in range(5):
            bottom.text(j, 0, f"{values[0, j]:.2f}", ha="center", va="center", fontsize=10,
                        color=INK)
        bottom.set_xticks(range(5), labels, fontsize=10)
        bottom.set_xticks([k + 0.5 for k in range(4)], minor=True)
        bottom.grid(which="minor", color=SURFACE, lw=2)
        bottom.set_yticks([])
        bottom.tick_params(which="both", length=0, colors=INK)
        for spine in bottom.spines.values():
            spine.set_visible(False)
        if c == 0:
            bottom.set_ylabel("② readout\nβ_s", fontsize=10.5, color=MUTED, rotation=0,
                              ha="right", va="center")
    fig.text(0.012, 0.62, "① 이웃\n가중치", fontsize=11.5, color=MUTED, weight="bold",
             va="center")
    bar = fig.add_subplot(grid[0, 4])
    colorbar = fig.colorbar(image, cax=bar)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=9.5, colors=INK, length=0)
    colorbar.set_label("가중치", fontsize=10, color=MUTED)
    fig.savefig(OUT / "scene_aggregation_toy_weights.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(OUT / "scene_aggregation_toy_weights.png")


# ---------------------------------------------------------------- figure 5: parameters
def figure_parameters(params):
    order = ["mean", "attn_pool", "gnn", "gnn_attn", "transformer"]
    names = ["mean", "attn_pool", "gnn", "gnn_attn", "transformer (3종 각각)"]
    colors = [FIXED, ATTN, COSINE, ATTN, BLOCK]
    values = [params[k] for k in order]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.barh(range(len(order))[::-1], values, color=colors, height=0.56,
                   edgecolor=SURFACE, linewidth=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 40_000, bar.get_y() + bar.get_height() / 2, f"{value:,}",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(order))[::-1], names, fontsize=11.5, color=INK)
    ax.set_xlim(0, 3_700_000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e8eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_title("그림 5. 공통 인코더에 더해지는 aggregation 파라미터 수", loc="left",
                 fontsize=14, weight="bold", color=INK, pad=12)
    ax.set_xlabel("추가 파라미터 (공통 projection·Scene MLP·SASRec 제외)", fontsize=10.5,
                  color=MUTED)
    fig.tight_layout()
    save(fig, "scene_aggregation_params.png")


if __name__ == "__main__":
    toy = toy_weights()
    params = parameter_counts()
    figure_pipeline(params)
    figure_design_map()
    figure_mechanisms(toy, params)
    figure_toy_weights(toy)
    figure_parameters(params)
