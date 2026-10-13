"""Draw a detailed view of the attention scene GCN (gnn_attn aggregation) against gnn.

Run: python docs/design/generate_gnn_attn_detail_diagram.py
Requires matplotlib and a Noto Sans CJK font; no model is executed.
Mirrors src/validation/scene_attention.py, scene_gcn.py and graph_context.py (sasrec-role-graph/v12).
The example scenes and weights are illustrative: gnn applies its real normalization to made-up
cosines, and gnn_attn applies a real row softmax to made-up logits; neither is a trained result.
"""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

INK, MUTED, LINE = "#1f2a37", "#5b6675", "#9aa5b1"
TEAL, TEAL_BG = "#0f766e", "#e6f4f2"
BLUE, BLUE_BG = "#1d4ed8", "#e8efff"
PURPLE, PURPLE_BG = "#6d28d9", "#f1ebff"
AMBER, AMBER_BG = "#b45309", "#fdf3e1"
GREY_BG = "#f3f4f6"

SCENES = ["기타 연주", "기타 줄 클로즈업", "무대 전경", "관객 박수", "기타 연주(후반)"]

# gnn: illustrative clipped cosines, then the real D^-1/2 A D^-1/2 with unit self-loops.
cosine = np.eye(5)
for (i, j), value in {(0, 1): .8, (0, 2): .4, (0, 3): 0, (0, 4): .9, (1, 2): .2, (1, 3): 0,
                      (1, 4): .7, (2, 3): .5, (2, 4): .4, (3, 4): .1}.items():
    cosine[i, j] = cosine[j, i] = value
degree = cosine.sum(1)
GNN = cosine / np.sqrt(np.outer(degree, degree))
# gnn_attn: illustrative logits q_i·k_j/√128, then the real row softmax.
logits = np.array([[1.0, 0.2, 0.0, 1.2, 0.3],
                   [1.5, 0.5, 0.0, 0.3, 0.8],
                   [0.4, 0.0, 0.6, 0.9, 0.3],
                   [1.1, -0.5, 0.4, 0.5, 0.9],
                   [0.6, 0.3, 0.1, 1.0, 0.8]])
ATTN = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

fig = plt.figure(figsize=(24, 14), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 240)
ax.set_ylim(0, 140)
ax.axis("off")


def box(x, y, w, h, text, *, fc="white", ec=LINE, size=11, color=INK, weight="normal", lw=1.3,
        ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 1.5
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color,
            weight=weight, linespacing=1.4)


def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.4, rad=0.0, ls="-", scale=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def panel(x, y, w, h, title, color, bg):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                                fc=bg, ec=color, lw=1.6, alpha=0.55))
    ax.text(x + 1.5, y + h - 2.6, title, fontsize=17, weight="bold", color=color, va="center")


def mix(color, amount):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def heatmap(x0, top, matrix, color, caption):
    cell = 4.0
    peak = matrix.max()
    for i in range(5):
        ax.text(x0 - 1.2, top - cell * (i + 0.5), f"s{i}", ha="right", va="center", fontsize=9,
                color=MUTED)
        ax.text(x0 + cell * (i + 0.5), top + 1.3, f"s{i}", ha="center", fontsize=9, color=MUTED)
        for j in range(5):
            amount = 0.85 * matrix[i, j] / peak
            ax.add_patch(Rectangle((x0 + cell * j, top - cell * (i + 1)), cell, cell,
                                   fc=mix(color, amount), ec="white", lw=1))
            ax.text(x0 + cell * (j + 0.5), top - cell * (i + 0.5), f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8.3,
                    color="white" if amount > 0.55 else INK)
    ax.add_patch(Rectangle((x0, top - cell), cell * 5, cell, fc="none", ec=INK, lw=1.8))
    ax.text(x0 + cell * 2.5, top - cell * 5 - 1.6, caption, ha="center", va="top", fontsize=9.3,
            color=MUTED, linespacing=1.35)


def focus_graph(cx, cy, row, color, directed):
    """Weights scene s0 reads from every scene (row s0 of the matrix)."""
    r = 8
    pos = [(cx + r * math.cos(math.pi / 2 + 2 * math.pi * k / 5),
            cy + r * math.sin(math.pi / 2 + 2 * math.pi * k / 5)) for k in range(5)]
    for j in range(1, 5):
        (x0, y0), (x1, y1) = pos[0], pos[j]
        if row[j] < 1e-9:
            ax.plot([x0, x1], [y0, y1], color=LINE, lw=1, ls=(0, (2, 2)), zorder=1)
        elif directed:
            length = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x0 - x1) / length, (y0 - y1) / length
            arrow(x1 + ux * 3, y1 + uy * 3, x0 - ux * 3.2, y0 - uy * 3.2, color=color,
                  lw=0.6 + 9 * row[j], scale=10 + 14 * row[j])
        else:
            ax.plot([x0, x1], [y0, y1], color=color, lw=0.6 + 9 * row[j], alpha=0.85, zorder=1)
        # Offset the label perpendicular to the edge, away from the graph centre; inner edges
        # (to s2, s3) place it lower so it clears the s1/s4 nodes.
        t = 0.5 if j in (1, 4) else 0.68
        mx, my = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        nx, ny = -(y1 - y0), x1 - x0
        norm = math.hypot(nx, ny)
        side = 1 if (mx - cx) * nx + (my - cy) * ny >= 0 else -1
        ax.text(mx + side * 2.2 * nx / norm, my + side * 2.2 * ny / norm, f"{row[j]:.2f}",
                fontsize=8.6, color=color if row[j] else MUTED, ha="center", va="center",
                weight="bold", bbox=dict(fc="white", ec="none", pad=0.6))
    for k, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), 2.8, fc="white", ec=color, lw=2.2 if k == 0 else 1.4,
                            zorder=3))
        ax.text(x, y, f"s{k}", ha="center", va="center", fontsize=9, zorder=4,
                weight="bold" if k == 0 else "normal")
    ax.add_patch(Circle((pos[0][0], pos[0][1] + 4.3), 1.5, fc="none", ec=color,
                        lw=0.6 + 9 * row[0], zorder=2))
    ax.text(pos[0][0] + 3.6, pos[0][1] + 4.6, f"self {row[0]:.2f}", fontsize=8.6, color=color,
            va="center", weight="bold")


ax.text(3, 136, "Scene Attention GCN 상세 구조  (--scene-aggregation gnn_attn · sasrec-role-graph/v12)",
        fontsize=23, weight="bold", color=INK, va="center")
ax.text(3, 131.6, "gnn과 입력·residual GCN·평균은 같고, 장면 사이 가중치 A만 cosine 고정 공식 → 학습 attention으로 "
        "바뀝니다   |   d = 128, 장면 벡터 = 512, 가중치 입력 = 384",
        fontsize=13, color=MUTED, va="center")

# ---------------- Panel 1: inputs, identical to gnn ----------------
panel(2, 4, 60, 122, "① 입력 (gnn과 동일)", TEAL, TEAL_BG)
box(5, 95, 54, 22,
    "예시 영상 (가상): 공연 2분 30초 → 30초 장면 5개\n"
    + "\n".join(f"  s{k}   {30 * k}–{30 * k + 30}s   {name}" for k, name in enumerate(SCENES)),
    size=11, ha="left", ec=TEAL)
box(5, 76, 54, 12,
    "장면마다 고정 BGE 1024 → 학습 Linear 128\n"
    "Title t · Entity e · Action a′(역할 결합 후) · Context c",
    size=10.5, ec=TEAL)
arrow(32, 95, 32, 88, color=TEAL)
box(5, 49, 26, 19, "값 경로\nx_s = [t; ē; ā′; c] 512\n→ Scene Residual MLP\n→ H (S × 512)",
    fc=TEAL_BG, ec=TEAL, size=10.5, lw=1.6)
box(33, 49, 26, 19, "가중치 입력\nf_s = [ē; ā′; c] 384\n제목 제외\nScene MLP 이전",
    fc=PURPLE_BG, ec=PURPLE, size=10.5, lw=1.6)
arrow(18, 76, 18, 68, color=TEAL)
arrow(46, 76, 46, 68, color=PURPLE)
ax.text(33.5, 46, "ē = 장면 안 Entity 평균\nā′ = 역할 결합 후 Action 평균", fontsize=9.5,
        color=MUTED, va="top", linespacing=1.35)
box(5, 7, 54, 13,
    "제목은 한 영상의 모든 장면에 같은 값입니다.\n장면을 구별하지 못하므로 가중치 입력에서는 빼고,\n"
    "값 경로(장면 벡터)에는 그대로 둡니다.",
    fc=GREY_BG, size=10)

# ---------------- Panel 2: the only difference, the neighbour weights ----------------
panel(65, 4, 100, 122, "② 영상 1개 내부: 장면 사이 가중치 A만 다름", BLUE, BLUE_BG)
ax.text(68, 117, "gnn — cosine 고정 공식", fontsize=14, weight="bold", color=BLUE)
ax.text(118, 117, "gnn_attn — 학습 attention", fontsize=14, weight="bold", color=PURPLE)
ax.plot([115.5, 115.5], [66, 118], color=LINE, lw=1, ls=(0, (3, 3)))
focus_graph(78, 101, GNN[0], BLUE, directed=False)
focus_graph(128, 101, ATTN[0], PURPLE, directed=True)
for cx in (78, 128):
    ax.text(cx, 88.5, "s0이 각 장면에서 가져오는 비중\n(행렬의 굵은 테두리 행)", ha="center",
            va="top", fontsize=9.3, color=MUTED, linespacing=1.35)
heatmap(93.5, 110, GNN, BLUE, "Â   행 = 받는 장면 i, 열 = 보내는 장면 j")
heatmap(143.5, 110, ATTN, PURPLE, "A   행 = 받는 장면 i, 열 = 보내는 장면 j")
box(68, 66, 45, 15,
    f"• 대칭: Â_01 = Â_10 = {GNN[0, 1]:.2f}\n• 비슷할수록 큼, cos < 0 → 0 (s0–s3)\n"
    "• 정규화 D^-½ A D^-½: 행 합 ≠ 1\n• 학습 파라미터 없음",
    size=10, ha="left", ec=BLUE)
box(118, 66, 45, 15,
    f"• 비대칭: A_01 = {ATTN[0, 1]:.2f},  A_10 = {ATTN[1, 0]:.2f}\n"
    f"• 외형이 달라도 도움이 되면 큼 (s0←s3 {ATTN[0, 3]:.2f})\n"
    "• 행 softmax: 행 합 = 1, 모든 쌍 > 0\n• W_q · W_k 학습 (98,304개)",
    size=10, ha="left", ec=PURPLE)

ax.text(68, 60, "gnn_attn 계산 순서", fontsize=13, weight="bold", color=INK, va="center")
ax.text(97, 60, "보라 = gnn과 다른 부분   흰색 = gnn과 같은 부분", fontsize=10.5, color=MUTED,
        va="center")
steps = [
    (48.5, True, "1) 점수 입력 정규화 (값 경로는 그대로)\n"
                 "û_s = √384 · f_s / max(‖f_s‖, 1e-8)   →  입력 크기에 불변, cosine과 같은 방향 정보"),
    (38.3, True, "2) query · key   W_q, W_k: 384 → 128, bias 없음\n"
                 "q_s = W_q û_s,   k_s = W_k û_s     초기 N(0, 1/768) → 초기 점수 표준편차 0.5"),
    (28.1, True, "3) 장면 사이 가중치 (매 forward 재계산, gradient 흐름)\n"
                 "A = row-softmax( Q Kᵀ / √128 )   자기 장면 포함, 행 합 = 1"),
    (17.9, False, "4) 1층 residual GCN   (W: 512 × 512, bias 있음)\nH′ = H + ReLU( A H W + b )"),
    (7.7, False, "5) 영상 벡터 = 유효 장면 평균 → SASRec\nv = (1/S) Σ_s H′_s   ∈ ℝ^512"),
]
for y, differs, text in steps:
    box(68, y, 94, 8.2, text, fc=PURPLE_BG if differs else "white",
        ec=PURPLE if differs else BLUE, size=10.5, lw=1.6 if differs else 1.3)
for (y1, _, _), (y2, _, _) in zip(steps, steps[1:]):
    arrow(115, y1, 115, y2 + 8.2, color=BLUE)

arrow(59, 56, 68, 53, color=PURPLE, lw=2)
ax.text(62.2, 57.5, "f", fontsize=13, weight="bold", color=PURPLE)
arrow(18, 49, 68, 22, color=INK, lw=2, rad=0.28)
ax.text(36, 30, "H", fontsize=13, weight="bold", color=INK)

# ---------------- Panel 3: where it sits and what to check ----------------
panel(168, 4, 70, 122, "③ 비교 위치와 학습 후 확인", AMBER, AMBER_BG)
ladder = [
    (107, "gnn   cosine 가중치 · 1층 residual GCN\n장면 사이 연산 파라미터 262,656개", "white", BLUE),
    (91, "gnn_attn   학습 attention 가중치\n파라미터 360,960개 (+ W_q·W_k)", PURPLE_BG, PURPLE),
    (75, "transformer   1층 Transformer 블록\n파라미터 3,152,386개", "white", AMBER),
]
for y, text, fc, ec in ladder:
    box(171, y, 64, 9.5, text, fc=fc, ec=ec, size=10.5, lw=1.6)
arrow(203, 107, 203, 100.5, color=MUTED)
ax.text(205, 103.8, "가중치 결정 방식만 학습 함수로", fontsize=9.6, color=MUTED, va="center")
arrow(203, 91, 203, 84.5, color=MUTED)
ax.text(205, 87.8, "+ LN · FFN · 2 heads · ReZero (용량)", fontsize=9.6, color=MUTED,
        va="center")
box(171, 51, 64, 19,
    "결과 해석\n"
    "• gnn_attn > gnn  →  학습한 가중치가 cosine 가정보다 유용\n"
    "• gnn_attn ≈ gnn  →  cosine으로 충분하거나 미학습 (아래 기록으로 구분)\n"
    "• transformer > gnn_attn  →  이득은 가중치 방식이 아닌 추가 용량",
    size=10, ha="left", ec=AMBER)
box(171, 25, 64, 22,
    "training.json → execution.graph_refit.scene_attention_summary\n"
    "장면 2개 이상인 영상만 집계 · 균등 가중치면 모두 1\n"
    "• normalized_entropy_mean: 1 균등 ↔ 0 한 장면 (합성 시작값 ≈ 0.95)\n"
    "• self_weight_ratio_mean: > 1 자기 장면, < 1 다른 장면을 더 참고\n"
    "• max_weight_ratio_mean: 가장 큰 가중치 / 균등값\n"
    "• query · key_weight_norm: 두 행렬의 크기",
    size=10, ha="left", ec=AMBER)
box(171, 6, 64, 15,
    "예외 경로와 주의\n"
    "• 장면 1개 → A = [1] → h + ReLU(hW + b)\n"
    "• 장면 0개 + 제목 → [t; 0; 0; 0] fallback, attention 건너뜀 · meta Arm 미적용\n"
    "• 다른 영상과 연결 없음 · 시간 순서 정보 없음\n"
    "• 높은 가중치 ≠ 사람이 보는 장면 중요도·선호의 원인",
    fc=GREY_BG, size=9.6, ha="left")

ax.text(3, 1.6, "② 상단의 장면·수치는 설명용 가상 예시이며 실제 학습 결과가 아닙니다. "
        "gnn 값은 가상 cosine에 실제 정규화 공식을, gnn_attn 값은 가상 점수에 실제 행 softmax를 적용했습니다.",
        fontsize=10, color=MUTED, va="center")

out = Path(__file__).parent / "gnn_attn" / "gnn_attn_detail_diagram.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
