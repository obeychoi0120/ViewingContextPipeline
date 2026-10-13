"""Draw a detailed view of the scene-role encoder and scene GCN (gnn aggregation).

Run: python docs/design/generate_gnn_detail_diagram.py
Requires matplotlib and a Noto Sans CJK font; no model is executed.
Mirrors src/validation/graph_model.py, scene_gcn.py and graph_context.py (sasrec-role-graph/v8).
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

INK, MUTED, LINE = "#1f2a37", "#5b6675", "#9aa5b1"
TEAL, TEAL_BG = "#0f766e", "#e6f4f2"
BLUE, BLUE_BG = "#1d4ed8", "#e8efff"
AMBER, AMBER_BG = "#b45309", "#fdf3e1"
GREY_BG = "#f3f4f6"

fig = plt.figure(figsize=(24, 14), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 240)
ax.set_ylim(0, 140)
ax.axis("off")


def box(x, y, w, h, text, *, fc="white", ec=LINE, size=11, color=INK, weight="normal", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, weight=weight, linespacing=1.35)


def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def panel(x, y, w, h, title, color, bg):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                                fc=bg, ec=color, lw=1.6, alpha=0.55))
    ax.text(x + 1.5, y + h - 2.6, title, fontsize=17, weight="bold", color=color, va="center")


ax.text(3, 136, "Scene GCN 모델 상세 구조  (--scene-aggregation gnn · sasrec-role-graph/v8)",
        fontsize=23, weight="bold", color=INK, va="center")
ax.text(3, 131.6, "BGE large-en-v1.5 고정(1024) → 학습 가능한 projection·역할 결합·Scene MLP·GCN·SASRec을 "
        "추천 loss로 함께 학습   |   d = 128, 장면 벡터 = 4d = 512",
        fontsize=13, color=MUTED, va="center")

# ---------------- Panel 1: inside one scene ----------------
panel(2, 4, 84, 122, "① 장면 1개 내부: 역할 그래프 → 장면 벡터", TEAL, TEAL_BG)

ax.text(4, 117, "예시 Scene Graph (v8)", fontsize=12, color=MUTED, weight="bold")
ax.text(4, 113.5, "[Entities] p1 person · f1 tomato · t1 knife · l1 kitchen\n"
        "[Actions]  p1 - cutting - f1; t1; l1", fontsize=10.5, color=MUTED, va="top",
        family="DejaVu Sans Mono")

ents = {"p1\nperson": (12, 92), "f1\ntomato": (12, 80), "t1\nknife": (12, 68), "l1\nkitchen": (12, 56)}
roles = ["actor", "target", "tool", "location"]
act = (44, 74)
for (label, (x, y)), role in zip(ents.items(), roles):
    ax.add_patch(Circle((x, y), 4.3, fc="white", ec=TEAL, lw=1.6))
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=INK)
    arrow(x + 4.3, y, act[0] - 6.5, act[1] + (y - 74) * 0.15, color=TEAL, lw=1.3)
    ax.text((x + act[0]) / 2 + 1, (y + act[1]) / 2 + 1.2, role, fontsize=9.5, color=TEAL,
            ha="center", style="italic")
box(act[0] - 6.5, act[1] - 4, 13, 8, "Action\ncutting", fc="white", ec=TEAL, size=10.5, lw=1.6)
ax.text(22, 49, "Entity 노드: BGE(\"Name: … Attributes: …\")\nAction 노드: BGE(동작 텍스트)\n"
        "ID는 연결에만 쓰고 BGE 입력에는 넣지 않음", fontsize=9.5, color=MUTED, va="top")

bx = 60
box(bx, 104, 24, 7, "Entity e_i = W_e·BGE\n1024 → 128", size=10)
box(bx, 94, 24, 7, "Action a = W_a·BGE\n1024 → 128", size=10)
box(bx, 84, 24, 7, "Context c = W_c·BGE\nformat·medium·topics", size=10)
box(bx, 74, 24, 7, "Title t = W_t·BGE\n(제목 없는 Arm은 0)", size=10)

box(54, 56, 30, 13,
    "역할 결합 (Entity → Action)\n"
    "a′ = a + ReLU(W_r [a; r_actor; r_target;\n        r_tool; r_location])   640 → 128\n"
    "연결 없음: none / unknown 학습 상태 벡터",
    fc="white", ec=TEAL, size=9.5)
arrow(56, 74, 60, 69, color=TEAL)

box(6, 30, 78, 12,
    "장면 readout   x_s = [ t ; mean(e_i) ; mean(a′_j) ; c ]  = 128 × 4 = 512\n"
    "유사도 특징   z_s = [ mean(e_i) ; mean(a′_j) ; c ]  = 384   (제목 제외, Scene MLP 이전)",
    fc="white", ec=TEAL, size=11, lw=1.6)
arrow(69, 56, 60, 42, color=TEAL)

box(6, 10, 78, 13,
    "Scene Residual MLP   h_s = x_s + W_2·ReLU(W_1 x_s)    512 → 512 → 512, LayerNorm 없음\n"
    "→  H ∈ ℝ^{S×512}  (S = 이 영상의 유효 장면 수)",
    fc=TEAL_BG, ec=TEAL, size=11, lw=1.6)
arrow(45, 30, 45, 23, color=TEAL)

# ---------------- Panel 2: inside one video ----------------
panel(89, 4, 78, 122, "② 영상 1개 내부: 장면 유사도 그래프 + GCN", BLUE, BLUE_BG)

cx, cy, r = 128, 97, 15
n = 5
pos = [(cx + r * math.cos(math.pi / 2 + 2 * math.pi * k / n),
        cy + r * math.sin(math.pi / 2 + 2 * math.pi * k / n)) for k in range(n)]
weights = {(0, 1): 0.9, (0, 2): 0.4, (1, 2): 0.7, (2, 3): 0.8, (3, 4): 0.9, (1, 4): 0.3,
           (0, 4): 0.6, (1, 3): 0.15, (0, 3): 0.0, (2, 4): 0.5}
for (i, j), wgt in weights.items():
    if wgt > 0:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=BLUE,
                lw=0.6 + 4 * wgt, alpha=0.25 + 0.6 * wgt, zorder=1)
for k, (x, y) in enumerate(pos):
    ax.add_patch(Circle((x, y), 4.2, fc="white", ec=BLUE, lw=1.8, zorder=3))
    ax.text(x, y, f"s{k}\n{30 * k}–{30 * k + 30}s", ha="center", va="center", fontsize=8.8,
            zorder=4)
    loop_dx, loop_dy = (x - cx) / r * 5.6, (y - cy) / r * 5.6
    ax.add_patch(Circle((x + loop_dx, y + loop_dy), 1.6, fc="none", ec=BLUE, lw=1.0, zorder=2))
ax.text(cx, cy - r - 7.5, "모든 장면 쌍 연결(완전 그래프) · 선 굵기 = 유사도\n"
        "작은 원 = self-loop(가중치 1) · 다른 영상과는 연결 없음 · 시간 순서 정보 없음",
        ha="center", fontsize=10, color=MUTED, va="top")

steps = [
    (57, "1) 인접 행렬 (학습 중 매 forward 재계산, gradient 흐름)\n"
         "A_ij = max( cos(z_i, z_j), 0 ),   A_ii = 1"),
    (46.5, "2) 대칭 정규화\nÂ = D^{-1/2} A D^{-1/2},   D_ii = Σ_j A_ij"),
    (36, "3) 1층 residual GCN   (W: 512 × 512, bias 있음)\nH′ = H + ReLU( Â H W + b )"),
    (25.5, "4) 영상 벡터 = 유효 장면 평균\nv = (1/S) Σ_s H′_s   ∈ ℝ^512"),
]
for y, text in steps:
    box(93, y, 70, 8.5, text, fc="white", ec=BLUE, size=11)
for (y1, _), (y2, _) in zip(steps, steps[1:]):
    arrow(128, y1, 128, y2 + 8.5, color=BLUE)

box(93, 6, 70, 14,
    "예외 경로\n"
    "• 장면 0개 + 제목 있음 → [t; 0; 0; 0] → Scene MLP (GCN 건너뜀)\n"
    "• 장면 1개 → Â = [1] → h + ReLU(hW + b)\n"
    "• 제목·장면 모두 없음 / padding → 0 벡터   • meta Arm → GCN 없음",
    fc=GREY_BG, ec=LINE, size=10)

# ---------------- Panel 3: user side ----------------
panel(170, 4, 68, 122, "③ 사용자 표현과 추천 점수", AMBER, AMBER_BG)

box(174, 104, 60, 10, "최근 시청 영상 최대 10개\n[v_1, …, v_10]  각 512  (②의 출력)", fc="white",
    ec=AMBER, size=11)
box(174, 89, 60, 9, "+ 위치 임베딩 (시청 순서)", fc="white", ec=AMBER, size=11)
box(174, 74, 60, 9, "SASRec Transformer\n2 blocks · 2 heads · 512 · causal", fc="white", ec=AMBER,
    size=11)
box(174, 59, 60, 9, "출력 LayerNorm → 마지막 유효 위치\n→ User Residual MLP (LayerNorm·GELU) = u", fc="white",
    ec=AMBER, size=11)
box(174, 42, 60, 10, "점수 = u · v_cand\n후보 = 전체 catalog (본 영상 마스킹)", fc=AMBER_BG,
    ec=AMBER, size=12, weight="bold")
for y1, y2 in [(104, 98), (89, 83), (74, 68), (59, 52)]:
    arrow(204, y1, 204, y2, color=AMBER)
ax.text(204, 37, "후보 영상 벡터 v_cand도 같은 ②로 계산\n(sequence encoder를 거치지 않음)",
        ha="center", va="top", fontsize=10, color=MUTED)
box(174, 6, 60, 26,
    "학습\n"
    "• loss: in-batch softmax + 인기도 logQ 보정\n  (모든 위치 next-item, 본 영상 마스킹)\n"
    "• BGE 외 모든 모듈 end-to-end 학습\n"
    "• AdamW lr 1e-4 · wd 0.1 · batch 128\n"
    "• selection(D−1 이전) → val NDCG@10으로 epoch 선택\n"
    "  → refit(D 이전) → test [D, D+1)\n"
    "• 7일 × seed 3개 × Arm 독립 학습",
    fc=GREY_BG, ec=LINE, size=10)

arrow(84, 16, 93, 40, color=INK, lw=2, rad=-0.2)
ax.text(86.5, 30, "H", fontsize=13, weight="bold", color=INK)
arrow(163, 29.75, 174, 108, color=INK, lw=2, rad=-0.25)
ax.text(167.5, 60, "v", fontsize=13, weight="bold", color=INK)

out = Path(__file__).with_name("gnn_detail_diagram.png")
fig.savefig(out, dpi=150)
print(out)
