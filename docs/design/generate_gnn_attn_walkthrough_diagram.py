"""Draw a step-by-step walkthrough of the gnn_attn attention computation with a small example.

Run: python docs/design/generate_gnn_attn_walkthrough_diagram.py
Requires matplotlib and a Noto Sans CJK font; no model is executed.
Mirrors SceneAttentionGCN in src/validation/scene_attention.py and SceneGCN.forward in scene_gcn.py.
Every number is computed here with the implemented formulas, but on a made-up example reduced to
3 scenes, a 4-d attention input, 2-d query/key and 3-d values; W_q, W_k, H and W are illustrative.
Only the parameter counts in panel C use the real sizes.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

INK, MUTED, LINE = "#1f2a37", "#5b6675", "#9aa5b1"
BLUE, BLUE_BG = "#1d4ed8", "#e8efff"
PURPLE, PURPLE_BG = "#6d28d9", "#f1ebff"
AMBER, AMBER_BG = "#b45309", "#fdf3e1"
GREY_BG = "#f3f4f6"

# ---------------- Example, computed with the implemented formulas ----------------
SCENES = ("s0", "s1", "s2")
F = np.array([[0.6, 0.4, 0.1, 0.0],    # s0 기타 연주
              [1.2, 0.9, 0.0, 0.3],    # s1 기타 줄 클로즈업 (s0과 비슷한 방향, 약 2배 크기)
              [0.0, 0.1, 0.7, 0.5]])   # s2 관객 박수
D, KEY = F.shape[1], 2
U = np.sqrt(D) * F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
W_Q = np.array([[0.5, 0.3, 0.0, 0.1], [0.1, 0.0, 0.4, 0.2]])
W_K = np.array([[0.4, 0.2, 0.1, 0.0], [0.3, 0.1, 0.5, 0.4]])
Q, K = U @ W_Q.T, U @ W_K.T
E = Q @ K.T / np.sqrt(KEY)
A = np.exp(E) / np.exp(E).sum(1, keepdims=True)
H = np.array([[1.0, 0.2, 0.0], [0.9, 0.3, 0.1], [0.1, 0.0, 1.0]])
W, BIAS = 0.5 * np.eye(3), np.zeros(3)
M = A @ H
H2 = H + np.maximum(M @ W.T + BIAS, 0)
V = H2.mean(0)
unit = U / np.sqrt(D)
cosine = np.clip(unit @ unit.T, 0, None)
np.fill_diagonal(cosine, 1)
degree = cosine.sum(1)
GNN = cosine / np.sqrt(np.outer(degree, degree))

fig = plt.figure(figsize=(24, 15), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 240)
ax.set_ylim(0, 150)
ax.axis("off")


def box(x, y, w, h, text="", *, fc="white", ec=LINE, size=11, color=INK, lw=1.3, ha="center",
        family=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2 if ha == "center" else x + 1.5, y + h / 2, text, ha=ha, va="center",
                fontsize=size, color=color, linespacing=1.4, family=family)


def arrow(x1, y1, x2, y2, *, color=MUTED, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def panel(x, y, w, h, title, color, bg):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
                                fc=bg, ec=color, lw=1.6, alpha=0.55))
    ax.text(x + 1.5, y + h - 2.6, title, fontsize=16, weight="bold", color=color, va="center")


def mix(color, amount):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def table(x, top, matrix, cols, *, rows=SCENES, cw=6.3, ch=3.6, color=None, highlight=None,
          title=None, size=9.4):
    """Small numeric table; `top` is the top of the column-header row."""
    matrix = np.atleast_2d(matrix)
    if title:
        ax.text(x + cw * len(cols) / 2, top + 1.6, title, ha="center", fontsize=10.5,
                weight="bold", color=INK)
    for j, name in enumerate(cols):
        ax.text(x + cw * (j + 0.5), top - ch * 0.45, name, ha="center", va="center", fontsize=9,
                color=MUTED)
    peak = np.abs(matrix).max() or 1
    for i, row in enumerate(matrix):
        y = top - ch * (i + 2)
        if rows:
            ax.text(x - 0.8, y + ch / 2, rows[i], ha="right", va="center", fontsize=9.2,
                    color=MUTED)
        for j, value in enumerate(row):
            amount = 0.8 * value / peak if color else 0
            ax.add_patch(Rectangle((x + cw * j, y), cw, ch, fc=mix(color or "white", amount),
                                   ec=LINE, lw=0.6))
            ax.text(x + cw * (j + 0.5), y + ch / 2, f"{value:.2f}", ha="center", va="center",
                    fontsize=size, color="white" if amount > 0.55 else INK)
    if highlight is not None:
        ax.add_patch(Rectangle((x, top - ch * (highlight + 2)), cw * len(cols), ch, fc="none",
                               ec=INK, lw=1.8))
    return top - ch * (len(matrix) + 1)


def card(x, top, w, h, step, formula, shape, note, *, ec, fc="white"):
    box(x, top - h, w, h, fc=fc, ec=ec, lw=1.6)
    ax.text(x + 1.8, top - 3, step, fontsize=12.5, weight="bold", color=ec, va="center")
    ax.text(x + 1.8, top - 7.4, formula, fontsize=11, color=INK, va="center")
    ax.text(x + 1.8, top - 11, shape, fontsize=9.6, color=MUTED, va="center")
    ax.text(x + 1.8, top - h + 1.2, note, fontsize=9.3, color=MUTED, va="bottom",
            linespacing=1.38)


ax.text(3, 146, "gnn_attn attention 계산 상세  (영상 1개 · 장면 S개 · sasrec-role-graph/v12)",
        fontsize=23, weight="bold", color=INK, va="center")
ax.text(3, 141.4, "실제 차원: f 384 → q·k 128, H 512   |   표의 숫자는 장면 3개 · f 4차원 · key 2차원 · "
        "H 3차원으로 줄인 가상 예시이며, 계산은 구현과 같은 공식으로 했습니다", fontsize=13, color=MUTED,
        va="center")
ax.text(3, 137.4, "예시 장면:  s0 기타 연주   ·   s1 기타 줄 클로즈업 (s0과 방향은 비슷, 크기 약 2배)   ·   "
        "s2 관객 박수 (외형이 다름)", fontsize=12, color=INK, va="center")

# ---------------- Panel A: computing the weights (differs from gnn) ----------------
panel(2, 83, 236, 51, "A. 가중치 계산 — gnn과 다른 부분   (매 forward마다 다시 계산, gradient가 흐름)",
      PURPLE, PURPLE_BG)
TOP, HEIGHT, WIDTH = 128, 43, 44.4
xs = [4.5 + k * (WIDTH + 2.5) for k in range(5)]
norms = np.linalg.norm(F, axis=1, keepdims=True)

card(xs[0], TOP, WIDTH, HEIGHT, "① 가중치 입력 f", "f_s = [ē; ā′; c]   (제목 제외)",
     "실제: S × 384  (Scene MLP 이전, gnn과 같은 입력)",
     "s1은 s0과 방향이 비슷하지만 크기가 약 2배입니다.\n이대로 내적하면 크기가 점수를 좌우합니다.",
     ec=PURPLE)
table(xs[0] + 5, TOP - 16.5, np.hstack([F, norms]), ["f₁", "f₂", "f₃", "f₄", "‖f‖"])

card(xs[1], TOP, WIDTH, HEIGHT, "② 점수 입력 정규화", "û_s = √D · f_s / max(‖f_s‖, 1e-8)",
     "실제: S × 384,  D = 384  (예시 D = 4)",
     "크기는 지우고 방향만 남깁니다 (gnn cosine과 같은 정보).\n√D를 곱해 원소 제곱 평균을 1로 맞춥니다\n"
     "→ 점수가 0 근처에 갇히지 않습니다.",
     ec=PURPLE)
table(xs[1] + 5, TOP - 16.5, np.hstack([U, np.linalg.norm(U, axis=1, keepdims=True)]),
      ["û₁", "û₂", "û₃", "û₄", "‖û‖"])

card(xs[2], TOP, WIDTH, HEIGHT, "③ query · key", "q_s = W_q û_s,    k_s = W_k û_s",
     "실제: W_q, W_k  384 → 128, bias 없음  (예시 4 → 2)",
     "q: '나는 어떤 장면을 찾는가', k: '나는 어떤 장면인가'.\n두 행렬을 따로 학습하므로 방향성이 생깁니다.\n"
     "초기 N(0, 1/(2·384)) · 예시 값은 학습 후를 가정",
     ec=PURPLE)
table(xs[2] + 5, TOP - 16.5, Q, ["q₁", "q₂"], title="Q")
table(xs[2] + 23, TOP - 16.5, K, ["k₁", "k₂"], rows=None, title="K")

card(xs[3], TOP, WIDTH, HEIGHT, "④ 점수", "e_ij = q_i · k_j / √d_k",
     "실제: S × S,  d_k = 128  (예시 d_k = 2)",
     "행 i = 받는 장면, 열 j = 보내는 장면 · 자기 자신 포함.\n"
     f"q_i·k_j ≠ q_j·k_i → 비대칭 (e_02 = {E[0, 2]:.2f}, e_20 = {E[2, 0]:.2f}).\n"
     "√d_k로 나눠 key 폭과 무관하게 크기를 유지합니다.",
     ec=PURPLE)
table(xs[3] + 7, TOP - 16.5, E, ["s0", "s1", "s2"], cw=7.5, highlight=0)

exp_row = np.exp(E[0])
card(xs[4], TOP, WIDTH, HEIGHT, "⑤ 행 softmax → A", "A_ij = exp(e_ij) / Σ_k exp(e_ik)",
     "실제: S × S,  각 행의 합 = 1",
     "s0 행 풀어보기:  exp(e_0·) = "
     + ", ".join(f"{v:.2f}" for v in exp_row) + f"\n합 {exp_row.sum():.2f}로 나누면  A_0· = "
     + ", ".join(f"{v:.2f}" for v in A[0]) + "\n모든 값 > 0, 다른 영상과의 칸은 없음",
     ec=PURPLE, fc=PURPLE_BG)
table(xs[4] + 5, TOP - 16.5, np.hstack([A, A.sum(1, keepdims=True)]), ["s0", "s1", "s2", "Σ"],
      cw=7.5, color=PURPLE, highlight=0)

for k in range(4):
    arrow(xs[k] + WIDTH, TOP - HEIGHT / 2, xs[k + 1], TOP - HEIGHT / 2, color=PURPLE)

# ---------------- Panel B: using the weights (same as gnn) ----------------
panel(2, 31, 148, 46, "B. 가중치 사용 — gnn과 같은 부분", BLUE, BLUE_BG)
TOP_B, HEIGHT_B, WIDTH_B = 70, 37, 46
xb = [4.5 + k * (WIDTH_B + 2.5) for k in range(3)]
card(xb[0], TOP_B, WIDTH_B, HEIGHT_B, "⑥ 메시지 M = A H", "m_i = Σ_j A_ij h_j",
     "실제: (S × S)(S × 512) → S × 512",
     f"s0: {A[0, 0]:.2f}·h_0 + {A[0, 1]:.2f}·h_1 + {A[0, 2]:.2f}·h_2\n"
     "H = Scene MLP 출력 (제목 포함, 정규화 없음)",
     ec=BLUE)
table(xb[0] + 5, TOP_B - 16.5, H, ["h₁", "h₂", "h₃"], cw=5.6, title="H (값)")
table(xb[0] + 26, TOP_B - 16.5, M, ["m₁", "m₂", "m₃"], rows=None, cw=5.6, color=BLUE, title="M")

card(xb[1], TOP_B, WIDTH_B, HEIGHT_B, "⑦ residual 갱신", "H′ = H + ReLU(M W + b)",
     "실제: W 512 × 512, bias 있음  (예시 W = 0.5·I, b = 0)",
     "원래 장면 정보 H는 그대로 두고,\n다른 장면에서 가져온 정보를 더합니다.",
     ec=BLUE)
table(xb[1] + 8, TOP_B - 16.5, H2, ["h′₁", "h′₂", "h′₃"], cw=7, color=BLUE)

card(xb[2], TOP_B, WIDTH_B, HEIGHT_B, "⑧ 영상 벡터", "v = (1/S) Σ_s H′_s",
     "실제: 512 → SASRec의 아이템 벡터",
     "장면 순서를 바꿔도 v는 같습니다.\n장면 1개 영상은 A = [1]입니다.",
     ec=BLUE)
table(xb[2] + 8, TOP_B - 16.5, V, ["v₁", "v₂", "v₃"], rows=("v",), cw=7, color=BLUE)

for k in range(2):
    arrow(xb[k] + WIDTH_B, TOP_B - HEIGHT_B / 2, xb[k + 1], TOP_B - HEIGHT_B / 2, color=BLUE)
# A from step 5 feeds step 6 (routed through the gap between the two rows).
land = xb[0] + WIDTH_B - 6  # Right of the panel B title.
ax.plot([xs[4] + WIDTH / 2, xs[4] + WIDTH / 2, land], [TOP - HEIGHT, 80, 80], color=PURPLE, lw=2)
arrow(land, 80, land, TOP_B, color=PURPLE, lw=2)
ax.text(120, 81.2, "A (S × S)", fontsize=11, weight="bold", color=PURPLE, ha="center")

# ---------------- Panel C: the same example under gnn, and the parameter cost ----------------
panel(153, 31, 85, 46, "C. gnn과 비교", AMBER, AMBER_BG)
ax.text(156, 70.2, "s0이 가져오는 비중 (예시 값)", fontsize=10.5, weight="bold", color=INK,
        va="center")
base, scale = 53, 24
for j, name in enumerate(SCENES):
    cx = 166 + j * 15
    for offset, value, color in ((-2.6, GNN[0, j], BLUE), (2.6, A[0, j], PURPLE)):
        ax.add_patch(Rectangle((cx + offset - 2.2, base), 4.4, value * scale, fc=color,
                               ec="none", alpha=0.85))
        ax.text(cx + offset, base + value * scale + 0.9, f"{value:.2f}", ha="center",
                fontsize=9.3, weight="bold", color=color)
    ax.text(cx, base - 2, f"s0 ← {name}", ha="center", fontsize=9.8, color=INK)
ax.plot([158, 204], [base, base], color=LINE, lw=1)
for k, (label, color) in enumerate((("gnn (cosine)", BLUE), ("gnn_attn (학습 attention)", PURPLE))):
    ax.add_patch(Rectangle((208, 65.4 - 3.4 * k), 2.4, 2.2, fc=color, ec="none"))
    ax.text(211.4, 66.5 - 3.4 * k, label, fontsize=9.4, color=INK, va="center")
ax.text(208, 58.4, f"관객 장면 s2를 보면\ncos(s0, s2) = {cosine[0, 2]:.2f}\n"
        f"→ gnn {GNN[0, 2]:.2f}, gnn_attn {A[0, 2]:.2f}", fontsize=9.3, color=INK, va="top",
        linespacing=1.4)

# Parameter counts of the scene-exchange module at the real sizes (512 values, 384 → 128 q/k).
GCN_W, QUERY_KEY = 512 * 512 + 512, 2 * 384 * 128
GNN_MODEL_TOTAL = 7_970_432  # Whole GraphSASRec with gnn (BGE 1024, 512, 2 blocks), counted.
ax.plot([156, 236], [47.5, 47.5], color=LINE, lw=0.8, ls=(0, (3, 3)))
ax.text(156, 44.6, "장면 사이 모듈 파라미터 (실제 설정)", fontsize=10.5, weight="bold", color=INK,
        va="center")
ax.text(236, 44.6, f"모델 전체 {GNN_MODEL_TOTAL / 1e6:.2f}M → "
        f"{(GNN_MODEL_TOTAL + QUERY_KEY) / 1e6:.2f}M (+{QUERY_KEY / GNN_MODEL_TOTAL:.1%})",
        fontsize=9.3, color=MUTED, va="center", ha="right")
unit = 38 / (GCN_W + QUERY_KEY)
for y, name, segments in ((38.6, "gnn", [(GCN_W, BLUE, "W 512×512")]),
                          (34.2, "gnn_attn", [(GCN_W, BLUE, "W 512×512"),
                                              (QUERY_KEY, PURPLE, "W_q·W_k")])):
    ax.text(170.5, y + 1.4, name, ha="right", va="center", fontsize=9.5, color=INK)
    x = 172
    for value, color, label in segments:
        ax.add_patch(Rectangle((x, y), value * unit, 2.8, fc=color, ec="white", lw=0.8,
                               alpha=0.85))
        ax.text(x + value * unit / 2, y + 1.4, label, ha="center", va="center", fontsize=8.3,
                color="white")
        x += value * unit
    total = sum(value for value, _, _ in segments)
    extra = f"  (+{QUERY_KEY:,} · +{QUERY_KEY / GCN_W:.0%})" if len(segments) > 1 else ""
    ax.text(x + 1.2, y + 1.4, f"{total:,}{extra}", va="center", fontsize=9.5, color=INK,
            weight="bold" if extra else "normal")

# ---------------- Bottom: why each choice, and the code ----------------
notes = [
    ("왜 ②에서 정규화하나",
     "장면 특징 노름 ≈ 1이면 점수 ≈ 0 → A가 균등에서 못 벗어남\n"
     "합성 300 step 엔트로피: 정규화 없음 0.99999 → 있음 0.886\n"
     "값 경로 H와 영상 벡터 크기는 바꾸지 않음"),
    ("③의 초기화와 시작 상태",
     "W_q, W_k ~ N(0, 1/(2·384))\n"
     "→ 초기 점수 표준편차 0.5 (scene Transformer와 같음)\n"
     "→ 초기 A는 균등에 가까움 (합성 엔트로피 ≈ 0.95)"),
    ("학습 중에는",
     "loss → v → H′ → M → A → e → q, k → W_q, W_k 로 gradient\n"
     "A는 저장하지 않고 매 forward 다시 계산\n"
     "높은 A_ij ≠ 사람이 보는 장면 중요도·선호의 원인"),
]
for k, (title, text) in enumerate(notes):
    x = 4 + k * 59
    box(x, 4.5, 56, 22, fc=GREY_BG)
    ax.text(x + 1.5, 23.6, title, fontsize=12, weight="bold", color=INK, va="center")
    ax.text(x + 1.5, 20.4, text, fontsize=11, color=INK, va="top", linespacing=1.75)
box(181, 4.5, 56, 22, fc="#111827", ec="#111827")
ax.text(182.5, 23.6, "실제 코드  (scene_attention.py · scene_gcn.py)", fontsize=11.5,
        weight="bold", color="white", va="center")
ax.text(182.5, 20.6,
        "u = unit_rms(f[idx])               # (B, S, 384)\n"
        "e = query(u) @ key(u).mT / √128    # (B, S, S)\n"
        "A = e.softmax(dim=-1)\n"
        "H2 = H + relu(projection(A @ H))   # (B, S, 512)",
        fontsize=10.2, color="#e5e7eb", va="top", family="DejaVu Sans Mono", linespacing=1.75)
ax.text(182.5, 7.2, "B = 장면 수가 같은 영상 묶음 (영상끼리 섞이지 않음)", fontsize=9.6,
        color="#cbd5e1", va="center")

out = Path(__file__).parent / "gnn_attn" / "gnn_attn_walkthrough_diagram.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
