"""Draw the figures of the gnn node/feature walkthrough (gnn_inputs_report.md).

Run: PYTHONPATH=src python docs/design/generate_gnn_inputs_report.py [--run-root DIR]
Requires matplotlib, torch, a Noto Sans CJK font and a Run whose graph inputs were prepared
by `embed-representations` (default: /mnt/microlens/artifacts/runs/261006_v9). No model is
trained; figure 4 runs the seed-42 initialized gnn encoder on one real video.
Mirrors src/validation/graph_inputs.py, graph_batching.py, graph_model.py and scene_gcn.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]

OUT = Path(__file__).parent
INK, MUTED, LINE, SURFACE = "#1f2a37", "#5b6675", "#c9ced4", "#fcfcfb"
GREY, GREY_BG = "#8a8f98", "#f1f2f4"
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
ORANGE, ORANGE_BG = "#eb6834", "#fdeee7"
AQUA, AQUA_BG = "#1baf7a", "#e5f6ef"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ROLES = ("actor", "target", "tool", "receiver", "location")
EXAMPLE = ("microlens_100k_00800", 1)


def canvas(width, height, scale=10):
    fig = plt.figure(figsize=(width / scale, height / scale), dpi=150, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", *, fc="white", ec=LINE, size=11, color=INK, weight="normal",
        lw=1.3, ha="center", family=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw))
    if text:
        tx = x + w / 2 if ha == "center" else x + 1.2
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color,
                weight=weight, linespacing=1.4, family=family)


def arrow(ax, x1, y1, x2, y2, *, color=MUTED, lw=1.4, rad=0.0, scale=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=scale,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=2))


def band(ax, x, y, w, h, title, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6, alpha=0.5))
    ax.text(x + 1.8, y + h - 3.2, title, fontsize=14, weight="bold", color=INK, va="center")


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(OUT / name)


def read_scenes(run_root, arm, content_id):
    path = run_root / "extraction" / "scenes" / arm / f"{content_id}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# ---------------------------------------------------------------- figure 1: pipeline
def figure_pipeline():
    fig, ax = canvas(244, 84)
    ax.text(3, 80, "그림 1. 영상에서 GNN 영상 벡터까지 — 고정 전처리와 학습 구간",
            fontsize=19, weight="bold", color=INK, va="center")
    ax.text(3, 75.2, "위 구간(A)은 Run마다 한 번 계산해 파일과 shared cache에 저장하고, 아래 구간(B)만 "
            "추천 loss로 학습합니다. BGE 가중치는 학습하지 않습니다.", fontsize=12.5, color=MUTED,
            va="center")

    band(ax, 2, 41, 240, 30, "A. 추출·전처리  (1회 계산 후 캐시, 학습 없음)", GREY, GREY_BG)
    top = [
        "영상\nMicroLens-100k\n(오디오 미사용)",
        "장면 = 30초 고정 구간\n키프레임 6장\n(5초 칸 중앙, 640×352)",
        "VLM 장면 추출\nQwen3-VL-2B / Gemini\nscene_graph v8",
        "파서\n텍스트 3개 섹션 →\n장면별 scene_graph JSON",
        "장면 필터\nraw(파싱 실패)·warning·\n구조 오류 제외",
        "Role graph 직렬화\nnode·edge·결측 상태\n+ node별 텍스트",
        "BGE large en v1.5\nCLS → L2 정규화, 1024\n고유 텍스트만 인코딩",
    ]
    w, gap = 31.0, 2.6
    xs = [4 + k * (w + gap) for k in range(len(top))]
    y, h = 44.5, 18
    for x, text in zip(xs, top):
        box(ax, x, y, w, h, text, size=11.6)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.3, y + h / 2, b - 0.3, y + h / 2, color=INK)

    band(ax, 2, 3, 240, 33, "B. 추천 학습  (BGE 이후 모든 모듈을 추천 loss로 함께 학습)", BLUE,
         BLUE_BG)
    # Snake layout: B runs right to left under the BGE box.
    bottom = [
        ("Projection ×4\nEntity·Action·Context·Title\n1024 → 128 (각각)", False),
        ("역할 결합\nEntity → Action\n640 → 128", False),
        ("장면 readout\nx_s = [t ; ē ; ā′ ; c]\n512", False),
        ("Scene MLP (1층)\nh_s = x_s + ReLU(W x_s)\n512", False),
        ("Scene GCN 1층\nÂ = 정규화 cosine(z)\nH′ = H + ReLU(ÂHW + b)", True),
        ("영상 readout\nv = 유효 장면 평균\n512", False),
        ("SASRec\n시청 이력 v → u\n점수 = u · v", False),
    ]
    y, h = 6.5, 18
    for x, (text, gcn) in zip(reversed(xs), bottom):
        box(ax, x, y, w, h, text, size=11.6, fc=BLUE_BG if gcn else "white",
            ec=BLUE if gcn else LINE, lw=2 if gcn else 1.3)
    for a, b in zip(list(reversed(xs)), list(reversed(xs))[1:]):
        arrow(ax, a - 0.3, y + h / 2, b + w + 0.3, y + h / 2, color=INK)
    arrow(ax, xs[6] + w / 2, 44.2, xs[6] + w / 2, y + h + 0.3, color=INK, lw=1.8)
    ax.text(xs[6] - 1.5, 38.5, "features.npy (고유 텍스트 × 1024)\n+ node·edge·offset 정수 배열",
            ha="right", va="center", fontsize=10.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LINE, lw=0.8))
    save(fig, "gnn_inputs_pipeline.png")


# ---------------------------------------------------------------- figure 2: one real scene
def figure_scene_example(run_root):
    content_id, scene_idx = EXAMPLE
    row = next(r for r in read_scenes(run_root, "graph_qwen", content_id)
               if r["scene_idx"] == scene_idx)
    graph = row["scene_graph"]
    fig, ax = canvas(262, 122)
    ax.text(3, 118, f"그림 2. 실제 장면 하나가 node와 텍스트로 바뀌는 과정  ({content_id}, "
            f"scene {scene_idx}, Qwen)", fontsize=18, weight="bold", color=INK, va="center")
    ax.text(3, 113.2, "장면 내부에서 Entity와 Action이 node가 되고, 역할(actor·target·tool·location)이 "
            "Entity → Action edge가 됩니다. ID(p1, f1 …)는 연결에만 쓰고 BGE 텍스트에는 넣지 "
            "않습니다.", fontsize=11.5, color=MUTED, va="center")

    # (1) extraction output, written in the v8 text format the VLM emits
    band(ax, 2, 4, 70, 104, "① VLM 출력 (v8 형식으로 표기)", GREY, GREY_BG)
    slot = lambda v: "none" if v is None else v
    lines = ["[Context]", f"medium: {graph['medium']}", f"format: {graph['format']}",
             "topics: " + "; ".join(graph["topics"]), "[Entities]"]
    lines += [f"{e['id']}: " + "; ".join([e["name"], *e["attributes"]]) for e in graph["entities"]]
    lines += ["[Actions]"]
    lines += [f"{a['actor']} - {a['action']} - {slot(a['target'])}; {slot(a['tool'])}; "
              f"{slot(a['location'])}" for a in graph["actions"]]
    height = 3.15 * len(lines) + 5
    box(ax, 4.5, 98 - height, 65, height, "\n".join(lines), size=10.6, ha="left",
        family="DejaVu Sans Mono")
    ax.text(5, 98 - height - 5, "• 실제 저장 형식은 파싱된 JSON입니다.\n"
            "• none → null,  unknown → \"unknown\"\n"
            "• 파싱에 실패한 장면은 원문 문자열로 저장되고\n   (raw) 학습 입력에서 빠집니다.\n"
            "• 이 영상은 11개 장면 중 scene 5가 raw입니다.",
            fontsize=11, color=INK, va="top", linespacing=1.6)

    # (2) role graph
    band(ax, 75, 4, 82, 104, "② 장면 내부 role graph", BLUE, BLUE_BG)
    entities = {e["id"]: e for e in graph["entities"]}
    ent_pos = {"p1": (88, 90), "f1": (88, 70), "p2": (88, 50), "t1": (88, 32), "l1": (88, 15)}
    act_pos = {0: (128, 82), 1: (128, 40)}
    colors = {"actor": BLUE, "target": ORANGE, "tool": AQUA, "location": GREY}
    for k, action in enumerate(graph["actions"]):
        ax_, ay = act_pos[k]
        for role in ("actor", "target", "tool", "location"):
            value = action[role]
            if value in entities:
                ex, ey = ent_pos[value]
                arrow(ax, ex + 4.6, ey, ax_ - 9, ay + (ey - ay) * 0.12, color=colors[role],
                      lw=2.0)
                mx, my = ex + 4.6 + (ax_ - 9 - ex - 4.6) * 0.45, ey + (ay - ey) * 0.45
                ax.text(mx, my + 1.6, role, fontsize=9.8, color=INK, ha="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for eid, (x, y) in ent_pos.items():
        ax.add_patch(Circle((x, y), 4.6, fc="white", ec=BLUE, lw=1.8, zorder=4))
        ax.text(x, y + 0.9, eid, ha="center", va="center", fontsize=10.5, weight="bold",
                zorder=5)
        ax.text(x, y - 1.9, entities[eid]["name"], ha="center", va="center", fontsize=8.6,
                zorder=5)
    for k, action in enumerate(graph["actions"]):
        x, y = act_pos[k]
        box(ax, x - 9, y - 4, 18, 8, action["action"], ec=INK, size=11, lw=1.8)
        states = [(role, action[role]) for role in ("actor", "target", "tool", "location")
                  if action[role] not in entities]
        for j, (role, value) in enumerate(states):
            label = f"{role}: {slot(value)}"
            box(ax, x - 9, y - 11.5 - 6 * j, 18, 4.6, label, fc=GREY_BG, ec=GREY, size=9.4)
        if states:
            ax.text(x + 10.5, y - 9.5, "← 연결이 없으면\n   학습 상태 벡터", fontsize=9.2,
                    color=MUTED, va="center")
    legend = [("actor", BLUE), ("target", ORANGE), ("tool", AQUA), ("location", GREY)]
    for j, (name, color) in enumerate(legend):
        ax.plot([99 + 14 * j, 104 + 14 * j], [8, 8], color=color, lw=2.4)
        ax.text(105.5 + 14 * j, 8, name, fontsize=9.6, color=INK, va="center")
    ax.text(116, 100.5, "Entity node 5개 · Action node 2개 · edge 6개",
            fontsize=10.2, color=INK, ha="center")

    # (3) text that BGE receives
    band(ax, 160, 4, 100, 104, "③ BGE 입력 텍스트 (node마다 1개)", ORANGE, ORANGE_BG)
    rows = [("Entity", f"Name: {e['name']}\nAttributes: {', '.join(sorted(e['attributes']))}",
             e["id"]) for e in sorted(graph["entities"], key=lambda e: e["id"])]
    rows += [("Action", a["action"], a["action"]) for a in graph["actions"]]
    rows += [("Context", f"Format: {graph['format']}\nMedium: {graph['medium']}\n"
              f"Topics: {', '.join(sorted(graph['topics']))}", "장면당 1개")]
    titles = [json.loads(line) for line in
              (run_root / "validation" / "cohort" / "metadata_titles.jsonl").open()]
    title = next(t["title"] for t in titles if t["content_id"] == content_id)
    rows += [("Title", title, "영상당 1개\n(제목 사용 Arm)")]
    y = 99
    for kind, text, tag in rows:
        height = 4.3 + 3.2 * text.count("\n")
        box(ax, 163, y - height, 15, height, kind, size=9.8, weight="bold",
            fc={"Entity": BLUE_BG, "Action": "white", "Context": AQUA_BG,
                "Title": GREY_BG}[kind])
        box(ax, 180.5, y - height, 60, height, text, size=9.3, ha="left",
            family="DejaVu Sans Mono")
        ax.text(242.5, y - height / 2, tag, fontsize=9.3, color=MUTED, va="center")
        y -= height + 1.6
    ax.text(163, 7.5, "같은 문자열은 한 번만 인코딩해 같은 벡터를 공유합니다. 속성은 정렬,\n"
            "Entity는 ID 순서로 정렬합니다(ID는 텍스트에 넣지 않음).", fontsize=9.6,
            color=INK, va="center", linespacing=1.45)
    save(fig, "gnn_inputs_scene_example.png")


# ---------------------------------------------------------------- figure 3: scene features
def lane(ax, points, color=INK, label=None, at=None):
    """Right-angled connector whose last segment ends in an arrow head."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.4, solid_joinstyle="round", zorder=2)
    ax.plot(xs[-2:-1] + xs[-1:], ys[-2:-1] + ys[-1:], color=color, lw=1.4, zorder=2)
    arrow(ax, *points[-2], *points[-1], color=color)
    if label:
        ax.text(*at, label, fontsize=10.5, color=INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LINE, lw=0.8), zorder=3)


def figure_scene_features():
    fig, ax = canvas(246, 98)
    ax.text(3, 94, "그림 3. 학습 단계에서 장면 벡터 h_s와 유사도 특징 z_s를 만드는 계산",
            fontsize=19, weight="bold", color=INK, va="center")
    ax.text(3, 89.2, "그림 2의 장면 기준입니다. d = 128. 모든 W는 추천 loss로 학습하며 BGE 벡터"
            "(1024)는 고정입니다.", fontsize=12.5, color=MUTED, va="center")

    band(ax, 2, 2, 30, 82, "고정 BGE", GREY, GREY_BG)
    band(ax, 34, 2, 34, 82, "Projection", BLUE, BLUE_BG)
    rows = [("Entity ×5", "e_i = W_e · BGE\n1024 → 128", 60),
            ("Action ×2", "a_j = W_a · BGE\n1024 → 128", 42),
            ("Context ×1", "c = W_c · BGE\n1024 → 128", 26),
            ("Title ×1\n(영상 공통)", "t = W_t · BGE  1024 → 128\n제목 없는 Arm은 0", 9)]
    for left, right, y in rows:
        box(ax, 4.5, y, 25, 11, left, size=12)
        box(ax, 36.5, y, 29, 11, right, size=11.2)
        arrow(ax, 29.8, y + 5.5, 36.2, y + 5.5, color=INK)

    band(ax, 70, 2, 86, 82, "역할 결합  (Action마다 1회)", ORANGE, ORANGE_BG)
    box(ax, 72.5, 55, 81, 20,
        "a′ = a + ReLU( W_r [ a ; r_actor ; r_target ; r_tool ; r_location ] )\n"
        "W_r : 640 → 128,  receiver 슬롯은 사용하지 않음\n"
        "r_role = 연결된 Entity의 e_i (Projection 직후 값)\n"
        "           연결이 없으면 상태 벡터 s_role[none | unknown]",
        size=11.2, ha="left", ec=ORANGE, lw=1.8)
    box(ax, 72.5, 35, 81, 15,
        "preparing:  [ a ; e_p2 ; e_f1 ; e_t1 ; e_l1 ]\n"
        "picking:      [ a ; e_p1 ; e_f1 ; s_tool[none] ; s_location[unknown] ]\n"
        "f1(pepper)은 두 Action 모두에 같은 e_f1로 들어감", size=11, ha="left")
    box(ax, 72.5, 13, 81, 17,
        "상태 벡터 s : 역할 4개 × 상태 2개 × 128,  N(0, 0.02²) 초기화\n"
        "Entity 벡터 e_i는 바뀌지 않음 (Entity → Action 한 방향)\n"
        "Action이 없는 장면은 이 단계를 건너뜀", size=11, ha="left", fc=GREY_BG)
    arrow(ax, 65.8, 64, 72.2, 64, color=INK)
    arrow(ax, 65.8, 47.5, 72.2, 59, color=INK)

    band(ax, 158, 2, 86, 82, "장면 readout → Scene MLP", AQUA, AQUA_BG)
    box(ax, 160.5, 61, 81, 13,
        "ē = mean_i(e_i)    Entity 5개 전체 평균 (Action에 안 쓰인 것 포함)\n"
        "ā′ = mean_j(a′_j)   Action 2개 평균 (Action 없으면 0 벡터)", size=11, ha="left")
    box(ax, 160.5, 43, 81, 12,
        "x_s = [ t ; ē ; ā′ ; c ]    128 × 4 = 512\n"
        "h_s = x_s + ReLU( W_s x_s + b_s )    512 → 512, LayerNorm 없음",
        size=11.4, ha="left", ec=AQUA, lw=1.8)
    box(ax, 160.5, 23, 81, 14,
        "z_s = [ ē ; ā′ ; c ]    384    → GCN 이웃 가중치 전용\n"
        "제목 t 제외 (영상 안 모든 장면에 같은 값)\n"
        "Scene MLP 이전 값, gradient는 차단하지 않음",
        size=11, ha="left", ec=BLUE, lw=1.8)
    box(ax, 160.5, 5, 81, 12.5,
        "영상의 장면 S개 → H (S × 512) = GCN의 값,  Z (S × 384) = 이웃 가중치 입력\n"
        "장면 0개 + 제목 있음 → [ t ; 0 ; 0 ; 0 ] → Scene MLP (GCN 생략)",
        size=10.8, ha="left", fc=GREY_BG)
    arrow(ax, 153.8, 65, 160.2, 65, color=INK)
    ax.text(157, 66.6, "a′", fontsize=10.5, ha="center", color=INK)
    # e_i reaches the entity mean directly, over the top of the role fusion column.
    lane(ax, [(65.8, 69), (69, 69), (69, 77.6), (157, 77.6), (157, 71), (160.2, 71)],
         label="e_i 그대로 → ē", at=(128, 77.6))
    # Context and title skip role fusion and enter x_s directly, under the column.
    ax.plot([65.8, 69, 69], [31.5, 31.5, 7.5], color=INK, lw=1.4, zorder=2)
    ax.plot([65.8, 69], [14.5, 14.5], color=INK, lw=1.4, zorder=2)
    lane(ax, [(69, 7.5), (157, 7.5), (157, 48), (160.2, 48)],
         label="c, t 그대로 → x_s", at=(113, 7.5))
    arrow(ax, 201, 60.7, 201, 55.3, color=INK)
    arrow(ax, 201, 42.7, 201, 37.3, color=BLUE)
    arrow(ax, 201, 22.7, 201, 17.8, color=INK)
    save(fig, "gnn_inputs_scene_features.png")


# ---------------------------------------------------------------- figure 4: real adjacency
def video_adjacency(run_root):
    """Seed-42 initialized gnn encoder on one prepared video; returns Â and scene labels."""
    import torch
    from validation.graph_inputs import GraphStore
    from validation.graph_model import GraphSASRec
    from validation.model import seed_everything
    from validation.scene_gcn import normalized_similarity

    content_id = EXAMPLE[0]
    directory = run_root / "validation" / "representations" / "graph" / "graph_qwen_meta_embeddings"
    catalog = json.loads((directory / "manifest.json").read_text())["catalog"]
    index = [item["content_id"] for item in catalog].index(content_id)
    seed_everything(42, deterministic=False)
    store = GraphStore(directory)
    model = GraphSASRec(store=store, aggregation="gnn", item_count=len(store), max_length=10,
                        embedding_dim=512, num_blocks=2, num_heads=2, dropout=0.1, arm="graph")
    model.eval()
    captured = {}
    hook = model.graph_encoder.gcn.register_forward_hook(
        lambda _, args, __: captured.update(features=args[1]))
    with torch.no_grad():
        model.graph_encoder(model.packed.batch(np.array([index]), "cpu", group_scenes=True))
    hook.remove()
    adjacency = normalized_similarity(captured["features"]).numpy()
    unit = torch.nn.functional.normalize(captured["features"], dim=-1)
    cosine = (unit @ unit.T).numpy()
    rows = [r for r in read_scenes(run_root, "graph_qwen", content_id)
            if isinstance(r["scene_graph"], dict) and not r["warning"]]
    labels = [f"s{r['scene_idx']}  " + ", ".join(r["scene_graph"]["topics"][:2]) for r in rows]
    return adjacency, cosine, labels


def figure_video_adjacency(run_root):
    adjacency, cosine, labels = video_adjacency(run_root)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", *BLUE_RAMP])
    size = len(labels)
    fig = plt.figure(figsize=(17, 9.6), dpi=150, facecolor=SURFACE)
    fig.text(0.012, 0.965, f"그림 4. 실제 영상 하나의 장면 유사도 그래프  ({EXAMPLE[0]}, Qwen, "
             f"유효 장면 {size}개)", fontsize=17, weight="bold", color=INK, va="center")
    fig.text(0.012, 0.928, "실제 BGE 특징에 seed 42로 초기화한 gnn 모델의 Projection을 적용해 계산"
             "했습니다(학습 전). 학습이 진행되면 Projection이 바뀌어 값도 바뀝니다. "
             "scene 5는 파싱 실패(raw)로 제외되었습니다.", fontsize=11.2, color=MUTED, va="center")
    grid = fig.add_gridspec(1, 4, left=0.2, right=0.96, top=0.84, bottom=0.2,
                            width_ratios=[1, 0.035, 1, 0.035], wspace=0.14)
    for c, (matrix, title, vmax) in enumerate([
            (cosine, "cos(z_i, z_j)   (0으로 자르기 전)", 1.0),
            (adjacency, "Â = D^-½ max(cos, 0) D^-½   (GCN 가중치)", 0.25)]):
        ax = fig.add_subplot(grid[0, 2 * c])
        image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12.5, color=INK, pad=10, weight="bold")
        for i in range(size):
            for j in range(size):
                value = float(matrix[i, j])
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.4,
                        color="white" if value > 0.6 * vmax else INK)
        short = [label.split()[0] for label in labels]
        ax.set_xticks(range(size), short, fontsize=9.5)
        ax.set_yticks(range(size), labels if c == 0 else [""] * size, fontsize=9.5)
        ax.set_xticks([k + 0.5 for k in range(size - 1)], minor=True)
        ax.set_yticks([k + 0.5 for k in range(size - 1)], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="both", length=0, colors=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2 * c + 1]))
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=9, colors=INK, length=0)
    row_sum = adjacency.sum(1)
    fig.text(0.2, 0.1, f"cosine 범위 {cosine[~np.eye(size, dtype=bool)].min():.2f}–"
             f"{cosine[~np.eye(size, dtype=bool)].max():.2f} (대각선 제외) · Â 행 합 "
             f"{row_sum.min():.2f}–{row_sum.max():.2f} · Â는 대칭이며 self-loop 가중치 1에서 "
             "출발해 degree로 나뉩니다.", fontsize=11, color=INK)
    fig.text(0.2, 0.065, "학습 전 Projection은 무작위이므로, 이 값은 초기 학습 시점에 BGE 특징끼리의 "
             "유사도가 어떻게 전달되는지 보여 줄 뿐입니다.", fontsize=11, color=MUTED)
    save(fig, "gnn_inputs_video_adjacency.png")
    return adjacency, cosine, labels


# ---------------------------------------------------------------- figure 5: data statistics
def role_states(run_root, arm):
    directory = run_root / "validation" / "representations" / "graph" / f"{arm}_embeddings"
    missing = np.load(directory / "missing.npy", mmap_mode="r")
    result = {}
    for slot, role in enumerate(ROLES):
        if role == "receiver":
            continue
        column = np.asarray(missing[:, slot + 1])
        result[role] = [(column == -1).mean(), (column == 0).mean(), (column == 1).mean()]
    return result


def figure_data_stats(run_root):
    data = {"Qwen": role_states(run_root, "graph_qwen_meta"),
            "Gemini": role_states(run_root, "graph_gemini_meta")}
    states = [("Entity에 연결", BLUE), ("none", GREY), ("unknown", ORANGE)]
    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels, y = [], 0
    positions = []
    for role in ("actor", "target", "tool", "location"):
        for name in ("Qwen", "Gemini"):
            positions.append((y, role, name))
            labels.append(f"{role} · {name}")
            y += 1
        y += 0.6
    for (y, role, name) in positions:
        left = 0.0
        for (state, color), value in zip(states, data[name][role]):
            if value <= 0:
                continue
            ax.barh(y, value, left=left, height=0.72, color=color, edgecolor=SURFACE,
                    linewidth=2)
            if value >= 0.06:
                ax.text(left + value / 2, y, f"{value:.0%}", ha="center", va="center",
                        fontsize=10, color="white" if color != GREY else INK)
            left += value
    ax.set_yticks([p[0] for p in positions], labels, fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in states]
    ax.legend(handles, [s for s, _ in states], loc="lower center", bbox_to_anchor=(0.45, -0.2),
              ncol=3, frameon=False, fontsize=10.5)
    ax.set_title("그림 5. Action 역할별 연결 상태 (Run 261006_v9, 유효 장면의 모든 Action)",
                 loc="left", fontsize=14, weight="bold", color=INK, pad=12)
    fig.tight_layout()
    save(fig, "gnn_inputs_role_states.png")
    return data


# ---------------------------------------------------------------- statistics (JSON)
def bundle_statistics(run_root, arm):
    directory = run_root / "validation" / "representations" / "graph" / f"{arm}_embeddings"
    load = lambda name: np.asarray(np.load(directory / f"{name}.npy", mmap_mode="r"))
    counts = json.loads((directory / "statistics.json").read_text())["counts"]
    scenes = np.diff(load("video_offsets"))
    offsets, types = load("scene_offsets"), load("types")
    entities = np.add.reduceat((types == 0).astype(int), offsets[:-1])
    actions = np.add.reduceat((types == 1).astype(int), offsets[:-1])
    references = len(types) + len(load("contexts")) + int((load("titles") != 0).sum())
    return {
        "counts": counts, "videos": len(scenes),
        "features_rows": int(np.load(directory / "features.npy", mmap_mode="r").shape[0]),
        "text_references": references,
        "scenes_per_video": {"median": float(np.median(scenes)), "mean": float(scenes.mean()),
                             "max": int(scenes.max()), "single": int((scenes == 1).sum())},
        "entities_per_scene": {"mean": float(entities.mean()),
                               "median": float(np.median(entities)), "max": int(entities.max())},
        "actions_per_scene": {"mean": float(actions.mean()),
                              "median": float(np.median(actions)), "max": int(actions.max())},
        "entity_nodes": int((types == 0).sum()), "action_nodes": int((types == 1).sum()),
        "edges": int(load("edges").shape[0]),
        "role_states": {role: dict(zip(("linked", "none", "unknown"), map(float, values)))
                        for role, values in role_states(run_root, arm).items()},
    }


def adjacency_statistics(run_root, arm, *, seed=42, pairs=200_000):
    """Seed-initialized gnn similarity over the whole catalog (videos with >= 2 scenes)."""
    import torch
    from validation.graph_inputs import GraphStore
    from validation.graph_model import GraphSASRec
    from validation.model import seed_everything
    from validation.scene_gcn import normalized_similarity

    directory = run_root / "validation" / "representations" / "graph" / f"{arm}_embeddings"
    seed_everything(seed, deterministic=False)
    store = GraphStore(directory)
    model = GraphSASRec(store=store, aggregation="gnn", item_count=len(store), max_length=10,
                        embedding_dim=512, num_blocks=2, num_heads=2, dropout=0.1, arm="graph")
    model.eval()
    within, entropy, ratio, vectors, owners = [], [], [], [], []
    state = {"video": 0}

    def collect(_, args, __):
        features, groups = args[1], args[2]
        unit = torch.nn.functional.normalize(features, dim=-1)
        for indices in groups:
            size = indices.shape[1]
            if size < 2:
                continue
            cosine = unit[indices] @ unit[indices].transpose(-1, -2)
            mask = ~torch.eye(size, dtype=torch.bool)
            within.append(cosine[:, mask].mean(-1))
            adjacency = normalized_similarity(features[indices])
            rows = adjacency / adjacency.sum(-1, keepdim=True)
            entropy.append((-(rows * rows.log()).sum(-1) / np.log(size)).flatten())
            ratio.append((rows.amax(-1) * size).flatten())
        vectors.append(unit)

    hook = model.graph_encoder.gcn.register_forward_hook(collect)
    with torch.no_grad():
        for start in range(0, len(store), 256):
            items = np.arange(start, min(start + 256, len(store)))
            batch = model.packed.batch(items, "cpu", group_scenes=True)
            owners.append(batch["scene_videos"].numpy() + start)
            model.graph_encoder(batch)
    hook.remove()
    within, entropy, ratio = (torch.cat(v).numpy() for v in (within, entropy, ratio))
    vectors, owners = torch.cat(vectors).numpy(), np.concatenate(owners)
    rng = np.random.default_rng(seed)
    left, right = rng.integers(0, len(owners), (2, pairs))
    keep = owners[left] != owners[right]
    across = (vectors[left[keep]] * vectors[right[keep]]).sum(-1)
    quantile = lambda v: {q: float(np.quantile(v, float(q))) for q in ("0.1", "0.5", "0.9")}
    return {
        "videos_with_2plus_scenes": int(len(within)),
        "within_video_offdiag_cosine_mean": float(within.mean()),
        "within_video_offdiag_cosine_quantiles": quantile(within),
        "across_video_pair_cosine_mean": float(across.mean()),
        "across_video_pairs": int(keep.sum()),
        "row_normalized_entropy_mean": float(entropy.mean()),
        "row_normalized_entropy_quantiles": quantile(entropy),
        "row_max_weight_ratio_mean": float(ratio.mean()),
        "note": "row entropy uses each Â row renormalized to sum 1; ratio = S x max weight",
    }


def bge_baseline(run_root, arm, *, pairs=100_000, seed=0):
    """Mean BGE cosine of random pairs of distinct texts of one kind (anisotropy baseline)."""
    directory = run_root / "validation" / "representations" / "graph" / f"{arm}_embeddings"
    features = np.load(directory / "features.npy", mmap_mode="r")
    nodes, types = np.load(directory / "nodes.npy"), np.load(directory / "types.npy")
    rng = np.random.default_rng(seed)
    result = {}
    for kind, ids in (("entity", nodes[types == 0]), ("action", nodes[types == 1]),
                      ("context", np.load(directory / "contexts.npy"))):
        left, right = rng.choice(np.unique(ids), (2, pairs))
        keep = left != right
        values = np.sum(np.asarray(features[left[keep]]) * np.asarray(features[right[keep]]), 1)
        result[kind] = float(values.mean())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=Path("/mnt/microlens/artifacts/runs/261006_v9"))
    run_root = parser.parse_args().run_root
    figure_pipeline()
    figure_scene_example(run_root)
    figure_scene_features()
    figure_video_adjacency(run_root)
    figure_data_stats(run_root)
    statistics = {"run": run_root.name}
    for arm in ("graph_qwen_meta", "graph_gemini_meta"):
        statistics[arm] = {"inputs": bundle_statistics(run_root, arm),
                           "initial_adjacency_seed42": adjacency_statistics(run_root, arm),
                           "bge_random_pair_cosine": bge_baseline(run_root, arm)}
    path = OUT / "gnn_inputs_statistics.json"
    path.write_text(json.dumps(statistics, indent=2, ensure_ascii=False) + "\n")
    print(path)
