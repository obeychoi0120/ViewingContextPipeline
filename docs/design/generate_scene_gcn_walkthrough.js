// Scene GCN 구현 설명 슬라이드 생성기.
// 실행: NODE_PATH=<pptxgenjs가 설치된 node_modules> APPLY_THEME=<pptx 스킬의 scripts/apply_theme.js> \
//       node docs/design/generate_scene_gcn_walkthrough.js
// 출력: docs/design/scene_gcn_walkthrough.pptx (OUT 환경변수로 변경 가능). PDF는 LibreOffice로 별도 변환.
// 내용은 src/validation/{graph_inputs,graph_batching,graph_model,scene_gcn,graph_context,model}.py 를 따릅니다.
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const THEME = {
  name: "SceneGCN",
  headFontFace: "Malgun Gothic",
  bodyFontFace: "Malgun Gothic",
  colors: {
    dk1: "1F2A37", lt1: "FFFFFF", dk2: "0F766E", lt2: "E6F4F2",
    accent1: "0F766E", accent2: "1D4ED8", accent3: "B45309",
    accent4: "5B6675", accent5: "C7CDD4", accent6: "F3F4F6",
    hlink: "1D4ED8", folHlink: "6D2E46",
  },
};
const H = THEME.colors;
const BLUE_BG = "E8EFFF", AMBER_BG = "FDF3E1", GREY_BG = "F3F4F6";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Scene GCN 구현 분석";
const C = pres.SchemeColor;
const S = pres.ShapeType;

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.4, fontSize: 40, bold: true, color: C.background1, margin: 0, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.6, fontSize: 18, color: C.background2, margin: 0, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 36, bold: true, color: C.background1, margin: 0, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 1.2, fontSize: 18, color: C.accent5, margin: 0, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.75, fontSize: 28, bold: true, color: C.text2, margin: 0, valign: "middle" }, text: "" } },
    { text: { text: "Scene GCN 구현 분석 · ViewingContextPipeline", options: { x: 0.5, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: H.accent4, align: "right" },
});

// ---------- helpers ----------
function box(slide, x, y, w, h, text, o = {}) {
  const opts = {
    x, y, w, h, shape: S.roundRect, rectRadius: 0.08,
    fill: { color: o.fill || H.lt1 }, line: { color: o.line || H.accent5, width: o.lw || 1 },
    fontSize: o.size || 13, color: o.color || H.dk1, bold: !!o.bold, align: o.align || "center",
    valign: o.valign || "middle", margin: o.margin === undefined ? 6 : o.margin,
    fontFace: o.mono ? "Courier New" : undefined, paraSpaceAfter: o.psa || 0,
  };
  if (o.name) opts.objectName = o.name;
  slide.addText(text, opts);
}
function note(slide, x, y, w, h, text, o = {}) {
  slide.addText(text, {
    x, y, w, h, fontSize: o.size || 12, color: o.color || H.accent4, align: o.align || "left",
    valign: o.valign || "top", margin: 0, isTextBox: true, bold: !!o.bold, italic: !!o.italic,
    fontFace: o.mono ? "Courier New" : undefined, paraSpaceAfter: o.psa || 0,
    objectName: o.name || "note",
  });
}
function arrow(slide, x1, y1, x2, y2, o = {}) {
  const dx = x2 - x1, dy = y2 - y1;
  slide.addShape(S.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(dx), h: Math.abs(dy),
    flipH: dx < 0, flipV: dy < 0,
    line: { color: o.color || H.accent4, width: o.lw || 1.5, endArrowType: o.noHead ? undefined : "triangle", dashType: o.dash || "solid" },
  });
}
function circle(slide, cx, cy, r, text, o = {}) {
  slide.addText(text, {
    x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, shape: S.ellipse,
    fill: { color: o.fill || H.lt1 }, line: { color: o.line || H.accent1, width: o.lw || 1.5 },
    fontSize: o.size || 11, color: o.color || H.dk1, align: "center", valign: "middle", margin: 0,
    objectName: o.name || "node",
  });
}
function card(slide, x, y, w, h, title, body, o = {}) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: o.fill || GREY_BG }, line: { color: o.line || H.accent5, width: 1 } });
  note(slide, x + 0.2, y + 0.15, w - 0.4, 0.4, title, { size: o.titleSize || 15, bold: true, color: o.titleColor || H.dk2 });
  note(slide, x + 0.2, y + 0.6, w - 0.4, h - 0.75, body, { size: o.size || 12.5, color: H.dk1, psa: 4 });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function bullets(items) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } }));
}
const tblBorder = { type: "solid", color: H.accent5, pt: 0.75 };
function headerRow(cells, fill = H.dk2) {
  return cells.map((t) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: fill }, align: "center" } }));
}

// ======================================================================
pres.addSection({ title: "개요" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "개요" });
  s.addText("Scene GCN은 어떻게 구현되어 있나", { placeholder: "title" });
  s.addText("ViewingContextPipeline · --representation-mode graph --scene-aggregation gnn\n모델 계약 sasrec-role-graph/v8 · 코드 기준 2026-10-06", { placeholder: "body" });
  note(s, 0.8, 6.6, 11.7, 0.4, "src/validation/graph_inputs.py · graph_batching.py · graph_model.py · scene_gcn.py · graph_context.py · model.py", { size: 11, color: H.lt2, mono: true });
}

// ---------- 2. 전체 흐름 ----------
{
  const s = content("전체 흐름: Scene Graph 파일에서 추천 점수까지 5단계", "개요");
  const stages = [
    ["① 입력 변환", "Scene JSONL →\n노드·간선·문맥 배열", "graph_inputs.py\nembed-representations", H.lt2, H.accent1],
    ["② 고정 특징", "노드 텍스트를 BGE로\n인코딩 (1024, 동결)", "graph_inputs.py\n→ <arm>_embeddings/", H.lt2, H.accent1],
    ["③ 장면 encoder", "projection → 역할 결합\n→ 장면 MLP", "graph_model.py\nSceneRoleEncoder", BLUE_BG, H.accent2],
    ["④ 장면 GCN", "영상 안 장면 유사도 그래프\n→ 1층 GCN → 평균", "scene_gcn.py\nSceneGCN", BLUE_BG, H.accent2],
    ["⑤ 추천", "영상 벡터 → SASRec\n→ 사용자·후보 내적", "graph_model.py\nGraphSASRec · model.py", AMBER_BG, H.accent3],
  ];
  stages.forEach(([t, d, f, bg, ln], i) => {
    const x = 0.5 + i * 2.56;
    box(s, x, 1.5, 2.3, 0.55, t, { fill: ln, color: H.lt1, bold: true, size: 15, line: ln });
    box(s, x, 2.05, 2.3, 1.1, d, { fill: bg, line: ln, size: 12.5 });
    note(s, x, 3.25, 2.3, 0.7, f, { size: 10.5, mono: true, align: "center" });
    if (i < 4) arrow(s, x + 2.3, 2.6, x + 2.56, 2.6, { lw: 2 });
  });
  note(s, 0.5, 4.0, 12.3, 0.4, "학습 시 ③④⑤는 하나의 모델(GraphSASRec)이며, 추천 loss의 gradient가 ⑤ → ④ → ③까지 흐릅니다. ①②는 임베딩 단계에서 한 번만 계산해 저장합니다.", { size: 12.5, color: H.dk1 });
  card(s, 0.5, 4.7, 6.0, 2.1, "학습되는 부분 (추천 loss로 갱신)",
    "• 4개 projection (Title·Entity·Action·Context, 1024→128)\n• 역할 결합 Linear (640→128)과 none/unknown 상태 벡터\n• Scene Residual MLP (512), GCN Linear (512→512)\n• SASRec Transformer, 위치 임베딩, User MLP", { fill: BLUE_BG, line: H.accent2, titleColor: H.accent2 });
  card(s, 6.8, 4.7, 6.0, 2.1, "고정된 부분",
    "• BGE large-en-v1.5 (텍스트 → 1024). 학습 중 갱신하지 않음\n• 그래프 구조: 어떤 Entity가 어떤 Action의 어느 역할인지\n• 장면 묶음 인덱스(scene_groups): 영상별 장면 수로 결정\n• 장면 유사도 Â는 고정이 아님 — 매 forward 다시 계산", { fill: GREY_BG });
}

// ======================================================================
pres.addSection({ title: "입력 데이터" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "입력 데이터" });
  s.addText("1. 입력: Scene Graph가 텐서가 되기까지", { placeholder: "title" });
  s.addText("graph_inputs.py · embed-representations 단계 · 출력 validation/representations/graph/<arm>_embeddings/", { placeholder: "body" });
}

// ---------- 3. JSONL → 노드/간선 ----------
{
  const s = content("장면 1개의 Scene Graph → 노드·간선·문맥 (graph_inputs.py)", "입력 데이터");
  box(s, 0.5, 1.35, 5.3, 2.75,
    "[Context]\nmedium: live_action\nformat: demonstration\ntopics: food preparation\n[Entities]\np1: person; wearing apron\nf1: tomato\nt1: knife\nl1: kitchen\n[Actions]\np1 - cutting - f1; t1; l1",
    { mono: true, size: 11.5, align: "left", valign: "top", fill: GREY_BG, margin: 10 });
  note(s, 0.5, 4.2, 5.3, 0.3, "v8 Scene Graph 출력 (파싱되어 scene_graph 객체로 저장)", { size: 11 });
  arrow(s, 5.85, 2.7, 6.45, 2.7, { lw: 2.5, color: H.accent1 });

  const rows = [
    headerRow(["배열", "이 장면에서 생기는 값", "뜻"]),
    ["nodes / types", "p1, f1, t1, l1 → type 0\ncutting → type 1", "노드 = Entity(0) 또는 Action(1).\n값은 BGE에 넣을 텍스트의 ID"],
    ["edges", "(p1→cutting, actor=0)\n(f1→cutting, target=1)\n(t1→cutting, tool=2)\n(l1→cutting, location=4)\n+ 역방향 4개 (role+5)", "Entity → Action 간선.\n역방향은 저장만 하고 모델은 쓰지 않음"],
    ["missing", "[cutting, −1, −1, −1, 0, −1]", "역할별 상태. −1 = 연결됨,\n0 = none, 1 = unknown.\nreceiver(4번째)는 항상 0"],
    ["contexts", "\"Format: demonstration\nMedium: live_action\nTopics: food preparation\"", "장면당 1개의 문맥 텍스트"],
  ].map((r, i) => i === 0 ? r : r.map((t, j) => ({ text: t, options: { fontFace: j === 1 ? "Courier New" : undefined, fontSize: j === 1 ? 10 : 11, valign: "middle" } })));
  s.addTable(rows, { x: 6.6, y: 1.35, w: 6.25, colW: [1.3, 2.6, 2.35], fontSize: 11, border: tblBorder, color: H.dk1, margin: 4, rowH: [0.33, 0.6, 1.05, 0.72, 0.72] });

  note(s, 0.5, 5.0, 12.3, 0.35, "BGE에 들어가는 텍스트 (ID는 넣지 않음)", { size: 14, bold: true, color: H.dk2 });
  box(s, 0.5, 5.4, 3.9, 0.85, "Entity\n\"Name: person\nAttributes: wearing apron\"", { mono: true, size: 11, fill: H.lt2, line: H.accent1 });
  box(s, 4.7, 5.4, 3.9, 0.85, "Action\n\"cutting\"", { mono: true, size: 11, fill: H.lt2, line: H.accent1 });
  box(s, 8.9, 5.4, 3.9, 0.85, "Context\n\"Format: …\\nMedium: …\\nTopics: …\"", { mono: true, size: 11, fill: H.lt2, line: H.accent1 });
  note(s, 0.5, 6.35, 12.3, 0.65, "같은 텍스트는 한 번만 인코딩합니다(text_id로 중복 제거). 제목은 Arm이 제목을 쓰는 경우에만 영상당 1개 text_id로 들어가고, 아니면 0(없음)입니다.\n구조 검사(validate_graph_structure)에 실패하거나 warning이 있는 장면은 제외되고 excluded_scenes에 기록됩니다.", { size: 11.5 });
}

// ---------- 4. GraphStore 저장 형식 ----------
{
  const s = content("저장 형식: 영상 → 장면 → 노드의 들쭉날쭉한 배열을 offsets로 이어 붙임", "입력 데이터");
  note(s, 0.5, 1.3, 12.3, 0.4, "예: 영상 A는 장면 2개(노드 3+2), 영상 B는 장면 1개(노드 4). 모든 영상의 장면·노드가 한 줄로 이어지고, 경계는 offsets 배열에만 있습니다.", { size: 12.5, color: H.dk1 });
  const rowsY = [1.95, 3.0, 4.05];
  const labels = ["영상 (video_offsets)", "장면 (contexts · scene_offsets)", "노드 (nodes · types · features)"];
  labels.forEach((l, i) => note(s, 0.5, rowsY[i] + 0.12, 2.6, 0.4, l, { size: 12, bold: true, color: H.dk2 }));
  // videos
  box(s, 3.3, rowsY[0], 5.4, 0.55, "영상 A  (장면 0–1)", { fill: AMBER_BG, line: H.accent3, size: 12 });
  box(s, 8.8, rowsY[0], 2.7, 0.55, "영상 B  (장면 2)", { fill: AMBER_BG, line: H.accent3, size: 12 });
  // scenes
  box(s, 3.3, rowsY[1], 3.2, 0.55, "장면 0  (노드 0–2)", { fill: BLUE_BG, line: H.accent2, size: 12 });
  box(s, 6.6, rowsY[1], 2.1, 0.55, "장면 1  (노드 3–4)", { fill: BLUE_BG, line: H.accent2, size: 12 });
  box(s, 8.8, rowsY[1], 2.7, 0.55, "장면 2  (노드 5–8)", { fill: BLUE_BG, line: H.accent2, size: 12 });
  // nodes
  const nodeLabels = ["E", "E", "A", "E", "A", "E", "E", "E", "A"];
  const nodeX = [3.3, 4.35, 5.4, 6.6, 7.65, 8.8, 9.45, 10.1, 10.75];
  const nodeW = [1.0, 1.0, 1.05, 1.0, 1.05, 0.6, 0.6, 0.6, 0.75];
  nodeLabels.forEach((t, i) => box(s, nodeX[i], rowsY[2], nodeW[i], 0.55, `${i}:${t}`, { fill: t === "A" ? H.lt2 : H.lt1, line: H.accent1, size: 11, mono: true }));
  note(s, 3.3, 4.7, 8.5, 0.3, "E = Entity (type 0), A = Action (type 1). 노드 번호는 전체 catalog에서 유일합니다.", { size: 11 });
  box(s, 0.5, 5.2, 6.0, 1.65,
    "video_offsets  = [0, 2, 3]        # 영상 i의 장면: [v[i], v[i+1])\nscene_offsets  = [0, 3, 5, 9]     # 장면 s의 노드: [o[s], o[s+1])\nedge_offsets, missing_offsets     # 같은 방식\ntitles         = [text_id_A, 0]   # 영상별 제목 text_id (0 = 없음)\nfeatures       = (고유 텍스트 수 × 1024) float32, 동결 BGE 출력",
    { mono: true, size: 10.5, align: "left", valign: "top", fill: GREY_BG, margin: 10 });
  card(s, 6.8, 5.2, 6.05, 1.65, "PackedGraphStore (graph_batching.py)",
    "• 학습 시작 때 video→node, video→edge 경계와 역할별 간선 목록을 미리 계산\n• 배치의 영상 ID를 받아 그 영상들의 장면·노드·간선을 한 번의 벡터 연산으로 모음(ranges)\n• BGE 특징 전체(features)를 GPU에 올릴 수 있으면 상주시킴 (feature_cache_mb, 기본 2048MB)", { fill: BLUE_BG, line: H.accent2, titleColor: H.accent2, size: 11.5 });
}

// ======================================================================
pres.addSection({ title: "모델 단계별" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "모델 단계별" });
  s.addText("2. 모델: SceneRoleEncoder.forward 를 한 단계씩", { placeholder: "title" });
  s.addText("graph_model.py 62–118행 · scene_gcn.py · 입력은 배치 하나(여러 영상의 장면·노드), 출력은 영상당 512차원 벡터", { placeholder: "body" });
}

// ---------- 5. projection ----------
{
  const s = content("단계 1 · Projection: 네 종류 입력을 각각 1024 → 128로", "모델 단계별");
  const cols = [
    ["Title", "영상당 1개\n(제목 없는 Arm·영상은 0)", "title_projection"],
    ["Entity", "장면의 Entity 노드\nE개", "entity_projection"],
    ["Action", "장면의 Action 노드\nA개", "action_projection"],
    ["Context", "장면당 1개\n(format·medium·topics)", "context_projection"],
  ];
  cols.forEach(([t, d, nm], i) => {
    const x = 0.6 + i * 3.1;
    box(s, x, 1.4, 2.8, 0.5, t, { fill: H.dk2, color: H.lt1, bold: true, size: 15, line: H.dk2 });
    box(s, x, 2.0, 2.8, 0.8, "BGE [동결]\n1024", { fill: GREY_BG, size: 12.5 });
    arrow(s, x + 1.4, 2.8, x + 1.4, 3.15);
    box(s, x, 3.15, 2.8, 0.8, `nn.Linear(1024, 128)\n${nm}`, { fill: BLUE_BG, line: H.accent2, size: 11.5 });
    arrow(s, x + 1.4, 3.95, x + 1.4, 4.3);
    box(s, x, 4.3, 2.8, 0.55, "128", { fill: H.lt2, line: H.accent1, bold: true, size: 14 });
    note(s, x, 4.95, 2.8, 0.7, d, { size: 11.5, align: "center" });
  });
  card(s, 0.6, 5.8, 12.1, 1.1, "코드에서 확인할 점",
    "• 네 projection은 가중치를 공유하지 않습니다(종류별 별도 Linear). bias 포함 각 131,200개 파라미터.\n• 제목은 title_available 마스크로 0 처리합니다. 즉 '제목 없음'은 Linear를 거친 뒤 0 벡터가 됩니다 (graph_model.py 65–66행).", { size: 12 });
}

// ---------- 6. 역할 결합 ----------
{
  const s = content("단계 2 · 역할 결합: Entity 정보를 Action 벡터에 한 번 더함", "모델 단계별");
  const roles = ["actor", "target", "tool", "location"];
  roles.forEach((r, i) => {
    const y = 1.5 + i * 1.0;
    circle(s, 1.3, y + 0.3, 0.46, `r_${r}`, { size: 9.5 });
    arrow(s, 1.75, y + 0.3, 3.3, 2.95 + (i - 1.5) * 0.12, { color: H.accent1 });
    note(s, 1.9, y - 0.05, 1.3, 0.3, r, { size: 10.5, italic: true, color: H.accent1 });
  });
  box(s, 3.3, 2.55, 1.9, 0.8, "Action a\n128", { fill: H.lt2, line: H.accent1, bold: true, size: 13 });
  arrow(s, 5.2, 2.95, 5.8, 2.95, { lw: 2 });
  box(s, 5.8, 2.2, 3.4, 1.5, "concat → 640\nnn.Linear(640, 128) → ReLU\n(role_fusion)", { fill: BLUE_BG, line: H.accent2, size: 12.5 });
  arrow(s, 9.2, 2.95, 9.8, 2.95, { lw: 2 });
  box(s, 9.8, 2.55, 2.9, 0.8, "a′ = a + ReLU(…)\n128", { fill: H.accent2, color: H.lt1, bold: true, size: 13, line: H.accent2 });
  note(s, 0.6, 5.5, 5.6, 0.5, "r_역할 = 연결된 Entity의 128차원 벡터 (단계 1 출력)\n연결이 없으면 → 학습 가능한 상태 벡터", { size: 12 });
  box(s, 0.6, 6.1, 5.6, 0.75, "role_states: nn.Parameter(4, 2, 128)\n[역할 4] × [none=0, unknown=1] × 128", { mono: true, size: 11, fill: GREY_BG, align: "left", margin: 8 });
  card(s, 6.5, 4.5, 6.3, 2.35, "구조상 특징 (바꿀 때 참고)",
    "• 방향이 한쪽입니다: Entity → Action. Entity 벡터는 갱신되지 않고 그대로 평균에 들어갑니다.\n• 1-hop 1회입니다. 층을 쌓거나 반복하지 않습니다.\n• receiver 역할은 입력에서 항상 '없음'으로 들어오고 모델도 읽지 않습니다 (ROLE_INDICES = 0,1,2,4).\n• 역할마다 Entity가 최대 1개입니다 (v8 형식: actor - action - target; tool; location).", { size: 12 });
}

// ---------- 7. readout + Scene MLP ----------
{
  const s = content("단계 3 · 장면 벡터: 네 성분 concat(512) → Residual MLP", "모델 단계별");
  const parts = [["t", "제목", H.accent3, AMBER_BG], ["mean(e)", "Entity 평균", H.accent1, H.lt2], ["mean(a′)", "보정 Action 평균", H.accent1, H.lt2], ["c", "Context", H.accent1, H.lt2]];
  parts.forEach(([f, d, ln, bg], i) => {
    const x = 0.6 + i * 1.75;
    box(s, x, 1.45, 1.6, 0.9, `${f}\n128`, { fill: bg, line: ln, size: 13, bold: true });
    note(s, x, 2.4, 1.6, 0.3, d, { size: 10.5, align: "center" });
    arrow(s, x + 0.8, 2.75, 3.95, 3.25, { color: ln });
  });
  box(s, 2.0, 3.25, 3.9, 0.7, "x_s = [t ; mean(e) ; mean(a′) ; c]  → 512", { fill: BLUE_BG, line: H.accent2, size: 13, bold: true });
  arrow(s, 3.95, 3.95, 3.95, 4.35, { lw: 2 });
  box(s, 1.3, 4.35, 5.3, 1.1, "Scene Residual MLP (scene_mlp)\nh_s = x_s + W₂ · ReLU(W₁ x_s)\n512 → 512 → 512 · LayerNorm 없음 (normalize=False)", { fill: BLUE_BG, line: H.accent2, size: 12.5 });
  arrow(s, 3.95, 5.45, 3.95, 5.85, { lw: 2 });
  box(s, 2.2, 5.85, 3.5, 0.65, "H  (S × 512)   S = 배치의 유효 장면 수", { fill: H.accent2, color: H.lt1, bold: true, size: 12.5, line: H.accent2 });

  // similarity branch
  arrow(s, 5.9, 1.9, 8.1, 1.9, { color: H.accent1, lw: 2, dash: "dash" });
  box(s, 8.1, 1.45, 4.6, 0.9, "z_s = [mean(e) ; mean(a′) ; c]  → 384\n(제목 제외 · Scene MLP 이전 값)", { fill: H.lt2, line: H.accent1, size: 12.5, bold: true });
  note(s, 8.1, 2.45, 4.6, 0.5, "→ 다음 단계의 장면 유사도 계산에만 사용 (gnn일 때만 만듦)", { size: 11.5, color: H.accent1 });
  card(s, 7.3, 3.3, 5.5, 3.3, "평균(segment_mean)의 뜻",
    "• mean(e): 이 장면에 속한 Entity 벡터의 산술 평균. Action에 연결되지 않은 Entity도 포함.\n• mean(a′): 역할 보정된 Action 벡터의 평균. Action이 없는 장면은 0.\n• 노드→장면 소속은 node_scenes 인덱스로 index_add_ 하여 계산.\n• 제목 t는 영상당 1개를 모든 장면에 복사해 붙입니다 (title[scene_videos]).\n\n따라서 장면 벡터는 '이 장면에 무엇이 있고, 무엇을 하며, 어떤 형식인가'의 평균 묘사이며, 노드 수나 순서는 남지 않습니다.", { size: 11.5 });
}

// ---------- 8. 유사도 그래프 ----------
{
  const s = content("단계 4a · 영상 안 장면 유사도 그래프 Â", "모델 단계별");
  // triangle graph
  const P = [[2.2, 2.0], [0.9, 4.2], [3.5, 4.2]];
  const W = [[0, 1, 0.8], [0, 2, 0.5], [1, 2, 0.0]];
  W.forEach(([i, j, w]) => { if (w > 0) arrow(s, P[i][0], P[i][1], P[j][0], P[j][1], { noHead: true, color: H.accent2, lw: 1 + 5 * w }); });
  P.forEach(([x, y], k) => circle(s, x, y, 0.42, `s${k}`, { line: H.accent2, size: 13 }));
  note(s, 2.75, 2.9, 1.2, 0.3, "0.8", { size: 12, bold: true, color: H.accent2 });
  note(s, 1.1, 2.9, 1.0, 0.3, "0.5", { size: 12, bold: true, color: H.accent2 });
  note(s, 1.9, 4.45, 0.8, 0.3, "0 (음수→0)", { size: 10.5, color: H.accent4 });
  note(s, 0.5, 5.0, 4.0, 1.2, "cos(z₁,z₂)가 0 이하면 간선이 사라집니다.\nself-loop(A_ii = 1)는 그림에서 생략.\n모든 장면 쌍을 보므로 장면 S개면 S² 항.", { size: 11.5 });

  const mat = (x, title, rows, hl) => {
    note(s, x, 1.35, 2.6, 0.35, title, { size: 13, bold: true, color: H.dk2 });
    s.addTable(rows.map((r) => r.map((v) => ({ text: v, options: { align: "center", fontFace: "Courier New", fontSize: 11, fill: { color: hl ? BLUE_BG : H.lt1 } } }))),
      { x, y: 1.75, w: 2.6, colW: [0.867, 0.867, 0.866], border: tblBorder, rowH: 0.42, margin: 2 });
  };
  mat(4.9, "A = max(cos, 0), 대각 1", [["1", ".80", ".50"], [".80", "1", "0"], [".50", "0", "1"]]);
  mat(7.8, "D = 행합 (대각 행렬)", [["2.30", "", ""], ["", "1.80", ""], ["", "", "1.50"]]);
  mat(10.7 - 0.5, "Â = D^-½ A D^-½", [[".435", ".393", ".269"], [".393", ".556", "0"], [".269", "0", ".667"]], true);
  note(s, 4.9, 3.15, 7.9, 0.5, "Â_ij = A_ij / √(D_i · D_j)      예: Â_01 = 0.80 / √(2.30 × 1.80) = 0.393", { size: 12, mono: true, color: H.dk1 });

  box(s, 4.9, 3.85, 7.9, 1.6,
    "unit    = F.normalize(z, dim=-1, eps=1e-8)          # 영벡터 보호\nweights = (unit @ unit.T).clamp_min(0)              # 음수 cosine → 0\nweights = where(eye, 1, weights)                    # self-loop = 1\nscale   = weights.sum(-1).rsqrt()                   # D^-1/2\nÂ       = weights * scale[:, None] * scale[None, :]",
    { mono: true, size: 10.5, align: "left", valign: "top", fill: GREY_BG, margin: 10 });
  card(s, 4.9, 5.65, 7.9, 1.25, "핵심 성질",
    "• Â는 학습 파라미터가 아니라 현재 z에서 매 forward 계산됩니다. z는 projection·역할 결합을 거치므로 gradient가 Â를 통해서도 흐릅니다(detach 없음).\n• temperature·top-k·threshold가 없어 cosine이 모두 높으면 Â의 모든 항이 비슷해집니다.", { size: 11.5, fill: BLUE_BG, line: H.accent2, titleColor: H.accent2 });
}

// ---------- 9. GCN 층 + 평균 ----------
{
  const s = content("단계 4b · GCN 1층과 영상 벡터: H′ = H + ReLU(ÂHW + b)", "모델 단계별");
  box(s, 0.6, 1.45, 2.4, 0.9, "H\nS × 512", { fill: BLUE_BG, line: H.accent2, bold: true, size: 13 });
  box(s, 0.6, 2.75, 2.4, 0.9, "Â\nS × S", { fill: H.lt2, line: H.accent1, bold: true, size: 13 });
  arrow(s, 3.0, 1.9, 3.7, 2.3, { lw: 2 });
  arrow(s, 3.0, 3.2, 3.7, 2.8, { lw: 2 });
  box(s, 3.7, 1.9, 2.6, 1.3, "Â @ H\n(이웃 장면 가중 평균)\nS × 512", { fill: GREY_BG, size: 12 });
  arrow(s, 6.3, 2.55, 6.9, 2.55, { lw: 2 });
  box(s, 6.9, 1.9, 2.6, 1.3, "nn.Linear(512, 512)\n+ ReLU\n(gcn.projection, bias 포함)", { fill: BLUE_BG, line: H.accent2, size: 12 });
  arrow(s, 9.5, 2.55, 10.1, 2.55, { lw: 2 });
  box(s, 10.1, 1.9, 2.7, 1.3, "H′ = H + ReLU(·)\nresidual\nS × 512", { fill: H.accent2, color: H.lt1, bold: true, size: 12.5, line: H.accent2 });
  arrow(s, 11.45, 3.2, 11.45, 3.65, { lw: 2 });
  box(s, 8.6, 3.65, 4.2, 0.9, "v = 유효 장면 평균 (segment_mean)\n영상당 512", { fill: AMBER_BG, line: H.accent3, bold: true, size: 13 });
  // residual arrow from H to H'
  arrow(s, 1.8, 1.45, 1.8, 1.25, { noHead: true });
  arrow(s, 1.8, 1.25, 11.45, 1.25, { noHead: true });
  arrow(s, 11.45, 1.25, 11.45, 1.9, {});
  note(s, 5.0, 1.28, 3.5, 0.25, "residual (H를 그대로 더함)", { size: 10, align: "center" });

  box(s, 0.6, 4.0, 7.6, 2.0,
    "class SceneGCN(nn.Module):\n    def __init__(self, dimension):           # dimension = 512\n        self.projection = nn.Linear(dimension, dimension)\n    def forward(self, scenes, similarity_features, groups):\n        output = torch.zeros_like(scenes)\n        for indices in groups:                  # 장면 수가 같은 영상 묶음\n            adjacency = normalized_similarity(similarity_features[indices])\n            values = scenes[indices]            # (영상 수, S, 512)\n            updated = values + relu(self.projection(adjacency @ values))\n            output = output.index_copy(0, indices.reshape(-1), updated.flatten(0, 1))\n        return output",
    { mono: true, size: 9.5, align: "left", valign: "top", fill: GREY_BG, margin: 10 });
  card(s, 8.6, 4.75, 4.2, 2.15, "이것이 GCN의 전부입니다",
    "• 층 1개, 가중치 행렬 1개(512×512 + bias = 262,656개)\n• 활성 ReLU, residual 있음, LayerNorm·dropout·gate 없음\n• 전파 후 평균이므로 장면 순서 정보 없음\n• 장면 1개 영상: Â=[1] → h + ReLU(hW + b)", { size: 11.5 });
  note(s, 0.6, 6.15, 7.6, 0.7, "유사도는 Scene MLP 이전의 z(384)로 계산하지만, 전파되는 값 H는 Scene MLP 이후의 512차원입니다. 즉 '누가 이웃인가'와 '무엇을 전달하는가'의 입력이 다릅니다.", { size: 11.5 });
}

// ---------- 10. 예외 경로 ----------
{
  const s = content("예외 경로: 장면이 없거나 제목이 없는 영상은 어떻게 처리되나", "모델 단계별");
  const cases = [
    ["유효 장면 ≥ 1", "일반 경로", "역할 결합 → Scene MLP → GCN → 평균.\n장면 1개도 self-loop GCN을 거칩니다.", H.lt2, H.accent1],
    ["장면 0 + 제목 있음", "fallback (GCN 건너뜀)", "[t ; 0 ; 0 ; 0] (512)을 같은 scene_mlp에 통과.\nGCN을 거치지 않으므로 출력 분포가 다를 수 있음.\n(graph_model.py 112–117행)", AMBER_BG, H.accent3],
    ["장면 0 + 제목 없음", "0 벡터", "available 마스크로 최종 출력 0.\n추천 점수는 항상 0 → 사실상 추천 불가.\n결측 때문에 catalog에서 빼지는 않음.", GREY_BG, H.accent5],
    ["Arm = meta", "GCN 없음", "GraphSASRec이 아니라 일반 SASRec(제목 1024 → 512 → MLP).\n같은 graph 모드로 실행해도 장면을 전혀 읽지 않음.", GREY_BG, H.accent5],
  ];
  cases.forEach(([h, t, b, bg, ln], i) => {
    const x = 0.5 + i * 3.1;
    box(s, x, 1.45, 2.9, 0.6, h, { fill: ln, color: i >= 2 ? H.dk1 : H.lt1, bold: true, size: 14, line: ln });
    box(s, x, 2.05, 2.9, 0.5, t, { fill: bg, line: ln, size: 12.5, bold: true });
    box(s, x, 2.55, 2.9, 2.0, b, { fill: H.lt1, line: ln, size: 11.5, align: "left", valign: "top", margin: 8 });
  });
  card(s, 0.5, 4.9, 12.3, 1.95, "어떤 장면이 '유효'한가 (graph_inputs.py 134–155행)",
    "• scene_graph가 dict가 아니면(파싱 실패 원문) → raw 로 제외\n• warning 필드가 있으면 → warning 으로 제외 (내용과 무관하게 제외)\n• actions 키가 없으면 unsupported_format, validate_graph_structure 실패면 invalid_structure 로 제외\n• 통과한 장면은 Action이 0개여도 유효 장면입니다(actionless 통계만 기록). 제외 사유는 <arm>_embeddings/의 statistics에 남습니다.", { size: 12 });
}

// ======================================================================
pres.addSection({ title: "실행·학습" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "실행·학습" });
  s.addText("3. 실행과 학습: 배치는 어떻게 만들고 무엇으로 학습하나", { placeholder: "title" });
  s.addText("graph_batching.py · graph_model.py GraphSASRec · model.py in_batch_loss · rolling_recommendation.py", { placeholder: "body" });
}

// ---------- 11. batching ----------
{
  const s = content("배치 처리: 장면 수별 묶음과 S² 기준 chunk 분할", "실행·학습");
  note(s, 0.5, 1.3, 6.2, 0.4, "scene_groups(offsets) — 배치 안 영상을 장면 수별로 묶음", { size: 14, bold: true, color: H.dk2 });
  const vids = [["A", 2], ["B", 3], ["C", 3], ["D", 5], ["E", 2]];
  vids.forEach(([n, c], i) => {
    const x = 0.5 + i * 1.25;
    box(s, x, 1.8, 1.1, 0.5, `영상 ${n}`, { fill: AMBER_BG, line: H.accent3, size: 12 });
    note(s, x, 2.35, 1.1, 0.3, `S = ${c}`, { size: 11, align: "center" });
  });
  arrow(s, 3.5, 2.7, 3.5, 3.1, { lw: 2 });
  box(s, 0.5, 3.1, 1.9, 0.9, "scene_group_2\n[A, E] → (2, 2)", { fill: BLUE_BG, line: H.accent2, size: 11.5, mono: true });
  box(s, 2.6, 3.1, 1.9, 0.9, "scene_group_3\n[B, C] → (2, 3)", { fill: BLUE_BG, line: H.accent2, size: 11.5, mono: true });
  box(s, 4.7, 3.1, 1.9, 0.9, "scene_group_5\n[D] → (1, 5)", { fill: BLUE_BG, line: H.accent2, size: 11.5, mono: true });
  note(s, 0.5, 4.1, 6.2, 1.1, "각 묶음은 (영상 수, S) 장면 인덱스 행렬입니다. 묶음마다 Â를 (영상 수, S, S)로 한 번에 계산하고 배치 행렬곱으로 전파합니다. 영상 사이에는 간선이 생기지 않습니다(묶음 안에서도 영상별로 독립된 Â).\n인덱스는 CPU에서 만들어 배치에 넣습니다(group_scenes=True).", { size: 11.5 });

  const rows = [headerRow(["chunk 기준 (config validation.graph_execution)", "기본값", "역할"]),
    ["chunk_items", "256", "chunk당 최대 영상 수"],
    ["chunk_nodes", "32,768", "chunk당 최대 노드 수"],
    ["chunk_scene_pairs (gnn만)", "1,048,576", "chunk당 Σ S² 상한. 한 영상이 넘으면 단독 처리"],
    ["checkpoint", "auto", "활성 메모리 추정이 크면 gradient checkpoint 사용"],
    ["feature_cache_mb", "2,048", "BGE 특징을 GPU에 상주시킬 상한"],
  ];
  s.addTable(rows, { x: 7.0, y: 1.35, w: 5.85, colW: [2.6, 0.95, 2.3], fontSize: 11, border: tblBorder, color: H.dk1, margin: 4, rowH: 0.42 });
  card(s, 7.0, 4.35, 5.85, 2.5, "학습 1 step에서 일어나는 일 (GraphSASRec)",
    "1. 배치의 시청 이력·정답에 나오는 영상 ID를 모아 unique 처리\n2. chunk로 나눠 SceneRoleEncoder 실행 → 영상 벡터 캐시(_vectors)\n3. SASRec이 이력 위치마다 캐시에서 벡터를 찾아 사용자 벡터 계산\n4. loss → backward: gradient가 encoder까지 전달\n5. optimizer step 후 캐시 폐기(train() 호출 시 clear_item_cache)\n평가 때는 catalog 전체(19,738개)를 한 번 인코딩해 catalog_vectors로 둡니다.", { size: 11.5, fill: BLUE_BG, line: H.accent2, titleColor: H.accent2 });
}

// ---------- 12. 추천 + loss ----------
{
  const s = content("추천 모델과 손실: 영상 벡터 → SASRec 사용자 벡터 → 내적", "실행·학습");
  const flow = [["시청 이력\n최대 10개 × 512", GREY_BG, H.accent5], ["+ 위치 임베딩\n(10 × 512)", H.lt1, H.accent5], ["Transformer\n2 blocks · 2 heads\nnorm_first · causal", BLUE_BG, H.accent2], ["LayerNorm →\n마지막 유효 위치", H.lt1, H.accent5], ["User Residual MLP\n(GELU · LayerNorm)", BLUE_BG, H.accent2], ["u · v_cand", H.accent3, H.accent3]];
  flow.forEach(([t, bg, ln], i) => {
    const x = 0.5 + i * 2.08;
    box(s, x, 1.45, 1.85, 1.1, t, { fill: bg, line: ln, size: 11.5, bold: i === 5, color: i === 5 ? H.lt1 : H.dk1 });
    if (i < 5) arrow(s, x + 1.85, 2.0, x + 2.08, 2.0, { lw: 1.5 });
  });
  note(s, 0.5, 2.65, 12.3, 0.4, "후보 영상 벡터 v_cand는 같은 SceneRoleEncoder 출력이며 Transformer를 거치지 않습니다. 사용자와 영상이 같은 512차원 공간에서 내적으로 비교됩니다.", { size: 12 });

  card(s, 0.5, 3.25, 6.0, 3.6, "학습 손실 in_batch_loss (model.py 253행~)",
    "• 이력의 모든 위치에서 다음 영상을 맞히는 next-item 예측\n• 후보 = 같은 배치의 정답 영상들 (in-batch negatives)\n• logits = u · v_cand − log p(인기도)   ← logQ 보정 (popularity_power 1.0)\n• 사용자가 이미 본 영상과 자기 정답 외 중복은 −1e4로 마스킹\n• cross-entropy (softmax over batch candidates)\n\n최적화: AdamW lr 1e-4 · weight decay 0.1 · batch 128 · gradient clipping · deterministic", { size: 12 });
  const rows = [headerRow(["단계", "데이터 구간", "역할"]),
    ["Selection", "D−1일 이전 학습, [D−1, D) 검증", "검증 NDCG@10으로 epoch 수 선택 (patience 5, max 50)"],
    ["Refit", "D 이전 전체", "모델·optimizer 초기화 후 선택한 epoch만큼 다시 학습"],
    ["Test", "[D, D+1)", "이벤트별 순위·HR·NDCG@K (K = 4, 8, 10, 20, 30)"],
  ];
  s.addTable(rows, { x: 6.8, y: 3.25, w: 6.05, colW: [1.1, 2.3, 2.65], fontSize: 11, border: tblBorder, color: H.dk1, margin: 4, rowH: [0.4, 0.7, 0.7, 0.7] });
  note(s, 6.8, 5.85, 6.05, 1.0, "평가일 7일 × seed 3개(42, 43, 44) × Arm마다 독립 학습. 각 조합은 recommendations/graph/gnn/<date>/seed_<seed>/<arm>/에 sasrec.pt · training.json · per_event_metrics.jsonl · catalog_vectors.npy로 저장됩니다.", { size: 11.5 });
}

// ---------- 13. mean vs gnn ----------
{
  const s = content("mean과 gnn의 차이는 GCN 한 층뿐입니다", "실행·학습");
  const rows = [headerRow(["항목", "--scene-aggregation mean", "--scene-aggregation gnn"]),
    ["모델 계약 (graph_context.py)", "sasrec-role-graph/v7 · GRAPH_MODEL", "sasrec-role-graph/v8 · GNN_MODEL (gcn 설정 포함)"],
    ["장면 encoder", "projection → 역할 결합 → Scene MLP", "동일"],
    ["장면 → 영상", "유효 장면 평균", "유사도 Â 계산 → 1층 residual GCN → 유효 장면 평균"],
    ["추가 파라미터", "없음", "gcn.projection 512×512 + 512 = 262,656개"],
    ["추가 입력", "없음", "z_s (384) · scene_group_* 인덱스 (CPU에서 생성)"],
    ["chunk 기준", "items · nodes", "items · nodes · Σ S² (chunk_scene_pairs)"],
    ["결과 경로", "recommendations/graph/mean/ · graph_mean_diagnosis.json", "recommendations/graph/gnn/ · graph_gnn_diagnosis.json"],
    ["고정 Graph 입력", "representations/graph/<arm>_embeddings/ 공유 (role-graph-inputs/v3)", "동일 — 재임베딩 불필요"],
  ];
  s.addTable(rows, { x: 0.5, y: 1.35, w: 12.3, colW: [2.6, 4.6, 5.1], fontSize: 11.5, border: tblBorder, color: H.dk1, margin: 5, rowH: 0.47 });
  note(s, 0.5, 5.8, 12.3, 1.0, "비용 참고 (docs/graph_training.md, 합성 입력 · 영상 64개 · 장면 315개 · L4 GPU): mean 21.1 ms/batch, gnn 44.6 ms/batch (약 2.1배). 실제 데이터의 장면 수 분포(평균 약 6개)에 따라 달라지며, 추천 품질 차이는 아직 측정되지 않았습니다.", { size: 11.5 });
}

// ---------- 14. 파라미터 ----------
{
  const s = content("학습 파라미터 수 (코드로 모델을 만들어 센 값)", "실행·학습");
  const stats = [["1,395,840", "SceneRoleEncoder (gnn)"], ["262,656", "그중 GCN 층"], ["6,837,248", "GraphSASRec 추천부"], ["8,233,088", "gnn 모델 합계"]];
  stats.forEach(([n, l], i) => {
    const x = 0.5 + i * 3.1;
    box(s, x, 1.35, 2.9, 1.35, n, { fill: i === 1 ? H.accent2 : H.lt2, color: i === 1 ? H.lt1 : H.dk2, bold: true, size: 26, line: i === 1 ? H.accent2 : H.accent1 });
    note(s, x, 2.75, 2.9, 0.35, l, { size: 12, align: "center" });
  });
  const rows = [headerRow(["모듈", "모양", "파라미터", "비고"]),
    ["title/entity/action/context_projection", "4 × Linear(1024, 128)", "524,800", "종류별 별도 가중치"],
    ["role_fusion", "Linear(640, 128)", "82,048", "Entity→Action 결합"],
    ["role_states", "Parameter(4, 2, 128)", "1,024", "none/unknown 상태"],
    ["scene_mlp", "2 × Linear(512, 512)", "525,312", "residual, LayerNorm 없음"],
    ["gcn.projection", "Linear(512, 512)", "262,656", "gnn에서만 존재"],
    ["position_embedding", "Embedding(10, 512)", "5,120", "시청 순서"],
    ["encoder (Transformer)", "2 × EncoderLayer(512, ff 2048)", "6,304,768", "SASRec 본체"],
    ["norm + user_mlp", "LayerNorm + 2 × Linear(512, 512)", "527,360", "사용자 벡터"],
  ];
  s.addTable(rows, { x: 0.5, y: 3.3, w: 12.3, colW: [3.4, 3.4, 1.6, 3.9], fontSize: 11.5, border: tblBorder, color: H.dk1, margin: 4, rowH: 0.36 });
  note(s, 0.5, 6.65, 12.3, 0.3, "mean 모델 합계는 7,970,432개입니다. BGE(약 3.35억)는 동결이며 포함하지 않았습니다. 추천부의 item_projection·item_mlp는 GraphSASRec에서 삭제됩니다.", { size: 11 });
}

// ======================================================================
pres.addSection({ title: "수정 가이드" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "수정 가이드" });
  s.addText("4. GNN 구조를 바꾸려면", { placeholder: "title" });
  s.addText("고칠 파일 · 캐시가 재사용되지 않게 하는 방법 · 현재 구조에서 보이는 특성과 미검증 가설", { placeholder: "body" });
}

// ---------- 15. 수정 지점 ----------
{
  const s = content("바꿀 파일과 반드시 함께 바꿀 것", "수정 가이드");
  const rows = [headerRow(["파일", "무엇이 있나", "바꿀 때"]),
    ["validation/scene_gcn.py", "normalized_similarity, SceneGCN", "인접 행렬 정의(temperature·top-k·시간 간선), 층 수, attention 등 GCN 본체"],
    ["validation/graph_model.py 84–111", "z_s 구성, GCN 호출 위치, 평균 풀링", "유사도 입력 바꾸기, GCN을 Scene MLP 앞으로, attention 풀링 등"],
    ["validation/graph_context.py 23–43", "GNN_ARCHITECTURE, GNN_MODEL", "필수: 새 버전 문자열과 gcn 설정 기록. 캐시 키·checkpoint metadata에 들어감"],
    ["validation/recommendation_contracts.py 6", "DIAGNOSIS_ARCHITECTURE_VERSIONS", "필수: 새 버전 추가. 없으면 run-diagnosis가 결과를 거부"],
    ["validation/cli.py 37 · graph_context.py", "--scene-aggregation choices, validate_mode", "새 이름(예: gnn_v2)을 추가할 때"],
    ["tests/validation/test_scene_gcn.py 등", "GCN 수치·gradient·격리 검사", "기존 수치 기대값은 바뀜. 새 구조에 맞춰 갱신"],
  ];
  s.addTable(rows, { x: 0.5, y: 1.35, w: 12.3, colW: [3.3, 3.4, 5.6], fontSize: 11.5, border: tblBorder, color: H.dk1, margin: 5, rowH: 0.55 });
  card(s, 0.5, 5.3, 6.0, 1.6, "왜 버전 문자열을 꼭 바꿔야 하나",
    "추천 결과는 identity(encoder·item_model·graph_model·graph_architecture 등)의 해시로 shared_cache에 저장됩니다. 구조만 바꾸고 GNN_MODEL을 그대로 두면 해시가 같아 이전 GNN 결과가 학습 없이 복원될 수 있습니다 (recommendation_cache.py 28–35행).", { size: 11.5, fill: AMBER_BG, line: H.accent3, titleColor: H.accent3 });
  card(s, 6.8, 5.3, 6.05, 1.6, "권장: 새 aggregation 이름으로 추가",
    "결과 폴더가 recommendations/graph/<aggregation>/ 이므로, 기존 gnn을 직접 고치면 같은 Run 안에서 이전 결과를 덮어씁니다. gnn_v2 같은 이름을 추가하면 mean · gnn · gnn_v2를 한 Run에서 나란히 비교할 수 있고 고정 Graph 입력은 공유합니다.", { size: 11.5, fill: BLUE_BG, line: H.accent2, titleColor: H.accent2 });
}

// ---------- 16. 관찰과 가설 ----------
{
  const s = content("현재 구조에서 보이는 특성 (코드로 확인) 과 개선 가설 (미검증)", "수정 가이드");
  const facts = [
    ["유사도가 고르면 GCN은 MLP 한 층과 같음", "Â의 모든 항이 1/S에 가까우면 Â·H ≈ 평균이 되어 v ≈ mean(H) + ReLU(W·mean(H) + b). 그래프 구조가 결과에 기여하지 않음. 같은 영상의 장면은 Context가 비슷해 이럴 가능성이 있으나 실제 Â 분포는 측정하지 않음."],
    ["시간 정보 없음", "장면 순서를 잇는 간선도, 위치 인코딩도 없음. 장면 순서를 섞어도 영상 벡터가 같음."],
    ["음수 유사도는 0으로 잘림", "대비되는 장면 사이 정보는 쓰지 않음. 날카롭게 만드는 temperature·top-k도 없음."],
    ["역할 그래프는 한 방향 1-hop", "Entity는 Action에서 정보를 받지 않음. 역방향 간선은 저장만 됨."],
  ];
  facts.forEach(([t, b], i) => {
    const x = 0.5 + (i % 2) * 6.25, y = 1.35 + Math.floor(i / 2) * 1.75;
    card(s, x, y, 6.05, 1.6, t, b, { size: 11.5, titleSize: 14 });
  });
  note(s, 0.5, 4.95, 12.3, 0.35, "개선 가설 — 효과는 아직 실험하지 않았습니다", { size: 14, bold: true, color: H.accent3 });
  const ideas = ["Â에 temperature softmax 또는 top-k 적용", "학습 가능한 attention 간선 (GAT 방식)", "시간상 인접 장면 간선 추가 또는 위치 인코딩", "GCN 2층 이상 · 평균 대신 attention 풀링", "Entity도 갱신하는 양방향 역할 메시지 전달", "구조 변경 전에 실제 Â의 비대각 가중치·행 entropy 측정"];
  ideas.forEach((t, i) => {
    const x = 0.5 + (i % 3) * 4.15, y = 5.4 + Math.floor(i / 3) * 0.75;
    box(s, x, y, 3.95, 0.6, t, { fill: AMBER_BG, line: H.accent3, size: 11.5, align: "left", margin: 8 });
  });
}

// ---------- 17. 요약 ----------
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "수정 가이드" });
  s.addText("요약", { placeholder: "title" });
  s.addText([
    { text: "GNN = 장면 안 '역할 결합 1-hop' + 영상 안 '유사도 GCN 1층' + 평균 풀링", options: { bullet: true, breakLine: true } },
    { text: "Â는 제목을 뺀 384차원 cosine, 매 forward 재계산, 파라미터 없음 · 학습 파라미터는 GCN Linear 512×512 하나", options: { bullet: true, breakLine: true } },
    { text: "구조를 바꾸면 graph_context.py의 버전·설정과 DIAGNOSIS_ARCHITECTURE_VERSIONS를 함께 바꿔야 캐시 재사용을 피함", options: { bullet: true } },
  ], { placeholder: "body" });
}

(async () => {
  const out = path.resolve(process.env.OUT || path.join(__dirname, "scene_gcn_walkthrough.pptx"));
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
})();
