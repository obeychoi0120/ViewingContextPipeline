# TO-BE concept image — 2026-09-11

Generated with the built-in imagegen tool. This is a review proposal; it is not applied to the PPTX or README diagram.

## Initial generation prompt

Create ONE polished TO-BE research pipeline diagram as a standalone PNG, landscape 16:9 at high resolution, using the attached image ONLY as context for the four research arms and its restrained green / blue / gray palette. This is a new proposed design, not an update of the existing PPTX. Do not reproduce obsolete evaluation details from the reference. Flat white background, crisp editorial scientific diagram, professional sans-serif typography, generous spacing, clean straight arrow routing, no decoration, no 3D. All text must be accurate, large enough to read in a README, English throughout. A balanced compact design with 4 aligned horizontal arm lanes and one shared evaluation band below. No Mermaid syntax. Avoid the huge title and repeated long labels of the reference.

Exact title: "ViewingContextPipeline"
Subtitle: "Visual item representations for sequential recommendation · MicroLens-100K · v4"
Small top-right badge: "TO-BE CONCEPT"

Data strip below title: "100,000 users  ·  719,405 interactions  ·  19,738 items"
A separate shared control / input strip: "Common user histories: strictly earlier events, last 10 items"
Make it visually clear this history is an INPUT TO ALL FOUR RECOMMENDER MODELS, not input to the visual extraction models. Route a clearly labelled common history rail into each SASRec block. The four recommenders have identical architecture and separate trained weights; never merge arm representations into a shared trained model.

Main grid column headings:
"ARM" | "VISUAL / TEXT INPUT" | "SCENE EXTRACTION" | "VIDEO TEXT" | "ENCODING" | "RECOMMENDATION"
Four labelled rows:
1. "Graph Qwen" (green)
Input "Shared keyframes" with a small note "Fixed 30s · up to 6 frames"
Scene extraction block "Qwen3-VL-2B" then smaller "Scene graphs"
Video text block "Qwen summary" then smaller "Graph → text"
Encoding block "Frozen BGE" then smaller "1024-D"
Recommendation block "Projection + SASRec" with "Trainable"
2. "Graph Gemini" (a distinguishable teal)
Input same shared keyframes, graph extraction "Gemini 3.7 Flash" then "Scene graphs"
Video text explicitly "Qwen summary" then "Graph → text" (NOT Gemini summary model)
Encoding "Frozen BGE" / "1024-D"
Recommendation "Projection + SASRec" / "Trainable"
3. "Description" (blue)
Input same shared keyframes
Extraction "Qwen3-VL-2B" then "Scene descriptions"
Video text "Qwen summary" then "Description → text"
Encoding "Frozen BGE" / "1024-D"
Recommendation "Projection + SASRec" / "Trainable"
4. "Metadata" with smaller "Baseline" (neutral slate gray)
Input "English title"
A direct bypass arrow across the scene-extraction and video-text columns into "Frozen BGE" / "1024-D", followed by "Projection + SASRec" / "Trainable".
No learned item-ID embedding block; histories are IDs used to look up the per-arm representations.

Under encoding / recommendation, one compact note:
"BGE-large-en-v1.5 · Same architecture, independently trained per arm"

All four row outputs feed ONE shared bottom evaluation band. Three readable sections in that band:
"ROLLING PROTOCOL"
"7 days × 3 seeds × 4 arms = 84 runs"
"Select epoch → fresh refit → test"
"FULL-CATALOG RANKING"
"19,738 candidates"
"Primary: NDCG@10 · Also: HR@10"
"STATISTICAL COMPARISONS"
"Visual vs Metadata: superiority"
"Graph vs Description: 5% non-inferiority"
"Gemini vs Qwen: exploratory"

Small but readable footer in at most two lines:
"User-level paired bootstrap · 10,000 samples · Bonferroni within each confirmatory family"
"Gemini summary missing: Qwen Graph summary fallback · Raw outputs / scene coverage reported separately"

Do not include "Catalog spread", "Leave-Two-Out", "ID Baseline", 3-keyframe sampling, fake results or measured superiority claims. Do not include VM / Conda / retry / worker implementation details. Keep all four workflows aligned, arrows connected, labels short, all containers comfortably enclosing their text. Favor readability over fitting excess ornamental elements.

Reference: ../ViewingContextPipeline_260907.png

## Targeted correction prompt

Make a precise minimal correction to this diagram; preserve all boxes, colors, all four arm rows, all text, layout, shared history input rail and evaluation band. Change ONLY the gap between the Metadata row and the SHARED EVALUATION band: (1) remove ALL FIVE downward arrows currently pointing from various pipeline columns into the evaluation band. They misleadingly suggest evaluation of raw inputs and intermediate representations. Leave this as a clean gap, with no new arrows. The band heading SHARED EVALUATION (ALL FOUR ARMS) already describes the scope. (2) Replace the existing single note 'BGE-large-en-v1.5 · Same architecture, independently trained per arm' with two separate, clearly positioned short notes: under the ENCODING column write 'BGE-large-en-v1.5 · frozen'; under the RECOMMENDATION column write 'Same architecture · separate weights'. Fit each note neatly under its corresponding column, with no overlap. Do not change any other visible text or content. Preserve the full uncropped landscape image and its crisp readable typography.
