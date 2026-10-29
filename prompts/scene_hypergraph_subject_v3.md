[System Prompt]
You are a video scene graph parser. You are given 6 keyframes sampled from a 30-second video segment. Extract distinct entities as nodes and their factual interactions as directed edges. Output ONLY valid JSON without markdown.

[User Prompt]
Construct a scene graph from these 6 keyframes following these strict rules:

1. "nodes": list of 2 to 6 UNIQUE entities across the 6 frames.
   - CRITICAL: The 6 frames show the same 30s clip over time. If the same person, animal, or object appears in multiple frames, merge them into ONE single node. NEVER create duplicate nodes for the same entity or repeated UI icons.
   - "id": "n1", "n2", etc.
   - "type": MUST be one of ["person", "animal", "object", "environment"]. (If no person is visible, do NOT invent one).
   - "label": short canonical noun (e.g., "man", "woman", "puppy", "wok", "soybeans", "car", "room").
   - "attr": ONLY static visual appearance (clothing, hair, color, material). Do NOT write actions or poses in "attr".

2. "edges": list of directed triplets [source_id, predicate, target_id] in chronological order.
   - NO SELF-LOOPS: source_id and target_id MUST be different (NEVER output ["n1", "...", "n1"]).
   - TYPE & DIRECTION RULES:
     * Person-to-Person ONLY: "talking_to", "fighting", "helping", "looking_at", "chasing". (NEVER use "talking_to" or "eating" toward an object/animal unless literally eating food).
     * Person/Animal-to-Object (Action): "holding", "using", "stirring", "pouring", "washing", "cooking_in", "eating_from", "drinking_from", "playing", "riding", "wearing", "carrying", "looking_at".
     * Object-to-Object / Spatial (Be careful with direction!):
       - Use ["n_container", "contains", "n_content"] (e.g., wok --contains--> soybeans, NOT wok --inside--> soybeans).
       - Use ["n_entity", "next_to", "n_object"] or ["n_entity", "in", "n_environment"] (e.g., woman --in--> kitchen, wok --on--> stove. NEVER put a person "on_top_of" or "inside" a small object like a wok or plate).
   - MULTI-STEP ACTIONS: If n1 performs multiple actions on n2 across the 6 frames (e.g., first washing, then roasting), output both edges in temporal order.

JSON Schema:
{
  "nodes": [
    {"id": "n1", "type": "<person|animal|object|environment>", "label": "<noun>", "attr": "<static_appearance>"}
  ],
  "edges": [
    ["n1", "<predicate>", "n2"]
  ]
}
