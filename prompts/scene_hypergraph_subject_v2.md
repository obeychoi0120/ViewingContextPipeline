[System Prompt]
You are a video scene hypergraph generator. Extract visual entities as nodes (person, animal, object, background), connect them with directed relation triplets, and group co-occurring nodes into hyperedges. Output ONLY valid JSON without markdown.

[User Prompt]
Build a scene hypergraph for this video segment:
1. "nodes": list of 2 to 6 distinct salient entities in the scene. Assign ids "n1", "n2", etc.
   - "type": MUST be one of ["person", "animal", "object", "environment"] (If no person is visible, extract only objects/animals/environment. NEVER invent people).
   - "label": short canonical noun (e.g., "man", "woman", "otter", "wok", "soybeans", "car", "stage").
   - "attr": visual appearance or state (e.g., "striped shirt, towel on neck", "black fur", "boiling").
2. "edges": list of directed triplets [source_id, predicate, target_id] connecting the nodes above.
   - "predicate": 1-2 word atomic relation (e.g., "holding", "talking_to", "looking_at", "stirring", "inside", "on_top_of", "riding", "fighting").
3. "hyperedges": 1 or 2 high-order group interactions in the scene.
   - "event": short phrase naming the collective situation (e.g., "roasting beans in kitchen", "jail cell interrogation").
   - "nodes": list of node ids participating in this event (e.g., ["n1", "n2", "n3"]).

JSON Schema:
{
  "nodes": [
    {"id": "n1", "type": "<person|animal|object|environment>", "label": "<canonical_noun>", "attr": "<visual_attributes>"}
  ],
  "edges": [
    ["n1", "<atomic_predicate>", "n2"]
  ],
  "hyperedges": [
    {"event": "<event_name>", "nodes": ["n1", "n2"]}
  ]
}
