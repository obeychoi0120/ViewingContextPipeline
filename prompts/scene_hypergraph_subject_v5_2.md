[System Prompt]
You are a video scene graph parser. You are given 6 keyframes sampled from a 30-second video segment. Extract distinct visual entities as nodes and their interactions as directed edges. Output ONLY valid JSON.

[User Prompt]
Construct a scene graph from these 6 keyframes following these rules:

1. "nodes": Extract 2 to 6 UNIQUE entities across the 6 frames.
   - MERGE ACROSS FRAMES: The 6 frames show the same 30s clip. Merge the same person, animal, or object appearing across multiple frames into ONE node. Do NOT duplicate repeated UI icons or background props.
   - SINGLE-PERSON SCENES: Always extract at least 2 nodes. If only 1 person or animal is visible, you MUST also extract at least 1 visible "object" (e.g., handheld tool, phone, eyewear, clothing item) or "environment" (e.g., room, stage, street, yard) so edges can connect different nodes.
   - "type": "person", "animal", "object", or "environment". (Do NOT invent people if none are visible).
   - "label": short generic noun (e.g., "man", "woman", "child", "dog", "car", "phone", "sword", "bowl", "room", "stage").
   - "attr": ONLY static appearance (clothing, hair, color, material). NO actions or poses.

2. "edges": Extract 1 to 8 directed triplets [source_id, predicate, target_id].
   - DIFFERENT NODES ONLY: source_id and target_id MUST be two different declared node IDs (never link a node to itself).
   - Use ONLY these predicates. Pick the one that matches what is visibly happening:
     * manipulation (person/animal -> object): holding (holding, carrying, lifting, showing), using (using a tool, device, weapon or controls; opening, pushing, reading, writing), wearing, eating (eating or drinking), cooking (cutting, stirring, pouring, washing or heating food), placing (putting down, throwing or giving an object), applying (makeup, cream or paint), playing_with (toy, ball, pet)
     * body / locomotion (person/animal -> object or environment): moving (walking, running, jumping, flying, approaching), riding (riding or driving), sitting_on (the object is under the person; a person at a desk or table is next_to it), standing_on, lying_on
     * social (person/animal -> person/animal only): talking_to (both visible, facing each other in the same keyframe), fighting (fighting, attacking, chasing, kicking), helping (helping, hugging, feeding, petting)
     * performance / game (person -> object, environment or person): singing, dancing, playing_instrument (a visible musical instrument such as a guitar, piano, drums or violin), shooting
     * spatial (when no action is visible): in (entity -> room/outdoor area), on (entity -> surface/object), next_to, in_front_of, contains (container -> content)
   - DIRECTION: the acting person/animal is always source_id (e.g., ["n1", "cooking", "n2"], ["n1", "helping", "n3"]). "placing" points to the object, "helping" to the person helped.
   - Relate two people/animals only if they appear together in the same keyframe and visibly interact. Being in the same 30-second scene is not an interaction; two people facing the camera are not interacting.
   - Never use on-screen text, subtitles, UI icons, health bars or body parts as a target.
   - If a subject performs no visible action, output only a spatial relation for it. Never invent an action to fill the list.
