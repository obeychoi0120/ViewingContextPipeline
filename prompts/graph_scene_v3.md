Represent the supplied video segment as a concise scene graph in English.

[Schema]
Return one JSON object with exactly these three fields:

entities: list[{id: str, name: str, attributes: list[str]}]
relations: list[{subject_id: str, predicate: str, object_id: str}]
context: list[str]

[Rules]

Evidence:
- The frames are chronological samples and may contain repeated views or unrelated shots. Consider all frames, but select only distinctive, important observations.
- Describe only supported actions and changes. Do not invent events between frames or assume continuity across unrelated shots.
- Use on-screen text to understand the visible content. Express the resulting meaning in your own words. Do not quote, transcribe, or translate the wording, or paraphrase captions line by line. Distinguish visible events from things that text merely mentions or compares.
- Grounded interpretations of genre, purpose, and setting are allowed. Keep them brief and preserve uncertainty. Do not invent identities, motives, personal histories, or relationships.

Entities:
- Include at most 4 important entities. Prioritize main subjects, action targets, and tools. Omit incidental background details.
- Give distinct instances distinct IDs, even when their names are the same. Reuse an ID for repeated views of the same instance when continuity supports it. Do not create a new entity for every frame.
- Use a short kind label for name, such as person, cup, or bird. Do not use real names or identities.
- Give each entity at most 3 attributes. Each attribute is a short phrase of at most 6 words describing distinctive appearance, state, or an action without an identified target.
- Do not create entities merely to store on-screen wording.

Relations:
- Include at most 4 distinct, supported relations.
- subject_id and object_id must use IDs from entities. predicate is a short phrase describing the connection.
- Preserve who acts on what. Do not connect unrelated shots, invent a target, or substitute a person for an object.
- Put actions without an identified target in attributes. Do not invent self-relations. An entity may have no relations.

Context:
- Usually write one short sentence. Use at most 3 sentences in total, with one sentence per string and at most 60 words across the entire array.
- Include only additional setting, presentation, or grounded interpretation not already captured by entities and relations. Briefly mention unrelated shot changes when relevant.
- Do not narrate each frame, repeat actions or attributes, or add a concluding summary. Use [] when no additional context is needed.

Length and formatting:
- All counts are maximums, not targets. Stop adding items when the important supported information is covered.
- State each observation once. Do not repeat entities, attributes, relations, or sentences to fill the available space.
- Return compact JSON on a single line, with no indentation, line breaks, or spaces outside string values. Keep normal spaces inside strings.
- Empty arrays are valid. Close every array and the JSON object, then stop. Return no Markdown or commentary.

[Expected Output Format]
This example illustrates the format. Replace its contents with observations from the supplied frames.

{"entities":[{"id":"person1","name":"person","attributes":["red jacket","smiling"]},{"id":"person2","name":"person","attributes":["blue shirt","raising a hand"]},{"id":"cup1","name":"cup","attributes":["white"]},{"id":"cup2","name":"cup","attributes":["blue"]},{"id":"bird1","name":"bird","attributes":["preening"]}],"relations":[{"subject_id":"person1","predicate":"holding","object_id":"cup1"},{"subject_id":"person2","predicate":"holding","object_id":"cup2"}],"context":["The segment switches between unrelated shots."]}