Represent the supplied video segment as a JSON scene graph using open vocabulary English. Select distinctive, informative content. Return only one JSON object with exactly entities, relations, and context.

Evidence rules:
- These keyframes are chronological samples and may include repeated views, cuts, or unrelated shots. Consider all supplied frames and preserve important content from distinct shots. Describe changes supported by the frames; do not invent events between them or assume continuity of people, places, or events across cuts.
- Use on-screen text internally to disambiguate visible content and interpret meaning. Express the resulting understanding in your own words, without transcribing, quoting, reproducing, or translating the wording of captions, subtitles, signs, or watermarks. Do not produce a line-by-line paraphrase of the text. Distinguish what is shown from what text merely mentions, compares, or claims: a comparison with another dish does not identify that dish as the visible food.
- Grounded interpretations of genre, content purpose, and relevant background knowledge are allowed. Tie each interpretation to evidence and retain uncertainty. Do not expand a brief clue into unsupported events, personal histories, motives, or relationships. Do not claim a person's real identity.
- Distinguish individual people and objects. Merge repeated views when continuity and appearance support the same entity; a change of camera angle or illustration style alone does not imply a new entity. Similar appearance alone does not establish identity across cuts. When identity or object kind is unclear, preserve the visible features without forcing a specific identification.

entities:
- An array of objects with exactly id, name, and attributes. id is a unique string within this graph. name is a kind label, not a real name or identity. attributes is an array of strings describing appearance, state, or actions without an identified target, such as smiling or raising a hand.
- Give separate instances separate IDs even when they have the same name, such as two people holding two separate cups. Repeated views of the same entity share one ID when supported. Do not link IDs across separately processed scenes.
- Include important objects, action targets, and tools, not only people. Use specific kinds when supported; retain visible features when the kind is uncertain. Do not create entities solely to store on-screen wording.

relations:
- An array of objects with exactly subject_id, predicate, and object_id, all strings. Both IDs must exactly match IDs in entities. A reference is an ID, not a new object name or a descriptive sentence.
- Each specific subject-predicate-object connection must be supported by the frames. Preserve who acts on whom. Do not connect unrelated shots or substitute another person for an unregistered object.
- Put an action without an identified target in attributes. Never invent a target or a self-relation merely to complete a triple. Entities need not participate in a relation; disconnected parts and an empty relations array are valid.

context:
- An array of strings for additional setting, presentation, or grounded interpretation, retaining uncertainty. Use it for information not already captured by entities and relations, including a shift between unrelated shots when relevant.
- Merge equivalent observations. Do not repeat attributes, relations, or context entries, or add generic narrative conclusions. The on-screen wording restriction applies to every field.

Illustrative example: a person in a red jacket smiles while holding a white cup; another person in a blue shirt holds a blue cup and raises a free hand. An unrelated later shot shows a bird preening.

{"entities":[{"id":"person1","name":"person","attributes":["red jacket","smiling"]},{"id":"person2","name":"person","attributes":["blue shirt","raising a hand"]},{"id":"cup1","name":"cup","attributes":["white"]},{"id":"cup2","name":"cup","attributes":["blue"]},{"id":"bird1","name":"bird","attributes":["preening"]}],"relations":[{"subject_id":"person1","predicate":"holding","object_id":"cup1"},{"subject_id":"person2","predicate":"holding","object_id":"cup2"}],"context":["The segment switches between unrelated shots."]}

Before returning, check that every relation ID exists, its subject and target are correct, and equivalent observations appear only once. Empty arrays are allowed. Return the JSON object without Markdown or commentary, then stop.
