Describe the supplied video segment as a concise list of entities and relations in English.

[Evidence Rules]
- The frames are chronological samples and may contain repeated views, cuts, or unrelated shots. Consider all frames and select distinctive, important content.
- Describe actions and changes supported by the frames. Do not invent events between frames or assume continuity across unrelated shots.
- Use on-screen text to understand the visible content. Express the resulting meaning in your own words. Do not quote, transcribe, translate, or paraphrase captions line by line. Distinguish visible events from things that text merely mentions, compares, or claims.
- Do not invent real identities, motives, personal histories, or relationships.
- Distinguish individual people and objects. Merge repeated views only when appearance and continuity support the same instance. Similar appearance across unrelated shots does not establish identity.

[Content Rules]

Entities:
- Select at most 4 important entities, including action targets and tools when needed.
- Give each selected instance a unique ID, such as person1 or cup1. Reuse its ID across repeated views. Do not create a new entity for every frame.
- Use a short kind label, such as person, cup, or bird, rather than a real identity.
- Add at most 3 short attributes per entity, each at most 6 words.
- Attributes describe distinctive appearance, state, or actions without an identified target, such as smiling or raising a hand.
- Do not create entities merely to store on-screen wording.

Relations:
- Write at most 4 distinct relations between listed entities.
- Use the listed IDs exactly. Preserve the correct subject and target.
- Use a short action or relation phrase, such as holding, cutting, or standing beside.
- Put actions without an identified target in entity attributes. Do not invent a target or a self-relation.
- Entities may have no relations. Do not connect unrelated shots.

All counts are maximums, not targets. Use fewer items when sufficient. State each observation once, then stop.

[Output Format]
Use all three section headers below, in this order:
[Entities]
[Relations]
[End]

Under [Entities], write one entity per line:
id: kind; attribute; attribute

The first value after the colon is the kind. Subsequent semicolon-separated values are attributes. Omit attributes when unnecessary.

Under [Relations], write one relation per line:
subject_id -> relation -> object_id

Write none on its own line for an empty section.
Use semicolons and -> only as separators.
Do not add bullets, numbering, JSON, Markdown fences, or explanations.
Finish with [End] on its own line.

[Example]
This example illustrates the format. Replace its contents with observations from the supplied frames.

[Entities]
person1: person; red jacket; smiling
cup1: cup; white
person2: person; blue shirt; raising a hand
cup2: cup; blue

[Relations]
person1 -> holding -> cup1
person2 -> holding -> cup2

[End]
