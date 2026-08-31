Using only the chronological scene descriptions below, produce a selective video-level summary of the visible content.

This is a synthesis task, not an exhaustive inventory of every described person, object, action, or relation.

Output rules:

- Output exactly seven physical lines in the required order.
- Put each value on the same physical line as its label.
- Every non-empty value must be one natural, complete English sentence.
- Start each non-empty value with a capital letter and end it with a period.
- Do not output keyword lists, tag lists, sentence fragments, or long comma-separated inventories.
- Prefer subject-verb sentences and conjunctions over comma-separated enumeration.
- Use at most 25 English words per field.
- Select only the most important distinct evidence needed to characterize the whole video.
- Merge repeated or equivalent observations into one statement.
- Mention a recurring action or relation once instead of reproducing every occurrence.
- Never create combinations by pairing every person with every object, action, or relation.
- Include a relation only when that specific relation is explicitly stated or directly visible in the input descriptions.
- Leave the value after the colon empty when no grounded evidence exists.
- Do not add facts absent from the descriptions.
- Do not infer identity, demographics, intent, story, genre, audience, private mental state, or audio information.
- Do not output JSON, Markdown, bullets, commentary, or additional fields.
- Stop immediately after the semantic_topics line.

Field guidance:

- setting_and_environments: Describe the primary visible settings and meaningful setting transitions as one sentence.
- main_characters_and_objects: Summarize the principal visible subject types and salient objects as one sentence; do not enumerate every occurrence.
- chronological_events: Connect the major visible actions in temporal order as one sentence, using words such as first, then, and finally only when supported.
- relations: Summarize only the most important distinct visible spatial, interaction, holding, wearing, or other explicit relations as one sentence.
- visual_atmosphere: Describe the overall visually evident atmosphere as one sentence using only described lighting, color, weather, setting, composition, and visible action tone.
- visible_affect: Describe explicitly visible expressions, postures, interactions, or action tone as one sentence without inferring private mental states.
- semantic_topics: Summarize the principal grounded topics as one sentence.

Required output shape:

setting_and_environments: <one complete sentence or empty>
main_characters_and_objects: <one complete sentence or empty>
chronological_events: <one complete sentence or empty>
relations: <one complete sentence or empty>
visual_atmosphere: <one complete sentence or empty>
visible_affect: <one complete sentence or empty>
semantic_topics: <one complete sentence or empty>

Chronological scene descriptions:

{scenes}
