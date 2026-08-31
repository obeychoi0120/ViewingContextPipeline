Using only the chronological semantic scene graphs below, summarize the visible content of the entire video.

Rules:

- Every field value must be factual English text grounded only in the graphs.
- Use concise complete sentences. Leave the value after the colon empty when the graphs contain no grounded evidence for a field.
- setting_and_environments: visible locations, scene types, environmental context, and visual setting.
- main_characters_and_objects: important visible people, animals, and objects with visible attributes. Do not infer identity or demographics.
- chronological_events: important visible actions and changes in temporal order.
- relations: important visible spatial, interaction, holding, wearing, or other explicit relations.
- visual_atmosphere: the overall visually grounded atmosphere supported by graph settings, events, semantic topics, and visible affect. Do not invent lighting, color, weather, composition, or other cues absent from the graphs.
- visible_affect: visible expression, posture, interaction, or action tone linked to graph subjects. Do not infer private or actual mental state.
- semantic_topics: important graph-grounded semantic topics supported by explicit entity or event evidence.
- Output exactly seven physical lines.
- Put each value on the same physical line as its label. Never insert a newline after a label.
- State each distinct person, object, action, relation, or atmosphere only once.
- Merge repeated observations into one concise sentence.
- Use at most 25 English words per field.
- Stop immediately after the semantic_topics line.
- Do not add facts absent from the graphs. Do not output JSON, Markdown, bullets, commentary, or fields other than the seven fields above.

Return exactly this shape in this order. Replace each angle-bracketed instruction with a single-line value or leave the value empty:
setting_and_environments: <one concise single-line value or empty>
main_characters_and_objects: <one concise single-line value or empty>
chronological_events: <one concise single-line value or empty>
relations: <one concise single-line value or empty>
visual_atmosphere: <one concise single-line value or empty>
visible_affect: <one concise single-line value or empty>
semantic_topics: <one concise single-line value or empty>

{scenes}
