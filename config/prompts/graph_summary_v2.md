Using only the chronological semantic scene graphs below, summarize the visible content of the entire video.

Return exactly one JSON object with these five fields in this order:
{{
  "setting_and_environments": "",
  "main_characters_and_objects": "",
  "chronological_events": "",
  "relations": "",
  "affect_or_topic": ""
}}

Rules:
- Every field value must be a factual English string grounded only in the graphs.
- Use concise complete sentences. Use an empty string when the graphs contain no grounded evidence for a field.
- setting_and_environments: visible locations, scene types, environmental context, and visual setting.
- main_characters_and_objects: important visible people, animals, and objects with visible attributes. Do not infer identity or demographics.
- chronological_events: important visible actions and changes in temporal order.
- relations: important visible spatial, interaction, holding, wearing, or other explicit relations.
- affect_or_topic: only explicitly visible affect or graph-grounded semantic topics. Do not infer mental state, intent, provenance, or category.
- Across all five values, use no more than 256 English words.
- Do not add facts absent from the graphs. Do not output Markdown, commentary, or fields other than the five fields above.

{scenes}
