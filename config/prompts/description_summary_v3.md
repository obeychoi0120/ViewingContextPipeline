Using only the chronological scene descriptions below, summarize the visible content of the entire video.

Rules:

- Every field value must be a factual English string grounded only in the descriptions.
- Use concise complete sentences. Use an empty string when the descriptions contain no grounded evidence for a field.
- setting_and_environments: visible locations, scene types, environmental context, and visual setting.
- main_characters_and_objects: important visible people, animals, and objects with visible attributes. Do not infer identity or demographics.
- chronological_events: important visible actions and changes in temporal order.
- relations: important visible spatial, interaction, holding, wearing, or other explicit relations.
- visual_atmosphere: the overall visually evident atmosphere grounded in described lighting, color, weather, setting, composition, and visible action tone. Do not infer genre, story, intent, private mental state, or audio mood.
- visible_affect: visible expression, posture, interaction, or action tone described for people or animals. Do not infer private or actual mental state.
- semantic_topics: important grounded topics supported by explicit people, objects, actions, or settings in the descriptions.
- Each value should be concise and factual.
- Do not add facts absent from the descriptions. Do not output Markdown, commentary, or fields other than the seven fields above.

Return exactly one JSON object with these seven fields in this order:
{{
  "setting_and_environments": "",
  "main_characters_and_objects": "",
  "chronological_events": "",
  "relations": "",
  "visual_atmosphere": "",
  "visible_affect": "",
  "semantic_topics": ""
}}

{scenes}
