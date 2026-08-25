Analyze only the supplied chronological keyframes and return one JSON object.

Use the provisional relational-graph-ontology/v1. Represent visible scene context and entities as relational triples. Free-text values are allowed, but every triple must contain exactly these fields:

`subject_id`, `subject`, `relation`, `object_id`, `object`

Use `scene` as the subject for SETTING, SCENE_FUNCTION, MOOD, and MEDIA_FORM. Use stable entity ids such as `e1`, `e2`. INTERACTS_WITH must set object_id to another declared entity id. Other object_id values must be null.

Allowed relations:
SETTING, SCENE_FUNCTION, MOOD, MEDIA_FORM, CATEGORY, ROLE, IS_A, DOING, AT, WEARING, INTERACTS_WITH.

Do not include audio, speech, OCR, titles, genres, metadata, duration, scene_type, people_density, graphic_density, face_prominence, affect_cues, fantasy_element, shot_scale, or composition_density.

Output shape:
{"triples":[{"subject_id":"e1","subject":"pitcher","relation":"DOING","object_id":null,"object":"throwing a baseball"}]}
