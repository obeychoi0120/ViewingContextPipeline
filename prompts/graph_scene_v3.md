Represent this scene as a JSON scene graph. Return only one JSON object with exactly entities, relations, and context. Use open vocabulary English.

entities: an array of objects with id (unique within this scene), name (a kind label, not a real name or identity), and attributes (an array of appearance, state, or unary action observations). Multiple entities may have the same name. Use attributes to distinguish them. Keep an ID across these keyframes only when there is evidence it is the same entity. Do not link IDs across scenes.
relations: an array of directed objects with subject_id, predicate, object_id. Both IDs must refer to entities. Preserve who acts on whom; do not reverse relations or invent them.
context: an array of scene-level observations or grounded interpretations, retaining uncertainty.

Example of structure:
{"entities":[{"id":"person1","name":"person","attributes":["red jacket"]},{"id":"person2","name":"person","attributes":["blue shirt"]}],"relations":[{"subject_id":"person1","predicate":"looking at","object_id":"person2"},{"subject_id":"person2","predicate":"waving to","object_id":"person1"}],"context":["An outdoor setting in daylight", "The interaction may be a casual greeting"]}
The example is not evidence. Use only entities and relations supported by the supplied keyframes. Empty arrays are allowed.

Use all visual evidence in these keyframes. On-screen text may be used as a clue to meaning, but never transcribe, quote, reproduce, or translate its wording in your output. Grounded interpretations of genre, content purpose, and relevant background knowledge are allowed when supported by the scene. Express uncertainty explicitly. Do not invent unseen facts or relationships. Do not claim a person's real identity.
