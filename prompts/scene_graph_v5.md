Extract compact recommendation features from the supplied chronological keyframes in English: content context, important entities, and observed actions. Preserve what distinguishes the activity, not a reconstruction of the original story. Output only the specified sections; no explanation or reasoning.

[Selection and Evidence]
- Prioritize the main activity, its targets and tools, then roles and states. Keep appearance only when needed to distinguish entities. Do not fill quotas or repeat equivalent observations.
- The frames may repeat an individual, change viewpoints, or cut to unrelated shots. Reuse an ID only when appearance and continuity support the same instance. A new pose, camera angle, cup contents, or avatar setting does not by itself create another entity. Never connect unrelated shots or assume unseen continuity.
- Distinguish real-world activity from animation, gameplay, UI operations, portraits, and messages. Menu icons are not people standing together; upgrades are changes to an item, not interactions among duplicate characters. Do not invent an unseen operator.
- Use on-screen text only to disambiguate a broad topic or activity. Do not quote, translate, transcribe, or paraphrase wording line by line. Do not create entities to store text. Omit real identities, dialogue, lyrics, detailed plot, inferred motives, histories, and unsupported relationships. Text claims are not observed events.

[Context]
Give one medium, one format, and up to 3 general topics, each at most 4 English words. Context describes this segment, not unseen parts of the video. Do not use a topic to invent actions.
medium: live_action, animation, gameplay, screen_recording, mixed, or unknown.
format: demonstration, explanation, review, narrative, highlights, performance, interview, other, or unknown.
Use unknown when uncertain; topics may be none. Topics are short categories, not sentences, names, or plot summaries.

[Entities]
- Select at most 6 entities, including the actual action targets and necessary tools. Use a unique short ID and a short kind label. Add at most 2 attributes per entity, each at most 6 words. Roles, states, important location, and untargeted movements may be attributes.
- Isolated entities are valid. Put static positions in attributes when important.

[Actions]
- Select at most 4 distinct observed actions. Identify the actor, action, actual target, and tool. A person holding an object is not a substitute for that object: cutting seafood targets seafood, not its holder. Preserve who acts on whom; prefer active wording.
- Every nonempty actor, target, or tool must be a declared entity ID. Write none when a reference is unknown or unnecessary; do not invent endpoints. At least the actor or target must be known. Tool alone is insufficient.
- Empty actions are valid. Do not connect entities merely to form a group, chain, or cycle. Order actions chronologically only when the frames support that order; do not infer causality or missing events.

[Output Format]
The three required sections are [Context], [Entities], [Actions], in exactly that order. An optional [End] marker may follow [Actions].
Context has exactly three lines: medium: value, format: value, topics: value; value.
Entities use: id: kind; attribute; attribute. Omit unnecessary attributes.
Actions use: actor - action - target; tool. Always include the tool slot, using none if absent.
Only space-dash-space ( - ) separates the three action fields. Hyphens inside words or IDs are allowed. Do not put a separator inside a field. Semicolons separate topics, attributes, or the tool slot.
Write none alone for an empty Entities or Actions section, and topics: none for no topic. No JSON, bullets, code fences, or additional sections. Stop after the actions or the optional [End] marker.

[Examples]
These illustrate format and evidence handling, not content to copy.

A visible person cuts a sea urchin with a knife:
[Context]
medium: live_action
format: demonstration
topics: seafood preparation
[Entities]
p1: person
f1: sea urchin
t1: knife
[Actions]
p1 - cutting - f1; t1
[End]

A game UI shows the same item's upgrade; no operator is visible:
[Context]
medium: gameplay
format: demonstration
topics: equipment upgrade
[Entities]
item1: game item; being upgraded
[Actions]
none - upgrading - item1; none
[End]

Several close-ups show the same animated character without an observed action:
[Context]
medium: animation
format: unknown
topics: character portrait
[Entities]
p1: character; black hair; red eyes
[Actions]
none
[End]
