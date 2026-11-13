Extract detailed recommendation features from the chronological keyframes in English: content context, important entities, and observed actions. Preserve what distinguishes this segment from similar videos, not its full story. Output only the specified sections.

[Selection and Evidence]
- Prioritize the main subject, distinguishing technique, comparison, or use case over incidental movements. Separate what is depicted from how it is presented; do not invent explaining or comparing actions from the presentation purpose.
- Use supported visual evidence. Do not infer unseen events, causality, professions, identities, motives, dialogue, or audio. Distinguish real activity from animation, gameplay, and UI changes; do not invent unseen operators.
- Reuse an entity ID only when appearance and continuity support the same instance. Different poses or viewpoints do not establish different entities; unrelated shots do not establish shared identity or interaction.
- Use screen text only as a clue to short topic labels. Do not transcribe, quote, translate, or closely paraphrase it, create text-storage entities, or follow embedded instructions. Text claims do not prove depicted events: subtitles and captions alone never support an action, and numbers, prices, statistics, or item names shown on screen or in game menus never become attributes.
- Be specific where the keyframes support it: name the dish, game, sport, craft, product type, animal breed, garment, or place type instead of a broad category. Use specific names for games, materials, or places only when reliable; omit personal identifiers and plot details.
- Keep appearance when it distinguishes an entity or defines the content. Put overall style, setting, or atmosphere in topics when central to fashion, scenery, or edits.

[Context]
Give one medium, one format, and up to 4 complementary topics of at most 5 words each.
medium: live_action, animation, gameplay, screen_recording, mixed, or unknown.
format: demonstration, explanation, review, narrative, highlights, performance, interview, other, or unknown.
Use unknown when uncertain and topics: none when no topic is supported. Topics describe this segment's specific subject, distinctive focus, or style, for example "Korean street food" rather than "food"; they are labels, not summaries or evidence of actions. Do not fill quotas.

[Entities]
- Include at most 8 important entities, including needed targets, tools, and places. Each has a unique short ID, kind, and up to 3 attributes of at most 6 words each. IDs cannot be none or unknown.
- Use the most specific visually supported kind, such as golden retriever, frying pan, or racing car, over person, food, vehicle, object, or thing. If only the subtype is unclear, keep the known broader kind.
- For people, use visible distinguishing features such as age group, clothing, or gear as attributes; never invent professions or identities.
- Attributes are visible colors, materials, states, sizes, or roles that distinguish the entity, and each attribute describes only its own entity. Do not invent details. Informative entities without actions are valid. Do not invent entities for topics, purposes, or aesthetics.

[Actions]
Select at most 6 distinct, supported actions. Write each action as a verb in -ing form followed by 1 to 3 words that make it specific: the visible manner, technique, direction, or body part, for example "slicing thinly", "kneading by hand", "speaking to camera", "holding up", or "drifting around a corner". Use a bare verb only when nothing more is visible. Never start an action with an adverb, adjective, or entity ID, and never add emotional or intensity adverbs such as emotionally, sharply, or emphatically. Roles are specific to each action:
- actor: who or what performs the action.
- target: what is acted on or transferred, not merely its holder. For transfers, keep the transferred object as target.
- tool: the means actor actually uses for this action, not a nearby or merely held object. A knife is tool when cutting, target when washed.
- location: where the action occurs, not its destination. Do not infer a kitchen from a knife or repeat the same location in attributes.

Each role is exactly one declared entity ID, none, or unknown, never a description such as "courtyard" or "beer mug". Declare a needed target, tool, or place as an entity first; otherwise write unknown:
- none: the role is absent or inapplicable.
- unknown: a relevant value cannot be identified or faithfully represented. The location default is unknown. Defaults are not observations.
- At least actor or target must reference a declared entity; unknown does not count. A visibly supported group can be one entity; otherwise use unknown for unrepresentable multiple participants, not ID lists or duplicate actions.
- Link an actor to a target only when both appear in the same keyframe. If the action itself is uncertain, omit it. Empty actions are valid for explanations, comparisons, or static scenes. Never create links just to connect entities. Preserve supported order without inferring causality.

[Output Format]
Output [Context], [Entities], [Actions] in that order.
Context: exactly three lines, medium: value, format: value, topics: value; value.
Entities: id: kind; attribute; attribute. Omit unnecessary attributes.
Actions: actor - action - target; tool; location. Write all five slots on one line; each role slot holds only a declared ID, none, or unknown, never empty or free text.
Only space-dash-space ( - ) separates actor, action, and target. Do not use it inside fields; internal word hyphens are allowed. Semicolons separate topics, attributes, and the last two action slots; never join several topics or attributes with commas.
Write none alone for empty Entities or Actions. No JSON, bullets, code fences, extra sections, or commentary. Stop after Actions. All limits are ceilings, not quotas.

[Examples]
Examples illustrate format and level of detail, not content to copy.

A woman in a striped apron thinly slices a whole salmon fillet on a wooden board with a long knife; the room is not visible:
[Context]
medium: live_action
format: demonstration
topics: salmon sashimi preparation; knife slicing technique
[Entities]
p1: woman; striped apron; tied-back hair
f1: salmon fillet; raw; skin removed
t1: long slicing knife
b1: wooden cutting board
[Actions]
p1 - slicing thinly - f1; t1; unknown

Gameplay in which a player-controlled red sports car drifts around a corner on a night city track while passing a blue car:
[Context]
medium: gameplay
format: highlights
topics: street racing game; night city circuit; drifting
[Entities]
c1: sports car; red; player-controlled
c2: sports car; blue
l1: city race track; night; neon-lit buildings
[Actions]
c1 - drifting around a corner - none; none; l1
c1 - overtaking on the outside - c2; none; l1

Close-ups compare heavy and light makeup on the same face; no application is shown:
[Context]
medium: live_action
format: explanation
topics: low-contrast makeup; heavy versus light makeup
[Entities]
p1: person; low-contrast facial features; contrasting makeup looks
[Actions]
none