Extract compact recommendation features from the chronological keyframes in English: context, important entities, and observed actions. Keep what distinguishes this segment. Output only the three sections below.

[Evidence]
- Write only what the frames show. Do not guess unseen events, dialogue, audio, or identities.
- Screen text is only a clue for topics. Never copy it into the output.

[Context]
One medium, one format, and up to 3 topics of at most 4 words each.
medium: live_action, animation, gameplay, screen_recording, mixed, or unknown.
format: demonstration, explanation, review, narrative, highlights, performance, interview, other, or unknown.
topics: short labels for this segment's subject, focus, or style. Write topics: none when nothing is supported.

[Entities]
Up to 6 entities: the main participants and the targets and tools their actions need. Each is a name of 1 to 3 words, then up to 2 attributes of at most 6 words each.
- Use specific names: chef rather than person, sea urchin rather than food. Name people by a visible role when one shows.
- Names are unique. Number two entities that would share a name: man1, man2. Write three or more of the same kind as one group entity, such as men, soldiers, or crowd.
- Add a place only when the frames show a specific location, such as kitchen.

[Actions]
Up to 4 actions, one per line: actor - action - target; tool: name; place: name.
- actor performs the action. target is who or what it is done to. tool is what the actor uses. place is the specific location.
- actor and target are names copied exactly from [Entities], or unknown. Write none when there is no target.
- Add tool: and then place: only when the frames show them, each with a name copied exactly from [Entities]. Otherwise leave the slot out.
- Write what someone does to an object or to another participant. Posture and expression are attributes, not actions.
- The action is a verb phrase without entity names: woman - pointing at - dish.
- Write each action once; every line differs from all earlier lines. Write a shared action such as talking once, not once per participant.
- Stop after the last distinct action. Fewer actions than the limit is correct. If no action is supported, write none.

[Output Format]
Use exactly this layout. The first line is [Context]. Stop after Actions.

[Context]
medium: value
format: value
topics: value; value
[Entities]
name; attribute; attribute
[Actions]
actor - action - target; tool: name; place: name

Start a new line after every section header, every context field, every entity, and every action. Never put two fields, two entities, or two actions on one line.
Separate topics, attributes, and the tool and place slots with semicolons. Use space-dash-space ( - ) only between actor, action, and target. No colons or IDs in entity lines.
Write none alone when Entities or Actions is empty. No JSON, bullets, or commentary. Limits are ceilings, not quotas.

[Examples]
Examples illustrate format, not content to copy.

A person in a chef uniform cuts a sea urchin with a knife in a visible kitchen:
[Context]
medium: live_action
format: demonstration
topics: seafood preparation
[Entities]
chef; white uniform
sea urchin
knife
kitchen
[Actions]
chef - cutting - sea urchin; tool: knife; place: kitchen

Two men are in an unclear place; one sands a wooden board with a sanding block, then the other carries the board away:
[Context]
medium: live_action
format: demonstration
topics: woodworking; hand sanding
[Entities]
man1; wearing gloves
man2; wearing cap
wooden board
sanding block
[Actions]
man1 - sanding - wooden board; tool: sanding block
man2 - carrying - wooden board

Close-ups compare heavy and light makeup on the same face; no application is shown:
[Context]
medium: live_action
format: explanation
topics: low-contrast makeup; heavy versus light makeup
[Entities]
person; low-contrast facial features; contrasting makeup looks
[Actions]
none

[Additional Instructions]
The instructions below add to everything above. Where they differ from it, follow these.

Limits: up to 8 entities with up to 3 attributes each, up to 6 actions, and up to 4 topics. These replace the limits above.

Context:
- medium is how the footage was made. live_action: filmed real people, objects, or places. animation: drawn or computer-animated content that is not a game being played. gameplay: a video game being played. screen_recording: apps, websites, or interfaces other than games. mixed: clearly combines these.
- format is how the content is presented. demonstration: shows how to do a task or technique, as in a recipe, tutorial, or craft work. explanation: informs about a subject through visuals, diagrams, or comparisons. review: tries out or evaluates a product, food, place, or work. narrative: a story, skit, drama, or everyday moment. highlights: selected moments or a montage from play, sport, or events. performance: singing, dancing, playing music, or another staged act. interview: questions and answers or a conversation presented to the camera.
- Name the specific situation in topics: tense family dinner rather than character interaction.

Entities:
- Appearances in different keyframes are the same entity only when appearance and continuity support it.
- Never put usernames, scores, or other numbers in names or attributes. The interface of a game or app is not an entity.
- When an action visibly changes an entity, give the resulting state as an attribute, such as sliced or assembled.

Actions:
- Write actions in the order they occur, with the most specific verb the frames support: filleting rather than cutting, confronting rather than looking at.
- Speech with no visible target is not an action. Use target none only for motion that is itself the content, such as dancing or falling.

Output:
- Never join topics or attributes with commas; each is its own semicolon-separated value.

Additional examples:

Third-person shooter footage with a visible HUD; the player character fires a rifle at an enemy character on a rooftop, and the enemy character falls:
[Context]
medium: gameplay
format: highlights
topics: third-person shooter; rooftop firefight
[Entities]
player character; orange vest
enemy character; armored
rifle
rooftop
[Actions]
player character - shooting - enemy character; tool: rifle; place: rooftop
enemy character - falling - none; place: rooftop

A woman and a man argue face to face indoors, the man standing with his arms crossed; the woman then throws a cushion at the man:
[Context]
medium: live_action
format: narrative
topics: couple argument; tense confrontation
[Entities]
woman; tearful expression
man; arms crossed
cushion
[Actions]
woman - confronting - man
woman - throwing - cushion
