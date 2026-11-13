Extract recommendation features from the chronological keyframes in English: context, important entities, and observed actions. Keep what tells this segment apart from other videos on the same subject. Output only the three sections below.

[Evidence]
- Write only what the frames show. Do not guess unseen events, dialogue, audio, motives, or identities.
- Screen text is only a clue for topics. Never copy on-screen text, usernames, scores, or numbers into the output.
- Appearances in different keyframes are the same entity only when appearance and continuity support it.

[Context]
One medium, one format, and up to 4 topics of at most 4 words each.

medium is how the footage was made:
- live_action: filmed real people, objects, or places.
- animation: drawn or computer-animated content that is not a game being played.
- gameplay: a video game being played.
- screen_recording: apps, websites, or interfaces other than games.
- mixed: clearly combines the above. unknown: cannot be determined.

format is how the content is presented:
- demonstration: a how-to with steps the viewer is meant to repeat, such as a recipe or tutorial.
- explanation: informs about a subject through visuals, diagrams, or comparisons.
- review: tries out or evaluates a product, food, place, or work.
- narrative: a story, skit, drama, or everyday moment.
- highlights: selected moments or a montage from play, sport, or events.
- performance: singing, dancing, playing music, or another staged act.
- interview: questions and answers or a conversation presented to the camera.
- other: none of the above fits. unknown: cannot be determined.

topics are specific labels for this segment's subject, focus, and style: tense family dinner rather than character interaction. Write topics: none when nothing is supported.

[Entities]
Up to 8 entities: the main participants and the targets and tools their actions need. Each is a name of 1 to 3 words, then up to 3 attributes of at most 6 words each.
- Use the most specific name the frames support: chef rather than person, sea urchin rather than seafood. Name people by a visible role such as soldier or street vendor; use man or woman only when no role shows.
- Names are unique. Number entities that would share a name: man1, man2. A visible group can be one entity, such as crowd.
- Attributes give distinguishing appearance, material, or state, such as sliced or assembled.
- Add a place only when the frames show a specific location, such as kitchen or rooftop. The interface of a game or app is not an entity.

[Actions]
Up to 6 actions in the order they occur, one per line: actor - action - target; tool; place.
- actor performs the action. target is who or what it is done to. tool is what the actor uses to do it. place is the specific location where it happens.
- Each slot is a name copied exactly from [Entities], or none, or unknown. none: the action has no such role. unknown: the role exists but cannot be identified; an unclear place is unknown.
- Write interactions: what someone does to an object or to another participant, with the most specific verb the frames support: filleting rather than cutting, confronting rather than looking at.
- Posture, expression, and undirected speech are attributes, not actions: man; arms crossed. Use target none only for motion that is itself the content, such as dancing or falling.
- If no action is supported, write none.

[Output Format]
The first line is [Context]. Then [Entities], then [Actions], each header on its own line. Write nothing after Actions.
Context is three lines: medium: value, format: value, topics: value; value.
Separate topics, attributes, and the tool and place slots with semicolons, never commas. Use space-dash-space ( - ) only between actor, action, and target. No colons or IDs in entity lines.
Every action line has all five slots and holds exactly one action. Write none alone when Entities or Actions is empty. No JSON, bullets, or commentary. Limits are ceilings, not quotas.

[Examples]
Examples illustrate format, not content to copy.

A person in a chef uniform cuts open a sea urchin with scissors, then scoops out the roe with a spoon in a visible kitchen:
[Context]
medium: live_action
format: demonstration
topics: seafood preparation; sea urchin roe extraction
[Entities]
chef; white uniform
sea urchin; spiny shell; cut open
roe; orange
scissors
spoon
kitchen
[Actions]
chef - cutting open - sea urchin; scissors; kitchen
chef - scooping - roe; spoon; kitchen

Two men are in an unclear place; one sands a wooden board with a sanding block while the other stands nearby:
[Context]
medium: live_action
format: narrative
topics: woodworking; hand sanding
[Entities]
man1; wearing gloves; gray beard
man2; wearing cap
wooden board; unfinished
sanding block
[Actions]
man1 - sanding - wooden board; sanding block; unknown

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
player character - shooting - enemy character; rifle; rooftop
enemy character - falling - none; none; rooftop

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
woman - confronting - man; none; unknown
woman - throwing - cushion; none; unknown

Close-ups compare heavy and light makeup on the same face; no application is shown:
[Context]
medium: live_action
format: explanation
topics: low-contrast makeup; heavy versus light makeup
[Entities]
person; low-contrast facial features; contrasting makeup looks
[Actions]
none
