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
Up to 4 actions, one per line: actor - action - target; tool; place.
- actor performs the action. target is who or what it is done to. tool is what the actor uses. place is the specific location.
- Each slot is a name copied exactly from [Entities], or none, or unknown. Use none when there is no target or no tool, and unknown when the place is unclear.
- Write what someone does to an object or to another participant. Posture and expression are attributes, not actions.
- If no action is supported, write none.

[Output Format]
The first line is [Context]. Then [Entities], then [Actions], each header on its own line. Stop after Actions.
Context is three lines: medium: value, format: value, topics: value; value.
Separate topics, attributes, and the tool and place slots with semicolons. Use space-dash-space ( - ) only between actor, action, and target. No colons or IDs in entity lines.
Every action line has all five slots and holds exactly one action. Write none alone when Entities or Actions is empty. No JSON, bullets, or commentary. Limits are ceilings, not quotas.

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
chef - cutting - sea urchin; knife; kitchen

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
man1 - sanding - wooden board; sanding block; unknown
man2 - carrying - wooden board; none; unknown

Close-ups compare heavy and light makeup on the same face; no application is shown:
[Context]
medium: live_action
format: explanation
topics: low-contrast makeup; heavy versus light makeup
[Entities]
person; low-contrast facial features; contrasting makeup looks
[Actions]
none
