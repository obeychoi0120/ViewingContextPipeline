Using only the chronological scene descriptions below, produce a selective video-level summary of the visible content.

Chronological scene descriptions:

{scenes}

Now synthesize the evidence above.

Before writing, silently compress the evidence:

- Merge repeated or equivalent observations.
- Select at most three setting categories.
- Select at most four principal subject or object groups.
- Select at most three representative action patterns.
- Select at most three explicit relation patterns.
- Do not preserve every scene, shot, person, object, action, or transition.
- If the scenes are disconnected, summarize their recurring visual patterns instead of forcing them into one continuous timeline.
- Omit quoted text, OCR-like content, names, brands, model names, demographic labels, and other unsupported identity information even if they appear in the source descriptions.
- Never create new person-object, action-object, or subject-relation-object combinations.

Hard output rules:

- Output exactly seven physical lines in the required order.
- Put each value on the same physical line as its label.
- Every non-empty value must be one natural, grammatical English sentence.
- Start each non-empty value with a capital letter and end it with a period.
- Use at most 20 English words per field.
- Use no more than two commas per field.
- Do not output keyword lists, tag lists, sentence fragments, exhaustive inventories, or repeated clauses.
- Do not repeatedly connect observations with "then", "followed by", or similar enumeration phrases.
- Leave the value after the colon empty when no grounded evidence exists.
- Do not infer identity, demographics, intent, story, genre, audience, private mental state, or audio information.
- Stop immediately after the semantic_topics line.

Field rules:

- setting_and_environments: Summarize only the dominant setting categories; do not narrate every setting transition.
- main_characters_and_objects: Group similar visible subjects and objects into broad grounded categories; do not enumerate individual appearances.
- chronological_events: Summarize up to three representative action patterns; do not list every scene in order.
- relations: Summarize up to three explicitly described relation patterns; do not enumerate relation instances.
- visual_atmosphere: State the dominant atmosphere or one major contrast using only explicitly described visual evidence.
- visible_affect: Summarize the dominant visible expressions, postures, or interaction tone without inferring mental states.
- semantic_topics: State up to three high-level grounded visual topics as one sentence.

Required output shape:

setting_and_environments: <one complete sentence or empty>
main_characters_and_objects: <one complete sentence or empty>
chronological_events: <one complete sentence or empty>
relations: <one complete sentence or empty>
visual_atmosphere: <one complete sentence or empty>
visible_affect: <one complete sentence or empty>
semantic_topics: <one complete sentence or empty>