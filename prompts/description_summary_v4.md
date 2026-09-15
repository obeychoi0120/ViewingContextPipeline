Summarize the supplied scene observations into one natural English paragraph describing the video. Include the most distinctive entities, attributes, actions, directed relationships, and background in order of importance. Merge repeated observations while preserving important differences and changes. Preserve grounded interpretations and their uncertainty already present in the observations; do not add facts, interpretations, or relationships absent from the input. Never reproduce, quote, or translate on-screen wording.

Aim for 100–200 words, with a hard maximum of 200 words. Fewer than 100 words are fine when evidence is sparse; do not pad. Output only the paragraph, without headings, lists, labels, or commentary. There are no field quotas. Entity IDs are scene-local references, not identities: describe distinguishable entities naturally using their appearance and preserve the direction of their interactions. Never assume equal IDs in different scenes refer to the same entity.

Scene observations:
{scenes}
