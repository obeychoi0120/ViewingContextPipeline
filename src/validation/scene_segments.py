"""Activity segments over a graph bundle's scenes: the scene_segments.npy sidecar of the segments extension.

A scene's activities are the consistency keys of its action phrases (below). A segment is a maximal run of a
video's valid scenes whose extracted scene_idx increase by exactly 1 and that all contain one activity; a gap
(an invalid scene or a re-appearance later) starts a new segment, length-1 segments are kept, and a scene with
k activities belongs to k segments. The "shuffled" control keeps every segment's activity and size but moves
the membership through one fixed permutation of each video's scenes. An Arm with the segments extension
reads the variant named by Arm.segments from its source bundle.

  python -m validation.scene_segments --run-id <RUN> --dataset <DATASET> --target g3 g3s
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SHUFFLE_SEED = "20261027"
IRREGULAR = {"ran": "run", "sat": "sit", "ate": "eat", "stood": "stand", "held": "hold", "fell": "fall",
             "went": "go", "took": "take", "gave": "give", "made": "make", "threw": "throw", "drove": "drive",
             "flew": "fly", "fought": "fight", "caught": "catch", "shot": "shoot", "hit": "hit", "put": "put",
             "cut": "cut", "set": "set", "lay": "lie", "lying": "lie", "dying": "die", "tying": "tie"}
SKIP = {"is", "are", "was", "were", "be", "being", "been", "a", "an", "the", "to"}


def activity_key(text):
    """Consistency key of an action phrase, not an English lemma (make / making / makes -> "mak").

    Lowercase tokens, participant markers (p1, o2, ...) and leading copulas/articles dropped; the first token
    is the verb: irregular forms mapped, one suffix stripped (-ing, -ies, -es after s/x/z/ch/sh, -s, -ed),
    a final "e" removed and a final doubled consonant undoubled. Empty or "unknown" phrases give None.
    """
    tokens = [t for t in re.findall(r"[a-z]+\d*", str(text).lower()) if not re.fullmatch(r"[a-z]\d+", t)]
    tokens = [t for t in tokens if t not in SKIP]
    if not tokens or tokens[0] in ("unknown", "none", "nothing"):
        return None
    word = IRREGULAR.get(tokens[0], tokens[0])
    if word.endswith("ing") and len(word) > 4:
        word = word[:-3]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("es") and re.search(r"(s|x|z|ch|sh)es$", word):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    word = IRREGULAR.get(word, word)
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def segments(scenes):
    """[(activity, [positions])] for scenes = [(scene_idx, activity set, ...)] in scene order."""
    out = []
    for activity in sorted({a for scene in scenes for a in scene[1]}):
        run = []
        for position, scene in enumerate(scenes):
            if activity in scene[1] and run and scene[0] == scenes[run[-1]][0] + 1:
                run.append(position)
            else:
                if run:
                    out.append((activity, run))
                run = [position] if activity in scene[1] else []
        if run:
            out.append((activity, run))
    return out


def shuffled(scene_count, groups, content_id):
    """Same segments (activity, size), membership moved by one fixed permutation of the video's scenes."""
    order = list(range(scene_count))
    random.Random(f"{SHUFFLE_SEED}:{content_id}").shuffle(order)
    return [(activity, sorted(order[p] for p in members)) for activity, members in groups]


def bundle_scenes(path, excluded, content_id):
    """[(scene_idx, activity set, action count)] for the scenes graph_inputs kept, in bundle order."""
    from artifact_io import read_jsonl
    from extraction.scene_storage import recover_scene_graph

    if not path.is_file():
        return []
    out = []
    for row in sorted(read_jsonl(path), key=lambda r: r["scene_idx"]):
        if (content_id, row["scene_idx"]) in excluded:
            continue
        graph, _ = recover_scene_graph(row["scene_graph"], row["warning"])
        phrases = [action["action"] for action in graph["actions"]]
        out.append((row["scene_idx"], {k for k in map(activity_key, phrases) if k}, len(phrases)))
    return out


def attach(context, arm_name):
    """Build the Arm's scene_segments variant and attach it to its source bundle; returns the statistics."""
    from arm_registry import feature_source, registry
    from artifact_io import read_json
    from validation.graph_inputs import write_scene_segments

    arm = registry(context.config)[arm_name]
    if "segments" not in arm.extensions:
        raise ValueError(f"Arm {arm_name} has no segments extension")
    variant = arm.segments
    directory = context.representations_dir / feature_source(arm)
    manifest = read_json(directory / "manifest.json")
    excluded = {(e["content_id"], e["scene_idx"]) for e in read_json(directory / "statistics.json")["excluded_scenes"]}
    video_offsets = np.load(directory / "video_offsets.npy")
    scene_offsets = np.load(directory / "scene_offsets.npy")
    types = np.load(directory / "types.npy")
    pairs, lengths, segment, moved = [], Counter(), 0, Counter()
    for index, item in enumerate(manifest["catalog"]):
        cid = item["content_id"]
        scenes = bundle_scenes(context.scene_arm_dir(arm.scene_arm) / f"{cid}.jsonl", excluded, cid)
        first = int(video_offsets[index])
        if len(scenes) != video_offsets[index + 1] - first:
            raise ValueError(f"scene count mismatch with the bundle: {cid}")
        for offset, scene in enumerate(scenes):
            start, stop = scene_offsets[first + offset], scene_offsets[first + offset + 1]
            if int((types[start:stop] == 1).sum()) != scene[2]:
                raise ValueError(f"action count mismatch with the bundle: {cid} scene {scene[0]}")
        groups = segments(scenes)
        if variant == "shuffled":
            permuted = shuffled(len(scenes), groups, cid)
            moved["segments"] += len(groups)
            moved["changed"] += sum(a[1] != b[1] for a, b in zip(groups, permuted))
            groups = permuted
        for _, members in groups:
            pairs.extend([first + p, segment] for p in members)
            lengths[len(members)] += 1
            segment += 1
    statistics = {
        "variant": variant, "arm": arm_name, "source": feature_source(arm), "shuffle_seed": SHUFFLE_SEED,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "segments": segment, "memberships": len(pairs),
        "length_distribution": {str(k): v for k, v in sorted(lengths.items())[:15]},
        "shuffled_segments_with_changed_members": moved["changed"] / moved["segments"] if moved["segments"] else None,
    }
    write_scene_segments(directory, np.asarray(pairs, dtype=np.int64).reshape(-1, 2), statistics=statistics,
                         variant=variant)
    return statistics


def main(argv=None):
    from pipeline_runtime import DATASETS, RunContext

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--target", nargs="+", required=True, help="Arms with the segments extension")
    args = parser.parse_args(argv)
    context = RunContext.load(args.run_id, dataset=args.dataset)
    for arm in args.target:
        print(json.dumps(attach(context, arm), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
