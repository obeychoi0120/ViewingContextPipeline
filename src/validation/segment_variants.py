"""Ablation variants of the activity segments: the same hypergraph model over a different scene incidence.

Each variant starts from a video's default segments (validation.scene_segments.segments) and changes only
which scenes a hyperedge contains, separating the two conditions a default segment combines:

  action_bag   same activity, contiguity dropped: one hyperedge per activity over all scenes that contain it
  time_window  contiguity kept, activity dropped: each default segment keeps its length and moves to a
               uniformly drawn run of consecutive scene_idx in the same video
  random       both dropped: one random permutation of a video's scenes applied to its default segments,
               which keeps the segment count, segment sizes and per-scene membership counts

Randomness is drawn per video from default_rng([seed, catalog index]), so a variant is reproducible and
independent of catalog order. Variant sidecars sit beside the default one and never replace it.

  python -m validation prepare-module --run-id <RUN> --dataset <DATASET>
"""

from collections import Counter

import numpy as np

from validation.scene_segments import bundle_scenes, segments

VARIANTS = ("action_bag", "time_window", "random")


def action_bag(scenes, groups, rng=None):
    """[(activity, positions)] with every scene of an activity in one hyperedge."""
    return [(activity, [p for p, scene in enumerate(scenes) if activity in scene[1]])
            for activity in sorted({a for scene in scenes for a in scene[1]})]


def contiguous_starts(scenes, length):
    """Start positions of the runs of `length` scenes whose scene_idx increase by exactly 1."""
    return [start for start in range(len(scenes) - length + 1)
            if all(scenes[start + i + 1][0] == scenes[start + i][0] + 1 for i in range(length - 1))]


def time_window(scenes, groups, rng):
    """Each default segment moved to a random contiguous run of the same length (its own run is a candidate)."""
    out = []
    for activity, members in groups:
        starts = contiguous_starts(scenes, len(members))
        start = starts[int(rng.integers(len(starts)))]
        out.append((activity, list(range(start, start + len(members)))))
    return out


def random_permutation(scenes, groups, rng):
    """Default segments under one random permutation of the video's scene positions."""
    permutation = rng.permutation(len(scenes))
    return [(activity, sorted(int(permutation[p]) for p in members)) for activity, members in groups]


BUILDERS = {"action_bag": action_bag, "time_window": time_window, "random": random_permutation}


def variant_segments(scenes, variant, seed, video_index):
    """(default segments, variant segments) of one video, both as [(activity, positions)]."""
    groups = segments(scenes)
    rng = np.random.default_rng([seed, video_index])
    return groups, BUILDERS[variant](scenes, groups, rng)


def overlap(groups, variant_groups):
    """Per variant hyperedge: (identical to a default hyperedge, best Jaccard with a default hyperedge)."""
    defaults = [frozenset(members) for _, members in groups]
    out = []
    for _, members in variant_groups:
        members = frozenset(members)
        best = max((len(members & d) / len(members | d) for d in defaults), default=0.0)
        out.append((members in defaults, best))
    return out


def attach_variant(context, arm_name):
    """Build the Arm's segment variant and attach it beside the source bundle's default sidecar."""
    from arm_registry import feature_source, registry
    from artifact_io import read_json
    from validation.graph_inputs import write_scene_segments

    arm = registry(context.config)[arm_name]
    if "segments" not in arm.extensions or arm.segments is None:
        raise ValueError(f"Arm {arm_name} has no segment variant")
    variant, seed = arm.segments
    if variant not in VARIANTS:
        raise ValueError(f"unknown segment variant: {variant}")
    directory = context.representations_dir / feature_source(arm)
    manifest = read_json(directory / "manifest.json")
    excluded = {(e["content_id"], e["scene_idx"]) for e in read_json(directory / "statistics.json")["excluded_scenes"]}
    video_offsets = np.load(directory / "video_offsets.npy")
    scene_offsets = np.load(directory / "scene_offsets.npy")
    types = np.load(directory / "types.npy")
    pairs, lengths, matches, segment = [], Counter(), [], 0
    default_count = default_singletons = 0
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
        groups, changed = variant_segments(scenes, variant, seed, index)
        default_count += len(groups)
        default_singletons += sum(len(members) == 1 for _, members in groups)
        matches.extend(overlap(groups, changed))
        for _, members in changed:
            pairs.extend([first + p, segment] for p in members)
            lengths[len(members)] += 1
            segment += 1
    statistics = {
        "arm": arm_name, "source": feature_source(arm), "variant": variant, "seed": seed,
        "segments": segment, "memberships": len(pairs),
        "length_distribution": {str(k): v for k, v in sorted(lengths.items())[:15]},
        "singleton_fraction": lengths[1] / segment if segment else 0.0,
        "default_segments": default_count,
        "default_singleton_fraction": default_singletons / default_count if default_count else 0.0,
        # How far the variant moved from the default incidence; a value near 1 means the comparison is
        # close to a repeat of the default Arm and carries little information.
        "identical_to_default_fraction": sum(m for m, _ in matches) / len(matches) if matches else 0.0,
        "mean_best_jaccard_with_default": sum(j for _, j in matches) / len(matches) if matches else 0.0,
    }
    write_scene_segments(directory, np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
                         statistics=statistics, variant=arm.segments)
    return statistics
