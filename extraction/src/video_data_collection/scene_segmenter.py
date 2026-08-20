import os
import json
import math
from scenedetect import detect, AdaptiveDetector, ContentDetector


DEDUPE_WINDOW_SEC = 1.0


class SceneSegmenter:
    def __init__(self, min_scene_length=15.0, split_threshold=60.0, max_scene_length=80.0, max_shots_per_scene=5, min_shot_length=3.0,
                 video_duration=None):
        self.min_scene_length = min_scene_length
        self.split_threshold = split_threshold
        self.max_scene_length = max_scene_length
        self.max_shots_per_scene = max(1, int(max_shots_per_scene))
        self.min_shot_length = min_shot_length
        self.video_duration = video_duration

    def dedupe_candidates(self, candidates, source_map):
        sorted_candidates = sorted(candidates)
        if not sorted_candidates:
            return []

        deduped = []
        merged_source_map = {}
        group = [sorted_candidates[0]]

        def flush_group(group_items):
            chosen = next((candidate for candidate in group_items if "end" in source_map.get(candidate, set())), None)
            if chosen is None:
                chosen = next(
                    (candidate for candidate in group_items if "visual" in source_map.get(candidate, set())),
                    group_items[0],
                )
            merged_sources = set()
            for candidate in group_items:
                merged_sources.update(source_map.get(candidate, set()))
            deduped.append(chosen)
            merged_source_map[chosen] = merged_sources

        for candidate in sorted_candidates[1:]:
            if candidate - group[-1] < DEDUPE_WINDOW_SEC:
                group.append(candidate)
            else:
                flush_group(group)
                group = [candidate]
        flush_group(group)

        source_map.clear()
        source_map.update(merged_source_map)
        return deduped

    def build_scene_payload(self, scene_start, scene_end, shot_candidates):
        canonical_start = math.ceil(scene_start)
        canonical_end = math.ceil(scene_end)
        canonical_shots = sorted(
            {
                math.ceil(shot)
                for shot in shot_candidates
                if scene_start <= shot < scene_end
            }
        )
        canonical_shots = [
            timestamp
            for timestamp in canonical_shots
            if canonical_start <= timestamp < canonical_end
        ]
        if canonical_start < canonical_end and canonical_start not in canonical_shots:
            canonical_shots.insert(0, canonical_start)
        canonical_shots = canonical_shots[:self.max_shots_per_scene]

        return {
            "scene_start": canonical_start,
            "scene_end": canonical_end,
            "duration": canonical_end - canonical_start,
            "shot_change_timestamps": list(canonical_shots),
            "keyframe_timestamps": list(canonical_shots),
        }

    def filter_visual_shots(self, candidate_timestamps, source_map):
        raw_candidates = sorted(set(float(t) for t in candidate_timestamps))
        if not raw_candidates:
            return []

        filtered_visual = [raw_candidates[0]]
        source_map[raw_candidates[0]] = {"visual"}

        for t in raw_candidates[1:]:
            if t - filtered_visual[-1] >= self.min_shot_length:
                filtered_visual.append(t)
                source_map.setdefault(t, set()).add("visual")

        return filtered_visual

    def segment_scenes(self, candidate_timestamps):
        if not candidate_timestamps:
            return [], []

        print(f"[Info] Starting Scene Segmentation... (Raw Shot Candidates: {len(candidate_timestamps)})")

        source_map = {}
        visual_shots = self.filter_visual_shots(candidate_timestamps, source_map)
        all_candidates = set(visual_shots)

        if self.video_duration is not None:
            all_candidates.add(float(self.video_duration))
            source_map.setdefault(float(self.video_duration), set()).add("end")

        candidate_timestamps = self.dedupe_candidates(all_candidates, source_map)

        if not candidate_timestamps:
            return [], []

        print(f"[Info] Visual cut candidates: {len(candidate_timestamps)}")

        final_scenes = []
        candidates_analysis = []
        forced_split_count = 0
        visual_split_count = 0

        current_scene_start = candidate_timestamps[0]
        current_shots = [current_scene_start]
        final_end = candidate_timestamps[-1]

        for t in candidate_timestamps[1:]:
            if t <= current_scene_start:
                continue

            while t - current_scene_start > self.max_scene_length:
                forced_end = current_scene_start + self.max_scene_length
                forced_end = round(forced_end, 2)
                if forced_end <= current_scene_start:
                    break

                final_scenes.append(self.build_scene_payload(current_scene_start, forced_end, current_shots))
                candidates_analysis.append([round(forced_end, 2), "forced", "SPLIT (Max Length)"])
                print(f"  [Forced Cut at {forced_end:05.1f}s] Scene exceeded max length -> SPLIT")

                forced_split_count += 1
                current_scene_start = forced_end
                current_shots = [forced_end]

            if t <= current_scene_start:
                continue

            sources = source_map.get(t, {"unknown"})
            source_str = ", ".join(sorted(sources))
            is_visual = "visual" in sources
            is_end = "end" in sources
            is_long_enough = (t - current_scene_start) >= self.min_scene_length
            passed_split_threshold = (t - current_scene_start) >= self.split_threshold
            would_exceed_shots = is_visual and not is_end and len(current_shots) >= self.max_shots_per_scene

            should_split = False
            split_kind = ""
            if would_exceed_shots:
                should_split = True
                split_kind = f"SPLIT (Max Shots: {self.max_shots_per_scene})"
                forced_split_count += 1
            elif is_visual and not is_end and passed_split_threshold and is_long_enough:
                should_split = True
                split_kind = f"SPLIT (Split Threshold: {self.split_threshold}s)"
                visual_split_count += 1

            if should_split:
                decision = split_kind
                final_scenes.append(self.build_scene_payload(current_scene_start, t, current_shots))
                current_scene_start = t
                current_shots = [t]
            else:
                if not is_long_enough:
                    decision = f"MERGE (Too Short, < {self.min_scene_length}s)"
                else:
                    decision = "MERGE (Same Scene)"

                if is_visual and not is_end and len(current_shots) < self.max_shots_per_scene:
                    current_shots.append(t)

            candidates_analysis.append([round(t, 2), source_str, decision])
            print(
                f"  [Cut at {t:05.1f}s] Sources: {source_str} | "
                f"Shots: {len(current_shots):02d}/{self.max_shots_per_scene} -> {decision}"
            )

        remaining_duration = final_end - current_scene_start
        if remaining_duration > 0.5:
            can_merge_tail = False
            if remaining_duration < self.min_scene_length and final_scenes:
                merged_start = final_scenes[-1]["scene_start"]
                merged_end = math.ceil(final_end)
                merged_shots = sorted(
                    {
                        math.ceil(shot)
                        for shot in final_scenes[-1]["shot_change_timestamps"] + current_shots
                        if merged_start <= math.ceil(shot) < merged_end
                    }
                )
                can_merge_tail = (
                    len(merged_shots) <= self.max_shots_per_scene
                    and merged_end - merged_start <= self.max_scene_length
                )

            if can_merge_tail:
                final_scenes[-1] = self.build_scene_payload(
                    final_scenes[-1]["scene_start"],
                    final_end,
                    merged_shots,
                )
            else:
                final_scenes.append(self.build_scene_payload(current_scene_start, final_end, current_shots))

        final_scenes = [
            scene
            for scene in final_scenes
            if scene["scene_start"] < scene["scene_end"] and scene["keyframe_timestamps"]
        ]

        print(f"\n[Info] Scene Segmentation Complete! Original: {len(candidate_timestamps)} candidate cuts -> {len(final_scenes)} Scenes.")
        print(f"{'='*100}")
        for idx, scene in enumerate(final_scenes):
            print(f"  [Scene #{idx}] Start Time: {scene['scene_start']} | End Time: {scene['scene_end']} | Duration: {scene['duration']} | Keyframes: {scene['keyframe_timestamps']}")
        print(f"{'='*100}")

        total_splits = forced_split_count + visual_split_count
        if total_splits > 0:
            visual_pct = (visual_split_count / total_splits) * 100
            forced_pct = (forced_split_count / total_splits) * 100
            print(f"[Info] Split Statistics:")
            print(f"       - SPLIT (Visual Shot)  : {visual_split_count} times ({visual_pct:.1f}%)")
            print(f"       - SPLIT (Forced)       : {forced_split_count} times ({forced_pct:.1f}%)")
        else:
            print(f"[Info] Split Statistics: No splits occurred.")

        return final_scenes, candidates_analysis

def extract_scene_timestamps(video_path: str, use_adaptive: bool = False, output_json: str = "timestamp.json") -> list:
    """
    비디오에서 Scene 전환 지점을 감지하여 각 Scene의 시작 시간을 초(second) 단위로 반환하고,
    결과를 JSON 파일로 저장합니다.

    Args:
        video_path (str): 분석할 비디오 파일의 경로
        use_adaptive (bool): True일 경우 AdaptiveDetector를 사용, False면 ContentDetector를 사용합니다.
        output_json (str): 결과를 저장할 JSON 파일 경로
        threshold (float): Scene 변화 감지 임계값. 숫자가 높을수록 더 보수적으로(큰 변화에만) 반응합니다. (기본값: Content=27.0, Adaptive=3.0)

    Returns:
        list: 각 씬의 시작 시점(timestamp) 리스트 (초 단위)
    """
    print(f"[Info] Extracting scene timestamps for: {video_path}")
    
    if use_adaptive:
        detector = AdaptiveDetector(adaptive_threshold=3.0)
    else:
        # 기존 기본값은 약 27.0이나, 잔잔한 조명 변화나 작은 모션을 무시하기 위해 높은 임계값(30.0) 적용
        detector = ContentDetector(threshold=30.0)

    # detect 함수는 각 씬의 (start_time, end_time) 튜플 리스트를 반환합니다.
    scene_list = detect(video_path, detector)
    
    if not scene_list:
        print("[Info] No scene cuts detected. Returning default timestamp 0.0.")
        timestamps = [0.0]
    else:
        timestamps = [scene[0].get_seconds() for scene in scene_list]
    
    # JSON 파일로 저장
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(timestamps, f, indent=4)
        print(f"[Info] Timestamps saved to {output_json}")
    except Exception as e:
        print(f"[Error] Failed to save JSON: {e}")

    print(f"[Info] Detected {len(timestamps)} scenes.")
    
    return timestamps

def run_scene_segmentation(candidate_timestamps, filtered_json,
                        min_scene_length=15.0, split_threshold=60.0, max_scene_length=80.0, max_shots_per_scene=5, min_shot_length=3.0,
                        video_duration=None):
    """
    Main.py 등 외부에서 호출할 수 있는 entry 함수입니다.
    """
    segmenter = SceneSegmenter(
        min_scene_length=min_scene_length,
        split_threshold=split_threshold,
        max_scene_length=max_scene_length,
        max_shots_per_scene=max_shots_per_scene,
        min_shot_length=min_shot_length,
        video_duration=video_duration,
    )
    
    filtered_timestamps, candidates_analysis = segmenter.segment_scenes(candidate_timestamps)
    
    # Save the filtered timestamps
    try:
        with open(filtered_json, 'w', encoding='utf-8') as f:
            json.dump(filtered_timestamps, f, indent=4)
        print(f"[Info] Filtered Scene Timestamps saved to {filtered_json}")
        
        # Save analysis file
        analysis_json = os.path.join(os.path.dirname(filtered_json), "timestamp_candidates_analysis.json")
        with open(analysis_json, 'w', encoding='utf-8') as f:
            json.dump(candidates_analysis, f, indent=4)
        print(f"[Info] Candidate Analysis saved to {analysis_json}")
        
    except Exception as e:
        print(f"[Error] Failed to save JSON: {e}")
        
    return filtered_timestamps
