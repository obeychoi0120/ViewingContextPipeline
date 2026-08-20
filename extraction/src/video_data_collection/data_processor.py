import json
import os
import difflib
from .utils import deduplicate_texts, clean_text

def intervalize_ocr(ocr_frames_file, output_file="ocr_cleaned.json", max_gap=2.0, similarity_threshold=0.75):
    print(f"[INFO] Intervalizing OCR Data: {os.path.basename(ocr_frames_file)}")
    
    if not os.path.exists(ocr_frames_file):
        print(f"[ERROR] Required OCR file not found: {ocr_frames_file}")
        return
        
    with open(ocr_frames_file, 'r', encoding='utf-8') as f:
        ocr_frames = json.load(f)
        
    final_intervals = []
    active_tracks = [] # list of dict: start_time, end_time, last_seen_time, text, clean_text
    
    for f in ocr_frames:
        t_time = f["frame_time"]
        t_texts = f.get("texts", [])
        
        # 1. Expire tracks that haven't been seen within max_gap
        still_active = []
        for track in active_tracks:
            if t_time - track["last_seen_time"] > max_gap:
                final_intervals.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active
        
        # 2. Match current texts to active tracks
        for raw_text in t_texts:
            c_text = clean_text(raw_text)
            if len(c_text) < 1:
                continue
                
            best_match_idx = -1
            best_ratio = -1.0
            
            for i, track in enumerate(active_tracks):
                ratio = difflib.SequenceMatcher(None, c_text, track["clean_text"]).ratio()
                # Subset matching (if one is a substring of another)
                if c_text in track["clean_text"] or track["clean_text"] in c_text:
                    ratio = max(ratio, 0.9)
                    
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_idx = i
                    
            if best_match_idx != -1 and best_ratio >= similarity_threshold:
                # Update matched track
                matched_track = active_tracks[best_match_idx]
                matched_track["last_seen_time"] = t_time
                matched_track["end_time"] = t_time
                # Keep the longer/richer text
                if len(raw_text) > len(matched_track["text"]):
                    matched_track["text"] = raw_text
                    matched_track["clean_text"] = c_text
            else:
                # Spawn a new track
                active_tracks.append({
                    "start_time": t_time,
                    "end_time": t_time,
                    "last_seen_time": t_time,
                    "text": raw_text,
                    "clean_text": c_text
                })
                
    # 3. Expire any remaining tracks when video ends
    for track in active_tracks:
        final_intervals.append(track)
        
    # 4. Format intervals for output to match old schema
    output_data = []
    # Sort by start_time to keep it chronological
    final_intervals.sort(key=lambda x: x["start_time"])
    
    for tr in final_intervals:
        output_data.append({
            "start_time": tr["start_time"],
            "end_time": tr["end_time"],
            "texts": [tr["text"]] # Put inside list to maintain compatibility with downstream
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"[SUCCESS] Intervalized OCR saved to {output_file} (Total Lines Tracked: {len(output_data)})")

def merge_scene_data(timestamp_file, asr_words_file, ocr_cleaned_file, output_file="final_dataset.json", video_duration=None, ocr_max_length=None):
    print(f"[INFO] Merging Granular Data by Scenes into: {os.path.basename(output_file)}")

    try:
        with open(timestamp_file, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
        
        # Load granular data, default to empty list if not exists
        asr_words = []
        if os.path.exists(asr_words_file):
            with open(asr_words_file, 'r', encoding='utf-8') as f:
                asr_words = json.load(f)
                
        ocr_cleaned = []
        if os.path.exists(ocr_cleaned_file):
            with open(ocr_cleaned_file, 'r', encoding='utf-8') as f:
                ocr_cleaned = json.load(f)
                
    except FileNotFoundError as e:
        print(f"[ERROR] Required file not found: {e}")
        return

    # video_duration이 마지막 SPLIT 후보와 겹칠 때 생기는 duration=0 유령 씬 제거
    scenes = [s for s in scenes if s.get("scene_start", 0) != s.get("scene_end", -1)]
    final_list = []
    
    for i, scene in enumerate(scenes):
        start_time = scene.get("scene_start", 0)
        end_time = scene.get("scene_end", 0)
        if end_time == 0:
            if i + 1 < len(scenes):
                end_time = scenes[i+1]["scene_start"]
            else:
                # 마지막 씬의 경우 비디오 총 길이를 사용, 없으면 999999.0
                end_time = video_duration if video_duration is not None else 999999.0
        
        shot_boundaries = scene.get("shot_change_timestamps", [])
        boundaries = sorted(list(set([start_time] + shot_boundaries + [end_time])))
        raw_keyframe_timestamps = scene.get("keyframe_timestamps")
        if not isinstance(raw_keyframe_timestamps, list):
            raise ValueError(f"scene {i} keyframe_timestamps must be a list")
        try:
            keyframe_timestamps = [int(round(float(timestamp))) for timestamp in raw_keyframe_timestamps]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scene {i} has an invalid keyframe timestamp") from exc
        if len(keyframe_timestamps) != len(boundaries) - 1:
            raise ValueError(
                f"scene {i} has {len(boundaries) - 1} shot intervals but "
                f"{len(keyframe_timestamps)} keyframe timestamps"
            )
        if len(set(keyframe_timestamps)) != len(keyframe_timestamps):
            raise ValueError(f"scene {i} has duplicate keyframe timestamps")
        
        timeline = []
        for shot_idx in range(len(boundaries) - 1):
            shot_start = int(boundaries[shot_idx] * 10) / 10.0
            shot_end = int(boundaries[shot_idx+1] * 10) / 10.0
            
            # 1. Aggregate Speech for this shot (오버랩 기준 매칭)
            shot_words = []
            for w in asr_words:
                overlap_start = max(shot_start, w["start"])
                overlap_end = min(shot_end, w["end"])
                if overlap_start < overlap_end:
                    overlap_dur = overlap_end - overlap_start
                    word_dur = w["end"] - w["start"]
                    # 단어와 샷이 최소 0.05초 이상 겹치거나, 단어 전체 길이의 10% 이상 겹치는 경우 포함
                    if overlap_dur > 0.05 or (word_dur > 0 and (overlap_dur / word_dur) > 0.1):
                        shot_words.append(w["word"])
            speech_str = " ".join(shot_words) if shot_words else ""
            
            # 2. Aggregate OCR for this shot
            shot_ocr_texts = []
            for f in ocr_cleaned:
                # OCR timestamps represent 1 FPS samples. Treat a sample at t as
                # evidence for [t, t + 1) so exact boundaries are not duplicated.
                ocr_start = float(f["start_time"])
                ocr_end = float(f["end_time"]) + 1.0
                if max(shot_start, ocr_start) < min(shot_end, ocr_end):
                    for t in f["texts"]:
                        shot_ocr_texts.append(t)
                        
            unique_ocr = deduplicate_texts(shot_ocr_texts)
            ocr_str = ", ".join(unique_ocr) if unique_ocr else ""
            
            # OCR 텍스트 길이 제한 ( Head & Tail Sampling )
            if ocr_max_length and len(ocr_str) > ocr_max_length:
                head_list = []
                tail_list = []
                current_len = 0
                # snip_str를 위한 여유 공간 (약 20자)
                limit = ocr_max_length - 20
                
                h_ptr = 0
                t_ptr = len(unique_ocr) - 1
                
                while h_ptr <= t_ptr:
                    # Head 추가
                    h_text = unique_ocr[h_ptr]
                    if current_len + len(h_text) + 2 > limit:
                        break
                    head_list.append(h_text)
                    current_len += len(h_text) + 2
                    h_ptr += 1
                    
                    if h_ptr > t_ptr: break
                    
                    # Tail 추가
                    t_text = unique_ocr[t_ptr]
                    if current_len + len(t_text) + 2 > limit:
                        break
                    tail_list.insert(0, t_text)
                    current_len += len(t_text) + 2
                    t_ptr -= 1
                
                ocr_str = ", ".join(head_list) + " ... [snip] ... " + ", ".join(tail_list)

            # 3. 원문 유지 (Late Fragmentation을 위해)
            timeline.append({
                "shot_idx": shot_idx,
                "timestamp": keyframe_timestamps[shot_idx],
                "raw_asr": speech_str,
                "raw_ocr": ocr_str
            })

        final_list.append({
            "scene_idx": i,
            "timeline": timeline
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in final_list:
            json_record = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
            f.write(json_record + '\n')

    print(f"[SUCCESS] Merged dynamic scene dataset saved to {output_file}")
