import torch
from faster_whisper import WhisperModel
import json
import warnings

warnings.filterwarnings("ignore")

def process_asr(audio_wav_path, output_json="ASR_words.json", model_name="small", download_root=None, lang=None,
                beam_size=5, condition_on_previous_text=False, no_speech_threshold=0.8,
                vad_threshold=0.25, vad_min_silence_duration_ms=1000, vad_speech_pad_ms=300,
                initial_prompt=None):
    """
    model_name: 'small', 'medium', 'large-v3' 등의 사이즈 명칭 혹은 로컬 경로
    download_root: 모델이 다운로드되어 저장될 로컬 경로 (예: /home_nvme/shared/models/)
    lang: 고정할 언어 코드 (예: 'ko', 'en'). None이면 자동 감지
    """
    print(f"[INFO] Initializing Faster-Whisper ASR with model: {model_name}")
    
    # GPU가 있으면 FP16, 없으면 CPU + INT8(양자화). 양자화 옵션이 기본 제공되어 Edge/CPU 환경에 매우 유리함
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    asr_model = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=download_root)
    
    # 1. 언어 감지 및 이에 따른 최적화된 initial_prompt 설정
    detected_lang = lang
    if detected_lang is None:
        print(f"[INFO] Auto-detecting language of the audio...")
        # VAD 없이 첫 30초 블록을 인코딩하여 언어를 감지 (generator를 순회하지 않아 매우 빠름)
        _, detect_info = asr_model.transcribe(audio_wav_path)
        detected_lang = detect_info.language
        print(f"[INFO] Detected language: '{detected_lang}' (probability: {detect_info.language_probability:.2f})")
    
    if initial_prompt:
        prompt = initial_prompt
    elif detected_lang == "ko":
        prompt = (
            "마침표, 쉼표, 물음표 등 구두점을 자연스럽게 포함해 주세요. "
            "고유명사, 숫자, 장소명, 인물명은 들리는 대로 정확히 적어 주세요."
        )
    elif detected_lang == "en":
        prompt = (
            "Please include punctuation marks naturally. "
            "Preserve proper nouns, numbers, place names, and person names as accurately as heard."
        )
    else:
        prompt = (
            "Please include punctuation marks naturally. "
            "마침표, 쉼표 등 구두점을 자연스럽게 포함해 주세요."
        )
        
    print(f"[INFO] Extracting word-level timestamps with VAD (Language: '{detected_lang}'). This may take a while...")
    
    segments, info = asr_model.transcribe(
        audio_wav_path,
        language=detected_lang,
        word_timestamps=True,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=prompt,
        no_speech_threshold=no_speech_threshold,
        vad_filter=True,                    # [중요] 이 옵션이 True여야만 아래 vad_parameters가 작동합니다!
        vad_parameters=dict(
            threshold=vad_threshold,
            min_silence_duration_ms=vad_min_silence_duration_ms,
            speech_pad_ms=vad_speech_pad_ms,
        ),
    )
    
    
    last_text = ""
    last_end_time = -1.0
    all_words = []
    
    for segment in segments:
        for word in segment.words:
            text = word.word.strip()
            
            if not text:
                continue
                
            start_time = word.start
            end_time = word.end
            
            if text == last_text and start_time < last_end_time:
                last_end_time = max(last_end_time, end_time) 
                continue

            last_text = text
            last_end_time = end_time
            
            all_words.append({
                "start": round(start_time, 2), # 소수점 2자리 반올림으로 JSON 용량 및 VLM 파싱 토큰 최적화
                "end": round(end_time, 2),
                "word": text
            })
            
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_words, f, ensure_ascii=False, indent=4)
            
        print(f"[SUCCESS] ASR word-level results saved to {output_json}\n")

    except Exception as e:
        print(f"[ERROR] Failed to save ASR JSON file: {e}")
