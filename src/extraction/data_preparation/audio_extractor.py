# MP4에서 16kHz 오디오 추출
def extract_audio(mp4_path, out_audio_path="temp_audio.wav"):
    try:
        from moviepy import VideoFileClip
    except ImportError as exc:
        raise RuntimeError(
            "audio extraction requires the 'multimodal' optional dependencies"
        ) from exc

    print(f"[INFO] Extracting WAV from {mp4_path}...")
    video = VideoFileClip(mp4_path)
    # YAMNet 모델 학습 기준인 16kHz 샘플링 레이트로 추출합니다.
    video.audio.write_audiofile(out_audio_path, fps=16000, nbytes=2, buffersize=2000, logger=None)
    video.close()
