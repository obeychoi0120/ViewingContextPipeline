from moviepy import VideoFileClip

# MP4에서 16kHz 오디오 추출
def extract_audio(mp4_path, out_audio_path="temp_audio.wav"):
    print(f"[INFO] Extracting WAV from {mp4_path}...")
    video = VideoFileClip(mp4_path)
    # YAMNet 모델 학습 기준인 16kHz 샘플링 레이트로 추출합니다.
    video.audio.write_audiofile(out_audio_path, fps=16000, nbytes=2, buffersize=2000, logger=None)
    video.close()