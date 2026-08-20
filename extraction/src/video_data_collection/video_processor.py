import os
import ffmpeg
import json
import math
import re
import shutil
import time
from pathlib import Path
import cv2
import av
import yt_dlp
from tqdm import tqdm

from .bot_check import BOT_CHECK_MAX_RETRIES, BOT_CHECK_RETRY_DELAY_SEC, is_youtube_bot_check_error
from .ytdlp_utils import ytdlp_base_opts


FRAME_EXTRACTION_VERSION = 2
FRAME_EXTRACTION_METADATA_FILENAME = ".pts_extraction.json"
DIRECT_KEYFRAME_EXTRACTION_VERSION = 1


def download_video(
    url,
    output_filename="input_video.mp4",
    bot_check_max_retries=BOT_CHECK_MAX_RETRIES,
    bot_check_retry_delay_sec=BOT_CHECK_RETRY_DELAY_SEC,
):
    print(f"[Info] Downloading video from: {url}")
    
    # 확장자 제외한 파일명 추출
    name_base = os.path.splitext(output_filename)[0]

    ydl_opts = {
        **ytdlp_base_opts(),
        # 파일명 포맷 지정 (깔끔한 결과물을 위해)
        'outtmpl': f'{name_base}.%(ext)s',
        
        # 최종 병합 포맷
        'merge_output_format': 'mp4',
        
        # [핵심 수정] 이어받기 설정
        'overwrites': False,       # False로 설정해야 기존 파일을 덮어쓰지 않고 이어서 받음
        'continuedl': True,        # 이어받기 강제 활성화
        
        # [네트워크 보완] 연결 끊김 시 재시도 설정 (사내망/Proxy 환경에 도움)
        'retries': 10,             # HTTP 에러 시 10번 재시도
        'fragment_retries': 10,    # 조각 다운로드 실패 시 10번 재시도
        
        # 출력 설정
        'quiet': False,            # 진행률을 보기 위해 True -> False로 변경 추천
        'no_warnings': False,
        
        # SSL/Proxy 문제 해결
        'nocheckcertificate': True,
        'ignoreerrors': True,

        # 봇 감지 우회 (yt-dlp 내부 로직에 위임)

        # [핵심] 포맷 가용성 체크를 건너뛰고 일단 요청함
        'check_formats': False, 

        # 자막 다운로드 설정
        'write_subs': True,             # 제작자 자막 다운로드
        'write_auto_subs': True,        # 자동 생성 자막 허용
        'sub_langs': ['ko', 'en'],      # 한국어와 영어 자막 요청
        
        # (선택) 자막 포맷 변환 (vtt -> srt)
        'postprocessors': [{
            'key': 'FFmpegSubtitlesConvertor',
            'format': 'srt',
        }],

    }

    def extract_info():
        with yt_dlp.YoutubeDL({**ydl_opts, 'quiet': True, 'skip_download': True}) as ydl:
            return ydl.extract_info(url, download=False)

    info = run_ytdlp_with_bot_retry(
        extract_info,
        f"format probe for {url}",
        max_retries=bot_check_max_retries,
        retry_delay_sec=bot_check_retry_delay_sec,
    )
    selected_format = select_download_format(info)
    if not selected_format:
        print_format_diagnostics(info)
        raise FileNotFoundError(
            f"No downloadable video format at height >= 480 for {url}. "
            "Use yt-dlp -F to inspect available formats."
        )
    ydl_opts['format'] = selected_format
    print(f"[Info] Selected yt-dlp format: {selected_format}")

    def download_selected_format():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.download([url])

    run_ytdlp_with_bot_retry(
        download_selected_format,
        f"download for {url}",
        max_retries=bot_check_max_retries,
        retry_delay_sec=bot_check_retry_delay_sec,
    )
    
    # 다운로드 완료 확인
    # (병합된 최종 파일이 있는지 확인)
    final_file = f"{name_base}.mp4"
    if os.path.exists(final_file):
        print(f"[Success] Download complete: {final_file}")

    else:
        raise FileNotFoundError(
            f"Download did not create {final_file}. "
            "The video may not have an available stream at height >= 480."
        )


def run_ytdlp_with_bot_retry(
    operation,
    label,
    max_retries=BOT_CHECK_MAX_RETRIES,
    retry_delay_sec=BOT_CHECK_RETRY_DELAY_SEC,
):
    attempts = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if attempts >= max_retries or not is_youtube_bot_check_error(exc):
                raise
            attempts += 1
            print(
                f"[WARN] YouTube bot check while running {label}. "
                f"Retrying in {retry_delay_sec // 60} minutes "
                f"({attempts}/{max_retries})."
            )
            time.sleep(retry_delay_sec)


def select_download_format(info):
    formats = (info or {}).get('formats') or []
    video_candidates = [
        item for item in formats
        if has_video(item)
        and format_height(item) >= 480
    ]
    if not video_candidates:
        return ""

    video = sorted(video_candidates, key=video_sort_key)[0]
    video_id = str(video.get('format_id') or "")
    if not video_id:
        return ""
    if has_audio(video):
        return video_id

    audio_candidates = [item for item in formats if has_audio(item) and not has_video(item)]
    if not audio_candidates:
        return video_id
    audio = sorted(audio_candidates, key=audio_sort_key)[0]
    audio_id = str(audio.get('format_id') or "")
    return f"{video_id}+{audio_id}" if audio_id else video_id


def has_video(format_info):
    vcodec = format_info.get('vcodec')
    if vcodec and vcodec != 'none':
        return True
    return format_height(format_info) > 0 and not str(format_info.get('resolution') or "").startswith("audio")


def has_audio(format_info):
    return bool(format_info.get('acodec') and format_info.get('acodec') != 'none')


def format_height(format_info):
    height = int(format_info.get('height') or 0)
    if height:
        return height
    for key in ('resolution', 'format_note', 'format'):
        parsed_height = parse_height_text(str(format_info.get(key) or ""))
        if parsed_height:
            return parsed_height
    return 0


def parse_height_text(text):
    resolution_match = re.search(r'(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)', text)
    if resolution_match:
        return int(resolution_match.group(2))
    p_match = re.search(r'(?<!\d)(\d{3,5})p(?:\d+)?(?!\d)', text)
    if p_match:
        return int(p_match.group(1))
    return 0


def video_sort_key(format_info):
    ext = str(format_info.get('ext') or "")
    codec = str(format_info.get('vcodec') or "")
    return (
        format_height(format_info),
        0 if ext == 'mp4' else 1,
        0 if codec.startswith('avc1') else 1,
        float(format_info.get('tbr') or 0),
    )


def audio_sort_key(format_info):
    ext = str(format_info.get('ext') or "")
    return (
        0 if ext == 'm4a' else 1,
        -float(format_info.get('abr') or 0),
        -float(format_info.get('tbr') or 0),
    )


def print_format_diagnostics(info):
    formats = (info or {}).get('formats') or []
    preview = []
    for item in formats:
        preview.append(
            "{id} height={height} parsed={parsed} ext={ext} vcodec={vcodec} acodec={acodec} note={note} res={res}".format(
                id=item.get('format_id'),
                height=item.get('height'),
                parsed=format_height(item),
                ext=item.get('ext'),
                vcodec=item.get('vcodec'),
                acodec=item.get('acodec'),
                note=item.get('format_note'),
                res=item.get('resolution'),
            )
        )
    print(f"[WARN] yt-dlp returned {len(formats)} formats, but none matched height >= 480.")
    for line in preview[-20:]:
        print(f"[WARN] format: {line}")


def get_video_height(input_file):
    probe = ffmpeg.probe(input_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        return 0
    return int(video_stream.get('height') or 0)


def validate_480p_video(video_path):
    path = os.fspath(video_path)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise RuntimeError(f"480p video is missing or empty: {path}")
    height = get_video_height(path)
    if height != 480:
        raise RuntimeError(f"Expected 480p video, got height={height}: {path}")


def ensure_480p_video(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if output_path.exists():
        try:
            validate_480p_video(output_path)
        except Exception as exc:
            print(f"[WARN] Existing canonical video is invalid and will be repaired: {exc}")
        else:
            return

    if input_path.exists():
        source_path = input_path
    elif output_path.exists():
        source_path = output_path
    else:
        raise FileNotFoundError(f"Video source is missing: {input_path}")

    temporary_path = output_path.with_name(f".{output_path.stem}.480p_tmp{output_path.suffix}")
    temporary_path.unlink(missing_ok=True)
    try:
        height = get_video_height(source_path)
        if height == 480:
            print(f"[INFO] Source is already 480p. Copying {source_path} to {temporary_path}...")
            shutil.copy2(source_path, temporary_path)
        else:
            resize_to_480p(source_path, temporary_path)
        validate_480p_video(temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_contained_480p_video(input_file, output_file):
    """Create an 854x480 canonical copy without stretching caller-owned media."""

    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Video source is missing: {input_path}")
    if output_path.exists():
        try:
            validate_480p_video(output_path)
        except Exception as exc:
            print(f"[WARN] Existing canonical video is invalid and will be repaired: {exc}")
        else:
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.480p_tmp{output_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        probe = ffmpeg.probe(str(input_path))
        next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        audio_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "audio"
            ),
            None,
        )
        source = ffmpeg.input(str(input_path))
        video = (
            source.video.filter(
                "scale",
                854,
                480,
                force_original_aspect_ratio="decrease",
            )
            .filter("pad", 854, 480, "(ow-iw)/2", "(oh-ih)/2", color="black")
            .filter("setsar", 1)
        )
        output_args = {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "preset": "slower",
        }
        if audio_stream is not None:
            stream = ffmpeg.output(
                video,
                source.audio,
                str(temporary_path),
                acodec="copy",
                **output_args,
            )
        else:
            stream = ffmpeg.output(video, str(temporary_path), **output_args)
        stream.overwrite_output().run()
        validate_480p_video(temporary_path)
        os.replace(temporary_path, output_path)
    except (StopIteration, ffmpeg.Error) as exc:
        temporary_path.unlink(missing_ok=True)
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc, ffmpeg.Error) and exc.stderr
            else str(exc)
        )
        raise RuntimeError(
            f"Failed to create aspect-preserving 480p video: {stderr}"
        ) from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def resize_to_480p(input_file, output_file):
    input_file = os.fspath(input_file)
    output_file = os.fspath(output_file)
    try:
        # 1. 입력 파일 정보 확인
        probe = ffmpeg.probe(input_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        if video_stream is None:
            raise RuntimeError(f"Cannot find video stream: {input_file}")

        print(f"[INFO] Resizing {input_file} to 480p...")
       
        in_file = ffmpeg.input(input_file)
        v = in_file.video.filter('scale', 854, 480)
        
        if audio_stream is not None:
            a = in_file.audio
            stream = ffmpeg.output(
                v, a,
                output_file,
                vcodec='libx264',    # 가장 권장되는 CPU 코덱
                pix_fmt='yuv420p',   # 호환성이 가장 좋은 픽셀 포맷
                preset='slower',     # 느리게 인코딩할수록 압축률과 화질이 좋아짐 (ultrafast ~ veryslow)
                acodec='copy'        # 오디오는 재인코딩 없이 그대로 복사 (화질/음질 유지)
            )
        else:
            stream = ffmpeg.output(
                v,
                output_file,
                vcodec='libx264',
                pix_fmt='yuv420p',
                preset='slower'
            )
            
        stream = stream.overwrite_output()

        # 3. 실행
        stream.run()

    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else str(e)
        raise RuntimeError(f"Failed to resize video to 480p: {stderr}") from e

def frame_extraction_metadata_path(output_folder: str | Path) -> Path:
    return Path(output_folder) / FRAME_EXTRACTION_METADATA_FILENAME


def read_frame_extraction_metadata(output_folder: str | Path) -> dict | None:
    try:
        with frame_extraction_metadata_path(output_folder).open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def extracted_frames_are_current(video_path: str | Path, output_folder: str | Path) -> bool:
    metadata = read_frame_extraction_metadata(output_folder)
    if metadata is None or metadata.get("version") != FRAME_EXTRACTION_VERSION:
        return False

    try:
        source_stat = Path(video_path).stat()
        duration = int(metadata["duration_seconds"])
        frame_count = int(metadata["frame_count"])
    except (OSError, KeyError, TypeError, ValueError):
        return False

    if metadata.get("source_size") != source_stat.st_size:
        return False
    if metadata.get("source_mtime_ns") != source_stat.st_mtime_ns:
        return False
    if frame_count != duration + 1:
        return False

    output_path = Path(output_folder)
    png_files = list(output_path.glob("*.png"))
    return (
        len(png_files) == frame_count
        and (output_path / "0000.png").is_file()
        and (output_path / f"{duration:04d}.png").is_file()
    )


def get_extracted_video_duration(output_folder: str | Path) -> int:
    metadata = read_frame_extraction_metadata(output_folder)
    if metadata is None:
        raise RuntimeError(f"PTS frame extraction metadata is missing: {output_folder}")
    try:
        return int(metadata["duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"PTS frame extraction metadata is invalid: {output_folder}") from exc


def get_video_duration_seconds(video_path: str | Path) -> float:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video source is missing: {path}")
    try:
        probe = ffmpeg.probe(str(path))
        video_stream = next(
            stream for stream in probe["streams"] if stream.get("codec_type") == "video"
        )
        raw_duration = video_stream.get("duration") or probe.get("format", {}).get("duration")
        duration = float(raw_duration)
    except Exception as exc:
        raise RuntimeError(f"Could not determine video duration: {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Video duration must be positive: {path}")
    return duration


def extract_resized_keyframes(
    video_path: str | Path,
    timestamps: list[int],
    output_folder: str | Path,
    image_size: tuple[int, int],
) -> None:
    """Decode selected display-oriented frames directly into padded PNGs."""

    source_path = Path(video_path)
    output_path = Path(output_folder)
    staging_path = output_path.with_name(f".{output_path.name}.direct_tmp")
    width, height = image_size
    if not source_path.is_file():
        raise FileNotFoundError(f"Video source is missing: {source_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size!r}")
    if not timestamps or any(type(value) is not int or value < 0 for value in timestamps):
        raise ValueError("timestamps must contain non-negative integers")
    if timestamps != sorted(set(timestamps)):
        raise ValueError("timestamps must be sorted and unique")

    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True)
    try:
        for timestamp in timestamps:
            destination = staging_path / f"{timestamp:04d}.png"
            source = ffmpeg.input(str(source_path), ss=timestamp)
            frame = (
                source.video.filter(
                    "scale",
                    width,
                    height,
                    force_original_aspect_ratio="decrease",
                )
                .filter(
                    "pad",
                    width,
                    height,
                    "(ow-iw)/2",
                    "(oh-ih)/2",
                    color="black",
                )
                .filter("setsar", 1)
            )
            try:
                (
                    ffmpeg.output(
                        frame,
                        str(destination),
                        vframes=1,
                        format="image2",
                        vcodec="png",
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
                raise RuntimeError(
                    f"Failed to extract keyframe at {timestamp}s from {source_path}: {stderr}"
                ) from exc
            image = cv2.imread(str(destination))
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Direct keyframe has invalid dimensions at {timestamp}s: {destination}"
                )

        if output_path.exists():
            shutil.rmtree(output_path)
        staging_path.replace(output_path)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise


def extract_frames(video_path, output_folder="frames"):
    print(f"[Info] Processing video frames by PTS: {video_path}")

    source_path = Path(video_path)
    output_path = Path(output_folder)
    staging_path = output_path.with_name(f".{output_path.name}.pts_tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True)

    try:
        try:
            container = av.open(str(source_path))
        except Exception as exc:
            raise RuntimeError(f"Could not open video file: {source_path}") from exc

        pbar = tqdm(desc="[Video] Extracting by PTS", unit="frames")
        saved_count = 0
        target_sec = 0
        first_pts_time = None
        last_pts_time = None
        previous_frame = None
        previous_pts_time = None

        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            source_fps = float(stream.average_rate) if stream.average_rate else 0.0
            print("[Info] Source resolution used as-is (pre-processed to 480p)")
            print(f"[Info] Source FPS: {source_fps:.3f} | Selecting nearest frame for each PTS second")

            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise RuntimeError(f"Decoded video frame has no PTS: {source_path}")

                pts_time = frame.pts * frame.time_base
                if first_pts_time is None:
                    first_pts_time = pts_time
                normalized_pts_time = pts_time - first_pts_time
                if previous_pts_time is not None and normalized_pts_time < previous_pts_time:
                    raise RuntimeError(f"Decoded video PTS is not monotonic: {source_path}")

                while target_sec <= normalized_pts_time:
                    selected_frame = frame
                    if (
                        previous_frame is not None
                        and target_sec - previous_pts_time <= normalized_pts_time - target_sec
                    ):
                        selected_frame = previous_frame

                    filename = staging_path / f"{target_sec:04d}.png"
                    image = selected_frame.reformat(format="bgr24").to_ndarray()
                    if not cv2.imwrite(str(filename), image):
                        raise RuntimeError(f"Failed to save extracted frame: {filename}")
                    saved_count += 1
                    target_sec += 1
                    pbar.update(1)

                previous_frame = frame
                previous_pts_time = normalized_pts_time
                last_pts_time = normalized_pts_time
        finally:
            pbar.close()
            container.close()

        if first_pts_time is None or last_pts_time is None or saved_count == 0:
            raise RuntimeError(f"No video frames were decoded: {source_path}")

        duration = math.floor(last_pts_time)
        metadata = {
            "version": FRAME_EXTRACTION_VERSION,
            "source_size": source_path.stat().st_size,
            "source_mtime_ns": source_path.stat().st_mtime_ns,
            "first_pts_seconds": float(first_pts_time),
            "last_pts_seconds": float(last_pts_time),
            "duration_seconds": duration,
            "frame_count": saved_count,
        }
        frame_extraction_metadata_path(staging_path).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        staging_path.replace(output_path)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise

    print(f"[Info] Last normalized video PTS: {float(last_pts_time):.6f}s")
    print(f"[Info] Captured {saved_count} frames in {output_folder}/ (duration={duration}s)")
    return metadata
