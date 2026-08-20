import os
import ssl
import warnings
import urllib3
import re
import difflib

def proxy_setup():
    proxy_url = "http://168.219.61.252:8080"
    cert_path = '/home_nvme/shared/DigitalCity.crt'

    # urllib3 경고 비활성화 (InsecureRequestWarning 제거)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # 대문자/소문자 프록시 설정
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url

    # SSL 인증서 경로 설정
    if os.path.exists(cert_path):
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['CURL_CA_BUNDLE'] = cert_path
        os.environ['SSL_CERT_FILE'] = cert_path
    else:
        os.environ['CURL_CA_BUNDLE'] = "" # 인증서가 없을 경우 HF Hub(httpx/requests)의 SSL 검증을 비활성화
    
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # 기타 Hub 관련 설정
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    warnings.filterwarnings(action="ignore")
    print(f"[INFO] Proxy & HF Backend setup complete (SSL Verify: False)")

def clean_text(text):
    """
    텍스트에서 띄어쓰기, 특수문자를 제거하고 영문/숫자/한글만 남깁니다.
    """
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', str(text))

def deduplicate_texts(texts, similarity_threshold=0.8):
    """
    공백/특수문자 차이, 오타, 부분 문자열 등을 필터링하여 고유한 문장만 남깁니다.
    """
    unique_texts = []
    for text in texts:
        text_clean = clean_text(text)
        if not text_clean:
            continue
            
        is_duplicate = False
        for i, existing in enumerate(unique_texts):
            existing_clean = clean_text(existing)
            
            if text_clean in existing_clean or existing_clean in text_clean:
                is_duplicate = True
                if len(text) > len(existing):
                    unique_texts[i] = text
                break
            
            ratio = difflib.SequenceMatcher(None, text_clean, existing_clean).ratio()
            if ratio >= similarity_threshold:
                is_duplicate = True
                if len(text) > len(existing):
                    unique_texts[i] = text
                break
                
        if not is_duplicate:
            unique_texts.append(text)
            
    return unique_texts

# [Patch] PaddleX TextRecPredictor 몽키 패치
# yaml의 character_dict_path를 읽어서 파일을 로드하도록 강제 수정
def patch_paddlex_predictor():
    try:
        from paddlex.inference.models.text_recognition.predictor import TextRecPredictor
        from paddlex.inference.models.text_recognition.processors import CTCLabelDecode

        def custom_build_postprocess(self, **kwargs):
            if kwargs.get("name") == "CTCLabelDecode":
                # yaml 파싱 방식에 따라 리스트일 수도 있고 문자열일 수도 있음
                char_dict_path = kwargs.get("character_dict_path")
                if isinstance(char_dict_path, list):
                    char_dict_path = char_dict_path[0]
                    
                character_list = None
                try:
                    with open(char_dict_path, "r", encoding="utf-8") as f:
                        # 줄바꿈 문자를 제거하고 리스트로 만듭니다.
                        character_list = [line.strip("\n").strip("\r") for line in f.readlines()]
                except Exception as e:
                    print(f"[Error] Failed to read character dict file: {char_dict_path}. Error: {e}")
                    character_list = char_dict_path

                return CTCLabelDecode(character_list=character_list)
            else:
                raise Exception(f"Unsupported PostProcess: {kwargs.get('name')}")

        # 기존 메서드를 커스텀 메서드로 덮어쓰기
        TextRecPredictor.build_postprocess = custom_build_postprocess
        print("[INFO] Successfully patched PaddleX TextRecPredictor.")

    except ImportError as e:
        print(f"[WARNING] Could not apply PaddleX patch. Is paddlex installed? Error: {e}")
    except Exception as e:
        print(f"[WARNING] Unexpected error during PaddleX patching: {e}")

def get_video_duration(video_path: str) -> int:
    """
    첫 video frame을 0으로 정규화한 마지막 frame PTS를 초 단위로 반환합니다.
    """
    import av
    import math
    try:
        container = av.open(video_path)
        try:
            stream = container.streams.video[0]
            first_pts_time = None
            last_pts_time = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    continue
                pts_time = frame.pts * frame.time_base
                if first_pts_time is None:
                    first_pts_time = pts_time
                last_pts_time = pts_time - first_pts_time
            return math.floor(last_pts_time) if last_pts_time is not None else 0
        finally:
            container.close()
    except Exception as e:
        print(f"[Error] Failed to get video duration: {e}")
        return 0
