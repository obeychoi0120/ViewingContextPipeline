import os
import json
import logging
import re
import numpy as np
import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [OCR] %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    JSON 직렬화를 위해 Numpy 데이터 타입을 Python 기본 타입으로 변환합니다.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def extract_frame_index(filename):
    """
    파일 이름에서 숫자를 추출하여 프레임 인덱스로 사용합니다.
    예: 'frame_123.jpg' -> 123
    """
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    return -1


def _get_poly_stats(poly, img_shape):
    if poly is None or len(poly) == 0:
        return None

    img_h, img_w = img_shape[:2]
    x_coords = [float(p[0]) for p in poly]
    y_coords = [float(p[1]) for p in poly]
    height = max(y_coords) - min(y_coords)

    return {
        "x_center": ((min(x_coords) + max(x_coords)) / 2.0) / max(img_w, 1),
        "y_center": ((min(y_coords) + max(y_coords)) / 2.0) / max(img_h, 1),
        "height": height,
        "height_ratio": height / max(img_h, 1),
    }


def _is_obvious_ui_or_chat_text(text):
    compact = re.sub(r'\s+', '', str(text)).lower()
    if not compact:
        return True

    ui_terms = ("youtube", "google", "구독", "좋아요", "댓글", "subscribe")
    if any(term in compact for term in ui_terms):
        return True

    if re.match(r'^user[-_][a-z0-9_.-]+$', compact):
        return True
    if re.search(r'(ㅋ|ㅎ){3,}|(ㅜ|ㅠ|t){3,}|ㄷ{2,}', compact):
        return True

    return False


def _looks_like_top_ui_text(text):
    compact = re.sub(r'\s+', '', str(text)).lower()
    top_ui_terms = ("검색", "search", "제목없음", "저장하지", "youtube", "google")
    return any(term in compact for term in top_ui_terms)


def _looks_like_chat_text(text):
    compact = re.sub(r'\s+', '', str(text)).lower()
    if _is_obvious_ui_or_chat_text(compact):
        return True
    if re.search(r'후원했습니다|님이|치즈|ㅋㅋ|ㅎㅎ', compact):
        return True
    if re.match(r'^[0-9②③④⑤⑥⑦⑧⑨∞•·-]+[가-힣a-zA-Z]', compact) and len(compact) >= 4:
        return True
    if re.search(r'[a-z][0-9]|[0-9][a-z]', compact) and len(compact) >= 5:
        return True
    if re.fullmatch(r'@?[a-z0-9_.-]{3,}', compact) and re.search(r'\d', compact):
        return True
    return False


def should_drop_ocr_text(text, poly, img_shape, filter_ui=True,
                         top_ui_region_y=0.1, right_chat_region_x=0.8,
                         small_text_height_ratio=0.1):
    if not filter_ui:
        return False

    if _is_obvious_ui_or_chat_text(text):
        return True

    stats = _get_poly_stats(poly, img_shape) if poly is not None else None
    if stats is None:
        return False

    in_top_ui = stats["y_center"] < top_ui_region_y
    in_right_chat = stats["x_center"] > right_chat_region_x
    is_small_text = stats["height_ratio"] <= small_text_height_ratio

    if in_top_ui and _looks_like_top_ui_text(text):
        return True
    if in_right_chat and is_small_text and _looks_like_chat_text(text):
        return True

    return False


def process_ocr(frame_folder, ocr_json_path, score_thr=0.7, generate_ref=False, min_height=0,
                filter_ui=True, top_ui_region_y=0.1, right_chat_region_x=0.8,
                small_text_height_ratio=0.1):
    """
    PaddleOCR을 사용하여 프레임 이미지에서 텍스트를 추출하고 결과를 저장합니다.
    
    Args:
        frame_folder (str): 이미지가 저장된 폴더 경로
        ocr_json_path (str): 결과를 저장할 JSON 파일 경로
    """
    
    # 1. 모델 경로 설정
    BASE_MODEL_DIR = "/home_nvme/shared/models/PaddleOCR"
    REC_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models", "korean_PP-OCRv5_mobile_rec")   # Monkey patch, but works!
    if generate_ref:
        logger.info("Generating Reference. Initializing PaddleOCR PP-OCRv5_server_det...")
        DET_MODEL_NAME = "PP-OCRv5_server_det"
        DET_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models", DET_MODEL_NAME)

    else:
        logger.info("Generating On-device Inference. Initializing PaddleOCR PP-OCRv5_mobile_det...")
        DET_MODEL_NAME = "PP-OCRv5_mobile_det"
        DET_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models", DET_MODEL_NAME)
    
    # DOC_ORI_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models","PP-LCNet_x1_0_doc_ori")
    # TEXT_ORI_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models","PP-LCNet_x1_0_textline_ori")
    # DOC_UNWARP_MODEL_DIR = os.path.join(BASE_MODEL_DIR, "official_models","UVDoc")
    
    # 모델 경로 존재 확인
    if not os.path.exists(DET_MODEL_DIR) or not os.path.exists(REC_MODEL_DIR):
        logger.error(f"Model directories not found. Please check paths:\nDet: {DET_MODEL_DIR}\nRec: {REC_MODEL_DIR}")
        return
    
    # 2. PaddleOCR 초기화 (PaddleOCR/_pipelines/ocr.py의 인자 구조 따름)
    # lang 인자는 model_dir을 직접 지정하므로 생략합니다 (지정 시 경고 발생 가능).
    ocr = PaddleOCR(
        text_detection_model_dir=DET_MODEL_DIR,
        text_detection_model_name=DET_MODEL_NAME,
        text_recognition_model_dir=REC_MODEL_DIR,
        # doc_orientation_classify_model_dir=DOC_ORI_MODEL_DIR,
        # textline_orientation_model_dir=TEXT_ORI_MODEL_DIR,
        # doc_unwarping_model_dir=DOC_UNWARP_MODEL_DIR,
        use_doc_orientation_classify=False,
        use_textline_orientation=False,
        use_doc_unwarping=False,
    )

    # 3. 이미지 파일 리스트 로드 및 정렬
    if not os.path.exists(frame_folder):
        logger.error(f"Frame folder not found: {frame_folder}")
        return

    frame_files = [f for f in os.listdir(frame_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    # 프레임 순서대로 처리하기 위해 숫자로 정렬
    frame_files.sort(key=extract_frame_index)

    if not frame_files:
        logger.warning("No images found in frame folder.")
        return

    logger.info(f"Processing OCR on {len(frame_files)} frames.")

    all_frames_data = []
    
    # 4. 프레임별 OCR 수행 (tqdm 적용)
    # desc: 진행바 제목, unit: 단위
    for idx, frame_file in enumerate(tqdm(frame_files, desc="OCR Progress", unit="frames")):
        frame_path = os.path.join(frame_folder, frame_file)

        img = cv2.imread(frame_path)
        result = ocr.predict(img)

        extracted_texts = []
        
        if result:  # list(dict())
            result = result[0]
            texts = result.get('rec_texts', [])
            scores = result.get('rec_scores', [])
            polys = result.get('dt_polys', [])

            if len(texts) != 0:
                for i, (text, score) in enumerate(zip(texts, scores)):
                    if score > score_thr:
                        poly = polys[i] if len(polys) > i else None
                        if min_height > 0 and len(polys) > i:
                            y_coords = [p[1] for p in poly]
                            height = (max(y_coords) - min(y_coords))
                            if height < min_height:
                                continue

                        if should_drop_ocr_text(
                            text, poly, img.shape,
                            filter_ui=filter_ui,
                            top_ui_region_y=top_ui_region_y,
                            right_chat_region_x=right_chat_region_x,
                            small_text_height_ratio=small_text_height_ratio,
                        ):
                            continue
                        
                        extracted_texts.append(text)

        # 100 Iteration 마다 결과 출력 (tqdm.write 사용해야 진행바가 깨지지 않음)
        if (idx + 1) % 100 == 0:
            preview = ", ".join(extracted_texts) if extracted_texts else "No Text"
            # 너무 길면 자르기
            if len(preview) > 100: 
                preview = preview[:97] + "..."
            tqdm.write(f"[Iter {idx+1}] {frame_file} -> {preview}")

        # 5. 결과 그룹화
        # 프레임 인덱스를 시간(초)으로 가정 (1 FPS 캡처 기준)하거나,
        # 단순히 순차적으로 interval 만큼 그룹화합니다.
        # 여기서는 파일명의 숫자를 시간(초)으로 매핑합니다.
        frame_time = extract_frame_index(frame_file)
        
        # 파일명에 숫자가 없으면 인덱스를 사용
        if frame_time == -1:
            frame_time = idx 
            
        all_frames_data.append({
            "frame_time": frame_time,
            "texts": extracted_texts
        })
        
    # JSON 저장
    try:
        with open(ocr_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frames_data, f, ensure_ascii=False, indent=4, default=convert_numpy_types)
            
        logger.info(f"OCR frame results saved to {ocr_json_path}")
    except Exception as e:
        logger.error(f"Failed to save OCR JSON file: {e}")
