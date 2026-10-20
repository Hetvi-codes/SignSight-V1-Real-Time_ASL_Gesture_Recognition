import json
import threading
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

from backend.config import ROOT_DIR, DATA_DIR, DEFAULT_WORDS, ensure_models
import backend.state as state
from backend.camera import generate_video_stream, camera_loop

router = APIRouter()

@router.get("/video_feed")
def get_video_feed():
    ensure_models()
    return StreamingResponse(
        generate_video_stream(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/api/predictions")
def get_predictions():
    return {
        "top_predictions": state.GLOBAL_PREDICTIONS,
        "word_log": state.GLOBAL_WORD_LOG,
        "is_signing": state.GLOBAL_IS_SIGNING,
        "left_hand_active": state.GLOBAL_LEFT_HAND_ACTIVE,
        "right_hand_active": state.GLOBAL_RIGHT_HAND_ACTIVE,
        "pose_status": state.GLOBAL_POSE_STATUS
    }

@router.post("/api/clear")
def clear_predictions_log():
    state.GLOBAL_WORD_LOG.clear()
    state.GLOBAL_PREDICTIONS.clear()
    return {"status": "cleared"}

@router.post("/api/toggle_mirror")
def toggle_mirror_flip():
    state.FLIP_INPUT = not state.FLIP_INPUT
    return {"mirror": state.FLIP_INPUT}

@router.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    threshold = data.get("confidence_threshold")
    if threshold is not None:
        state.GLOBAL_CONFIDENCE_THRESHOLD = float(threshold)
    return {"confidence_threshold": state.GLOBAL_CONFIDENCE_THRESHOLD}

@router.post("/api/camera/start")
def start_camera():
    if not state.CAMERA_ACTIVE:
        state.CAMERA_ACTIVE = True
        state.CAMERA_THREAD = threading.Thread(target=camera_loop, daemon=True)
        state.CAMERA_THREAD.start()
    return {"status": "started"}

@router.post("/api/camera/stop")
def stop_camera():
    state.CAMERA_ACTIVE = False
    state.GLOBAL_IS_SIGNING = False
    state.GLOBAL_PREDICTIONS.clear()
    return {"status": "stopped"}

@router.get("/api/words")
def get_words_dictionary():
    label_map_path = DATA_DIR / "label_map.json"
    all_words = []
    
    if label_map_path.exists():
        try:
            with open(label_map_path) as f:
                label_info = json.load(f)
            all_words = sorted(list(label_info["word_to_index"].keys()))
        except Exception:
            all_words = DEFAULT_WORDS
    else:
        all_words = DEFAULT_WORDS
        
    return all_words

@router.get("/api/video/{word}")
def get_reference_video(word: str):
    word_dir = ROOT_DIR / "Dataset_v1" / word
    if not word_dir.exists():
        word_dir = ROOT_DIR / "Dataset_v1" / word.lower()

    if word_dir.exists():
        video_files = list(word_dir.glob("*.mp4"))
        if video_files:
            return FileResponse(str(video_files[0]), media_type="video/mp4")
            
    return HTMLResponse("Reference clip not found.", status_code=404)
