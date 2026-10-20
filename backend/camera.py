import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time
import threading
from collections import deque
from mediapipe.tasks import python as mp_python

from backend.config import (
    DATA_DIR, MODELS_DIR, POSE_MODEL_PATH, HAND_MODEL_PATH,
    SEQUENCE_LENGTH
)
import backend.state as state
from backend.preprocessor import (
    build_landmark_vector, interpolate_zeros,
    normalize_sequence, draw_landmarks
)

mp_vision = mp.tasks.vision
RunningMode = mp.tasks.vision.RunningMode

def run_model_prediction(seq_input, index_to_word, model):
    try:
        try:
            probs = model(seq_input, training=False).numpy()[0]
        except Exception as tf_err:
            print(f"[WARN] Callable inference failed, falling back to predict: {tf_err}")
            probs = model.predict(seq_input, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:5]
        
        preds = [
            {"word": index_to_word[str(i)], "confidence": float(probs[i])} 
            for i in top_idx
        ]
        state.GLOBAL_PREDICTIONS = preds
        
        best_word = preds[0]["word"]
        best_conf = preds[0]["confidence"]
        
        if best_conf >= state.GLOBAL_CONFIDENCE_THRESHOLD:
            if not state.GLOBAL_WORD_LOG or state.GLOBAL_WORD_LOG[-1] != best_word:
                state.GLOBAL_WORD_LOG.append(best_word)
    except Exception as e:
        print(f"[ERROR] Background inference failed: {e}")

def camera_loop():
    model_path = MODELS_DIR / "signsight_bilstm.keras"
    label_map_path = DATA_DIR / "label_map.json"

    if not model_path.exists() or not label_map_path.exists():
        print("[ERROR] Model files missing. Cannot execute stream predictions.")
        return

    model = tf.keras.models.load_model(str(model_path))
    with open(label_map_path) as f:
        label_info = json.load(f)
    index_to_word = label_info["index_to_word"]

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    is_active_signing = False
    active_sequence = []
    prev_wrists = None
    quiet_frames_count = 0
    motion_history = deque(maxlen=5)

    try:
        with mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm, \
             mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:

            frame_ts = 0

            while cap.isOpened() and state.CAMERA_ACTIVE:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.015)
                    continue

                processing_frame = cv2.flip(frame, 1) if state.FLIP_INPUT else frame.copy()
                rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                cur_pose_r = pose_lm.detect_for_video(mp_img, frame_ts)
                cur_hand_r = hand_lm.detect_for_video(mp_img, frame_ts)
                frame_ts += 33

                landmarks = build_landmark_vector(cur_pose_r, cur_hand_r)

                current_motion = 0.0
                pose_detected = cur_pose_r and cur_pose_r.pose_landmarks
                
                if pose_detected:
                    lm = cur_pose_r.pose_landmarks[0]
                    if lm[11].visibility > 0.5 and lm[12].visibility > 0.5:
                        shoulder_width = abs(lm[11].x - lm[12].x)
                        if shoulder_width > 0.45:
                            state.GLOBAL_POSE_STATUS = "TOO_CLOSE"
                        elif state.GLOBAL_POSE_STATUS != "TOO_FAST":
                            state.GLOBAL_POSE_STATUS = "OK"
                    elif state.GLOBAL_POSE_STATUS != "TOO_FAST":
                        state.GLOBAL_POSE_STATUS = "NOT_DETECTED"
                else:
                    state.GLOBAL_POSE_STATUS = "NOT_DETECTED"

                left_detected = False
                right_detected = False
                if cur_hand_r and cur_hand_r.handedness and cur_hand_r.hand_landmarks:
                    for hlist in cur_hand_r.handedness:
                        label = hlist[0].category_name
                        if label == "Left":    left_detected = True
                        elif label == "Right": right_detected = True
                
                state.GLOBAL_LEFT_HAND_ACTIVE = left_detected
                state.GLOBAL_RIGHT_HAND_ACTIVE = right_detected

                if pose_detected:
                    lm = cur_pose_r.pose_landmarks[0]
                    curr_wrists = np.array([
                        [lm[15].x, lm[15].y, lm[15].z],
                        [lm[16].x, lm[16].y, lm[16].z]
                    ])
                    if prev_wrists is not None:
                        motion_dist = np.linalg.norm(curr_wrists - prev_wrists, axis=1)
                        current_motion = float(np.max(motion_dist))
                    prev_wrists = curr_wrists
                else:
                    prev_wrists = None

                motion_history.append(current_motion)
                avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0.0
                hands_detected = cur_hand_r and cur_hand_r.hand_landmarks

                if not is_active_signing:
                    if avg_motion > 0.018 and hands_detected:
                        is_active_signing = True
                        state.GLOBAL_IS_SIGNING = True
                        state.GLOBAL_POSE_STATUS = "OK"
                        active_sequence = [landmarks]
                        quiet_frames_count = 0
                else:
                    active_sequence.append(landmarks)
                    if avg_motion < 0.011 or not hands_detected:
                        quiet_frames_count += 1
                    else:
                        quiet_frames_count = 0

                    if quiet_frames_count >= 8 or len(active_sequence) > 120:
                        is_active_signing = False
                        state.GLOBAL_IS_SIGNING = False
                        total_frames = len(active_sequence) - quiet_frames_count
                        
                        if total_frames >= 10:
                            trimmed_seq = active_sequence[:total_frames]

                            if total_frames >= SEQUENCE_LENGTH:
                                indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
                                seq_arr = np.array([trimmed_seq[i] for i in indices])
                            else:
                                pad_count = SEQUENCE_LENGTH - total_frames
                                feat_dim = len(trimmed_seq[0])
                                padding = [np.zeros(feat_dim, dtype=np.float32)] * pad_count
                                seq_arr = np.array(trimmed_seq + padding)

                            seq_arr = interpolate_zeros(seq_arr)
                            seq_arr = normalize_sequence(seq_arr)
                            seq_input = seq_arr[np.newaxis]

                            threading.Thread(
                                target=run_model_prediction, 
                                args=(seq_input, index_to_word, model), 
                                daemon=True
                            ).start()
                        elif total_frames > 2:
                            state.GLOBAL_POSE_STATUS = "TOO_FAST"

                draw_landmarks(frame, cur_pose_r, cur_hand_r)
                
                disp_frame = cv2.flip(frame, 1)

                ret, jpeg_buffer = cv2.imencode('.jpg', disp_frame)
                if not ret:
                    continue

                state.GLOBAL_FRAME = jpeg_buffer.tobytes()
                time.sleep(0.001)
    finally:
        cap.release()
        state.GLOBAL_FRAME = None
        state.GLOBAL_LEFT_HAND_ACTIVE = False
        state.GLOBAL_RIGHT_HAND_ACTIVE = False
        state.GLOBAL_POSE_STATUS = "NOT_DETECTED"

def generate_video_stream():
    while True:
        if not state.CAMERA_ACTIVE or state.GLOBAL_FRAME is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Offline", (190, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.putText(placeholder, "Press [Start Camera] to begin translation", (130, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.2)
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.GLOBAL_FRAME + b'\r\n')
            time.sleep(0.033)
