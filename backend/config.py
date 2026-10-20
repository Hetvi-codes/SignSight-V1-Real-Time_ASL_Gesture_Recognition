import os
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 258
USE_POSE = True
USE_LEFT_HAND = True
USE_RIGHT_HAND = True

BG_DARK = (15, 15, 25)
PURPLE = (180, 50, 230)
WHITE = (245, 245, 245)

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_WORDS = [
    "hello", "good", "bye", "please", "sorry", "yes", "no", "help",
    "want", "need", "like", "who", "what", "where", "when", "why",
    "how", "because", "wait", "go", "come", "stop", "call", "tell",
    "give", "take", "mother", "father", "brother", "sister", "family",
    "man", "woman", "doctor", "deaf", "cousin", "daughter", "eat",
    "drink", "water", "food", "pizza", "apple", "today", "tomorrow",
    "yesterday", "work", "study", "school", "computer", "phone", "cold"
]

def ensure_models():
    pose_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    if not POSE_MODEL_PATH.exists():
        print(f"  Downloading Pose Landmarker model...")
        urllib.request.urlretrieve(pose_url, POSE_MODEL_PATH)
        print("  ✅ Download completed.")

    if not HAND_MODEL_PATH.exists():
        print(f"  Downloading Hand Landmarker model...")
        urllib.request.urlretrieve(hand_url, HAND_MODEL_PATH)
        print("  ✅ Download completed.")
