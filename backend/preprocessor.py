import cv2
import numpy as np
from backend.config import (
    USE_POSE, USE_LEFT_HAND, USE_RIGHT_HAND,
    PURPLE
)

def build_landmark_vector(pose_r, hand_r) -> np.ndarray:
    parts = []

    if USE_POSE:
        if pose_r and pose_r.pose_landmarks:
            lm = pose_r.pose_landmarks[0]
            pose = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32).flatten()
        else:
            pose = np.zeros(33 * 4, dtype=np.float32)
        parts.append(pose)

    left_lm = right_lm = None
    if hand_r and hand_r.handedness and hand_r.hand_landmarks:
        for idx, hlist in enumerate(hand_r.handedness):
            label = hlist[0].category_name
            if label == "Left":
                left_lm = hand_r.hand_landmarks[idx]
            elif label == "Right":
                right_lm = hand_r.hand_landmarks[idx]

    if USE_LEFT_HAND:
        parts.append(
            np.array([[p.x, p.y, p.z] for p in left_lm], dtype=np.float32).flatten()
            if left_lm is not None else np.zeros(21 * 3, dtype=np.float32)
        )
    if USE_RIGHT_HAND:
        parts.append(
            np.array([[p.x, p.y, p.z] for p in right_lm], dtype=np.float32).flatten()
            if right_lm is not None else np.zeros(21 * 3, dtype=np.float32)
        )

    return np.concatenate(parts)

def interpolate_zeros(sequence: np.ndarray) -> np.ndarray:
    seq = sequence.copy()
    T = len(seq)
    
    slices = [
        (0, 132),
        (132, 195),
        (195, 258)
    ]
    
    for start, end in slices:
        for t in range(T):
            if np.sum(np.abs(seq[t, start:end])) < 1e-6:
                prev = next((i for i in range(t - 1, -1, -1) if np.sum(np.abs(seq[i, start:end])) > 1e-6), None)
                nxt  = next((i for i in range(t + 1, T)     if np.sum(np.abs(seq[i, start:end])) > 1e-6), None)
                
                if prev is not None and nxt is not None:
                    alpha = (t - prev) / (nxt - prev)
                    seq[t, start:end] = seq[prev, start:end] * (1 - alpha) + seq[nxt, start:end] * alpha
                elif prev is not None:
                    seq[t, start:end] = seq[prev, start:end]
                elif nxt is not None:
                    seq[t, start:end] = seq[nxt, start:end]
    return seq

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    if not USE_POSE:
        return sequence.astype(np.float32)

    pose_end = 33 * 4
    pose = sequence[:, :pose_end].reshape(-1, 33, 4)

    left_shoulder = pose[:, 11, :3]
    right_shoulder = pose[:, 12, :3]
    
    center = (left_shoulder + right_shoulder) / 2.0
    pose[:, :, :3] -= center[:, np.newaxis, :]

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    shoulder_dist = np.maximum(shoulder_dist, 1e-6)
    pose[:, :, :3] /= shoulder_dist[:, np.newaxis, :]

    sequence[:, :pose_end] = pose.reshape(-1, pose_end)

    if USE_LEFT_HAND:
        lh = sequence[:, 132:195].reshape(-1, 21, 3)
        lh_wrist = lh[:, 0, :]
        lh_centered = lh - lh_wrist[:, np.newaxis, :]
        sequence[:, 132:195] = lh_centered.reshape(-1, 63)

    if USE_RIGHT_HAND:
        rh = sequence[:, 195:258].reshape(-1, 21, 3)
        rh_wrist = rh[:, 0, :]
        rh_centered = rh - rh_wrist[:, np.newaxis, :]
        sequence[:, 195:258] = rh_centered.reshape(-1, 63)

    return sequence.astype(np.float32)

def draw_landmarks(image, pose_r, hand_r):
    if pose_r and pose_r.pose_landmarks:
        lm = pose_r.pose_landmarks[0]
        for pair in [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]:
            h, w = image.shape[:2]
            p1 = (int(lm[pair[0]].x * w), int(lm[pair[0]].y * h))
            p2 = (int(lm[pair[1]].x * w), int(lm[pair[1]].y * h))
            cv2.line(image, p1, p2, PURPLE, 2)
            cv2.circle(image, p1, 4, PURPLE, -1)
            cv2.circle(image, p2, 4, PURPLE, -1)

    if hand_r and hand_r.hand_landmarks:
        h, w = image.shape[:2]
        for hand in hand_r.hand_landmarks:
            for joint in hand:
                px = (int(joint.x * w), int(joint.y * h))
                cv2.circle(image, px, 3, (230, 180, 50), -1)
