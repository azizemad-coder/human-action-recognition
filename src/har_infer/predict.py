from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)


def _topk(logits: torch.Tensor, id2label: dict[int, str], k: int) -> List[Prediction]:
    probs = torch.softmax(logits, dim=-1)
    scores, ids = torch.topk(probs, k=min(k, probs.shape[-1]))
    out: List[Prediction] = []
    for s, i in zip(scores.tolist(), ids.tolist()):
        out.append(Prediction(label=id2label[int(i)], score=float(s)))
    return out


def predict_image(
    image: Image.Image,
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    *,
    device: torch.device,
    top_k: int = 5,
) -> List[Prediction]:
    """Predict top-k labels for a PIL image."""

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
    return _topk(logits, model.config.id2label, top_k)


def sample_video_frames(
    video_path: str, *, sample_fps: float = 2.0, max_frames: int = 64
) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(native_fps / max(sample_fps, 1e-6))), 1)

    frames: List[Image.Image] = []
    idx = 0
    read_count = 0
    while read_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
            read_count += 1
        idx += 1

    cap.release()
    return frames


def predict_video(
    video_path: str,
    model: AutoModelForImageClassification,
    processor: AutoImageProcessor,
    *,
    device: torch.device,
    sample_fps: float = 2.0,
    max_frames: int = 64,
    top_k: int = 5,
) -> Tuple[List[Prediction], List[List[Prediction]]]:
    """Predict action for a video by aggregating sampled frame predictions.

    Returns a tuple of (aggregated_topk, per_frame_topk_list).
    Aggregation is the mean of per-class probabilities across frames.
    """

    frames = sample_video_frames(video_path, sample_fps=sample_fps, max_frames=max_frames)
    if not frames:
        raise RuntimeError("No frames sampled from video")

    per_frame: List[List[Prediction]] = []
    all_probs: List[np.ndarray] = []

    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            per_frame.append(_topk(logits, model.config.id2label, top_k))

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    # Convert to torch for reuse of _topk by building a tensor
    logits_sim = torch.from_numpy(np.log(mean_probs + 1e-9))
    aggregated = _topk(logits_sim, model.config.id2label, top_k)
    return aggregated, per_frame


