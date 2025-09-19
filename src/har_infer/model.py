from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from .config import get_default_model_id


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_processor(
    model_id: str | None = None, *, device: torch.device | None = None
) -> Tuple[AutoModelForImageClassification, AutoImageProcessor, torch.device]:
    """Load a pre-trained image classification model and its processor.

    Parameters
    ----------
    model_id: Optional model repo id. If None, uses get_default_model_id().
    device: Optional torch.device. If None, auto-detects (cuda/mps/cpu).

    Returns
    -------
    model, processor, device
    """

    resolved_model_id = model_id or get_default_model_id()
    resolved_device = device or _get_device()

    processor = AutoImageProcessor.from_pretrained(resolved_model_id)
    model = AutoModelForImageClassification.from_pretrained(resolved_model_id)
    model.to(resolved_device)
    model.eval()
    return model, processor, resolved_device


