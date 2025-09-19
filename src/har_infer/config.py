from __future__ import annotations

import os
from typing import Final


DEFAULT_MODEL_ID: Final[str] = (
    os.getenv(
        "HAR_MODEL_ID",
        # ViT-B/16 fine-tuned on the Bingsu HAR dataset (15 classes)
        "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224",
    )
)


def get_default_model_id() -> str:
    """Return the default image-classification model id for HAR inference.

    Respects the HAR_MODEL_ID environment variable.
    """

    return DEFAULT_MODEL_ID


