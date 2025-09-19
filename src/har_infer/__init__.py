"""HAR inference package.

Provides pre-trained model loading and prediction utilities for Human Action Recognition.
"""

__all__ = [
    "get_default_model_id",
    "load_model_and_processor",
    "predict_image",
]

from .config import get_default_model_id
from .model import load_model_and_processor
from .predict import predict_image


