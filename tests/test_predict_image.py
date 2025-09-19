import io

import numpy as np
import pytest
from PIL import Image

from src.har_infer.model import load_model_and_processor
from src.har_infer.predict import predict_image


@pytest.mark.network
def test_predict_image_smoke():
    # Create a simple RGB image for smoke test
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :112] = [255, 0, 0]  # half red
    img = Image.fromarray(arr)

    model, processor, device = load_model_and_processor()
    preds = predict_image(img, model, processor, device=device, top_k=3)
    assert len(preds) >= 1
    assert 0.0 <= preds[0].score <= 1.0


