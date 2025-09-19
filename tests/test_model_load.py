import os
import pytest


from src.har_infer.model import load_model_and_processor


@pytest.mark.network
def test_model_load_smoke():
    # Allow overriding model via env if needed
    model_id = os.getenv("HAR_MODEL_ID", None)
    model, processor, device = load_model_and_processor(model_id)
    assert hasattr(model, "config")
    assert hasattr(processor, "preprocess") or hasattr(processor, "__call__")
    assert str(device) in {"cpu", "cuda", "mps"}


