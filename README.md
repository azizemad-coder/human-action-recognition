Human Action Recognition (Inference-Only)

This project provides a ready-to-run inference application (no training) for Human Action Recognition using a pre-trained model fine-tuned on the 15-class HAR dataset. It includes a CLI and a Gradio web UI.

Reference dataset: [Bingsu/Human_Action_Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition)

Classes (15): calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop

ملحوظة: المشروع يعتمد على موديل مدرّب مسبقًا؛ لا يوجد تدريب داخل هذا الريبو. عند التشغيل لأول مرة سيقوم بتنزيل الموديل تلقائيًا من Hugging Face.

Project structure

```
.
├─ app.py                         # Launch the Gradio UI
├─ requirements.txt               # Python dependencies
├─ README.md                      # This file
├─ pytest.ini                     # Pytest config
├─ .gitignore
├─ src/
│  └─ har_infer/
│     ├─ __init__.py
│     ├─ app.py                  # Gradio interface definition
│     ├─ cli.py                  # CLI for image/video inference
│     ├─ config.py               # Configuration (model id, device)
│     ├─ labels.py               # Class label mappings
│     ├─ model.py                # Model/processor loading
│     └─ predict.py              # Image & video inference helpers
└─ tests/
   ├─ test_labels.py
   ├─ test_model_load.py
   └─ test_predict_image.py
```

Requirements

- Python 3.9+
- Internet access on first run to download the model

Installation

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configuration (optional)

- HAR_MODEL_ID: override the default model id. By default, the app uses a ViT model fine-tuned for this dataset: `rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224`.

```bash
set HAR_MODEL_ID=rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224  # Windows (cmd)
$env:HAR_MODEL_ID="rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224" # PowerShell
export HAR_MODEL_ID=rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224 # Bash
```

Run the Gradio app

```bash
python app.py
```

This opens a local URL (e.g., http://127.0.0.1:7860) for testing. Upload an image or a video. Video predictions are aggregated across sampled frames.

CLI usage

```bash
# Image
python -m har_infer.cli --image path/to/image.jpg --top-k 5

# Video (aggregates per-frame predictions)
python -m har_infer.cli --video path/to/video.mp4 --sample-fps 2 --max-frames 64
```

Notes

- First run downloads model weights to the Hugging Face cache.
- If you have a GPU, the app will use it automatically when available.

Testing

```bash
pytest -q
```

Some tests are marked as network-dependent and will be skipped if there is no internet connection.

Why these libraries?

- transformers/torch: load and run pre-trained image classification models.
- gradio: fast, simple web UI for demos.
- opencv-python: lightweight video frame sampling.
- pillow: image handling.

Acknowledgments

- Dataset: [Bingsu/Human_Action_Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition)


