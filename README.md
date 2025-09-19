# 🎭 Human Action Recognition

> **Ready-to-use AI model** that recognizes 15 different human activities from images and videos

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-4.41+-orange.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/🤗-transformers-yellow.svg)](https://huggingface.co/transformers/)

## ✨ Features

- **🚀 Zero Training** - Uses pre-trained ViT model
- **🖼️ Image & Video** - Supports both formats
- **🌐 Web UI** - Beautiful Gradio interface
- **⚡ CLI Tool** - Command-line interface
- **🎯 15 Actions** - calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop

## 🚀 Quick Start

### Option 1: Colab (Recommended)
```python
# 1. Clone and install
!git clone https://github.com/azizemad-coder/human-action-recognition.git
%cd human-action-recognition
!pip install -r requirements.txt

# 2. Test with dataset sample
!python test_har.py

# 3. Launch web UI
from src.har_infer.app import build_interface
demo = build_interface()
demo.launch(share=True)  # Creates public link
```

### Option 2: Local Setup
```bash
git clone https://github.com/azizemad-coder/human-action-recognition.git
cd human-action-recognition
pip install -r requirements.txt
python app.py  # Opens http://127.0.0.1:7860
```

## 🎯 Usage

### Web Interface
```bash
python app.py
```
Upload images or videos and get instant predictions!

### Command Line
```bash
# Single image
python -m har_infer.cli --image photo.jpg --top-k 5

# Video analysis
python -m har_infer.cli --video dance.mp4 --sample-fps 2
```

### Python API
```python
from har_infer import load_model_and_processor, predict_image
from PIL import Image

model, processor, device = load_model_and_processor()
image = Image.open("action.jpg")
predictions = predict_image(image, model, processor, device=device)

for pred in predictions:
    print(f"{pred.label}: {pred.score:.3f}")
```

## 📊 Dataset

Built on [Bingsu/Human_Action_Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition) dataset with 18K labeled images across 15 action classes.

## ⚡ Performance

- **Model**: ViT-Base/16 fine-tuned for HAR
- **Speed**: ~50ms per image (GPU), ~200ms (CPU)
- **Memory**: ~500MB model download
- **Accuracy**: High precision on real-world images

## 🛠️ Advanced

### Custom Model
```bash
export HAR_MODEL_ID="your-custom-model-id"
python app.py
```

### Testing
```bash
pytest -q  # Run all tests
```

## 📁 Project Structure

```
📦 human-action-recognition
├── 🚀 app.py                 # Launch Gradio UI
├── 🧪 test_har.py           # Comprehensive test script
├── 📋 requirements.txt      # Dependencies
├── 📁 src/har_infer/        # Core package
│   ├── 🖥️ app.py           # Gradio interface
│   ├── ⚡ cli.py           # Command-line tool
│   ├── 🤖 model.py         # Model loading
│   └── 🔮 predict.py       # Inference logic
└── 🧪 tests/               # Unit tests
```

---

**Dataset Reference**: [Bingsu/Human_Action_Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition)