from __future__ import annotations

import gradio as gr
from PIL import Image

from .model import load_model_and_processor
from .predict import predict_image, predict_video


MODEL, PROCESSOR, DEVICE = load_model_and_processor()


def predict_image_ui(img: Image.Image, top_k: int = 5):
    preds = predict_image(img, MODEL, PROCESSOR, device=DEVICE, top_k=top_k)
    return {p.label: float(p.score) for p in preds}


def predict_video_ui(video_file: str, sample_fps: float = 2.0, max_frames: int = 64, top_k: int = 5):
    agg, per_frame = predict_video(
        video_file,
        MODEL,
        PROCESSOR,
        device=DEVICE,
        sample_fps=sample_fps,
        max_frames=max_frames,
        top_k=top_k,
    )
    return {p.label: float(p.score) for p in agg}


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Human Action Recognition (Inference)") as demo:
        gr.Markdown("""
        **Human Action Recognition (15 classes)** — pre-trained model inference demo.
        Upload an image or a video. Video predictions aggregate across sampled frames.
        """)

        with gr.Tab("Image"):
            with gr.Row():
                img = gr.Image(type="pil", label="Input Image")
                topk = gr.Slider(1, 15, value=5, step=1, label="Top K")
            out = gr.Label(label="Top-K Predictions")
            btn = gr.Button("Predict")
            btn.click(predict_image_ui, inputs=[img, topk], outputs=[out])

        with gr.Tab("Video"):
            with gr.Row():
                vid = gr.Video(label="Input Video")
            with gr.Row():
                fps = gr.Slider(0.5, 8.0, value=2.0, step=0.5, label="Sample FPS")
                maxf = gr.Slider(8, 128, value=64, step=8, label="Max Frames")
                topk2 = gr.Slider(1, 15, value=5, step=1, label="Top K")
            out2 = gr.Label(label="Aggregated Predictions")
            btn2 = gr.Button("Predict Video")
            btn2.click(predict_video_ui, inputs=[vid, fps, maxf, topk2], outputs=[out2])

        return demo


def launch():
    demo = build_interface()
    demo.launch()


