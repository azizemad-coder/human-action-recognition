from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image

from .model import load_model_and_processor
from .predict import Prediction, predict_image, predict_video


def _print(preds: List[Prediction], *, top_k: int):
    rows = [{"label": p.label, "score": round(p.score, 4)} for p in preds[:top_k]]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HAR inference (image/video)")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to image file")
    g.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predictions")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Video sampling FPS")
    parser.add_argument("--max-frames", type=int, default=64, help="Max frames to sample from video")

    args = parser.parse_args(argv)

    model, processor, device = load_model_and_processor()

    if args.image:
        image_path = Path(args.image)
        im = Image.open(image_path).convert("RGB")
        preds = predict_image(im, model, processor, device=device, top_k=args.top_k)
        _print(preds, top_k=args.top_k)
        return 0

    if args.video:
        aggregated, per_frame = predict_video(
            args.video,
            model,
            processor,
            device=device,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            top_k=args.top_k,
        )
        _print(aggregated, top_k=args.top_k)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


