"""
Timing benchmark for Ultralytics SAM3 — mirrors the C++ test_timing test.
Runs 1-10 text prompts, 3 warmup + 10 measured runs each, reports min/max/avg.

Usage:
    python3 timing_ultralytics.py --model sam3.pt --image test/test.png
"""

import argparse
import time
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
import torchvision
import logging

# Suppress all INFO and DEBUG messages from Ultralytics
logging.getLogger("ultralytics").setLevel(logging.WARNING)

ALL_TEXTS = [
    "car", "building", "road", "person", "tree",
    "sky", "window",   "door", "sign",   "sidewalk",
]

WARMUP_RUNS  = 3
MEASURE_RUNS = 10

def run_benchmark(model_path: str, image_path: str) -> None:
    overrides = {
        "conf":     0.25,
        "task":     "segment",
        "mode":     "predict",
        "model":    model_path,
        "quantize": 16,
        "save":     False,
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)

    predictor.set_image(image_path)

    print(f"\nModel : {model_path}")
    print(f"Image : {image_path}")
    print(f"Warmup: {WARMUP_RUNS}  Measured: {MEASURE_RUNS}\n")
    print(f"{'':2}{'Texts':<12}{'Min (ms)':<14}{'Max (ms)':<14}{'Avg (ms)':<14}")
    print("-" * 54)

    for n in range(1, 11):
        texts = ALL_TEXTS[:n]

        # Warmup
        for _ in range(WARMUP_RUNS):
            predictor(text=texts)

        # Measured runs — only timing the predict call, not image encoding
        times = []
        for _ in range(MEASURE_RUNS):
            t0 = time.perf_counter()
            predictor(text=texts)
            times.append((time.perf_counter() - t0) * 1000.0)

        times = np.array(times)
        print(f"  {n:<12}{times.min():<14.2f}{times.max():<14.2f}{times.mean():<14.2f}")

    print("-" * 54)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sam3.pt",
                        help="Path to SAM3 model weights")
    parser.add_argument("--image", default="test/test.png",
                        help="Path to test image")
    args = parser.parse_args()

    run_benchmark(args.model, args.image)
