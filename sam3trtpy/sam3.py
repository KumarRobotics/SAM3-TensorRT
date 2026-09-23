import sys
from pathlib import Path

import cv2

from sam3trt import CONFIG_DIR, Sam3Model, visualize

MODELS_DIR = Path.home() / "models"


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <image_path> <label>", file=sys.stderr)
        return 1
    image_path = Path(sys.argv[1])
    label = sys.argv[2]

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"could not read image: {image_path}", file=sys.stderr)
        return 1

    model = Sam3Model(
        MODELS_DIR,
        CONFIG_DIR / "merges.txt",
        CONFIG_DIR / "vocab.json",
        CONFIG_DIR / "params.json",
    )

    result = model.infer(image, label)
    print(f"[Sam3] Detected: {len(result.detections)} object(s)")

    viz_path = image_path.parent / f"{image_path.stem}_viz.png"
    cv2.imwrite(str(viz_path), visualize(image, result))
    print(f"[Sam3] Saved: {viz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
