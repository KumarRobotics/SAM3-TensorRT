import cv2
import numpy as np

PALETTE = (
    (56, 56, 255),
    (255, 122, 56),
    (56, 219, 255),
    (107, 255, 56),
    (255, 56, 220),
    (56, 255, 149),
    (255, 191, 56),
    (143, 56, 255),
    (255, 56, 87),
    (56, 149, 255),
)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5


def _draw_mask(output, mask, color, mask_alpha):
    if mask is None or mask.size == 0:
        return
    if mask.shape[:2] != output.shape[:2]:
        mask = cv2.resize(mask, output.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.uint8)

    region = mask > 0
    blended = output[region] * (1.0 - mask_alpha) + np.array(color) * mask_alpha
    output[region] = np.round(blended).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 1, cv2.LINE_AA)


def _draw_bbox_and_label(output, detection, color):
    if len(detection.bbox) != 4:
        return
    x, y, w, h = detection.bbox
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, output.shape[1]) - 1, min(y + h, output.shape[0]) - 1
    cv2.rectangle(output, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    label = f"{detection.class_label} {detection.confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    label_y = max(y0, text_h + 4)

    cv2.rectangle(output, (x0, label_y - text_h - baseline - 4), (x0 + text_w + 4, label_y), color, cv2.FILLED)
    cv2.putText(output, label, (x0 + 2, label_y - baseline - 2), FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def visualize(image, result, mask_alpha=0.45):
    if image is None or image.size == 0:
        raise ValueError("visualize: input image is empty")
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    for detection in result.detections:
        color = PALETTE[detection.class_id % len(PALETTE)]
        _draw_mask(output, detection.mask, color, mask_alpha)
        _draw_bbox_and_label(output, detection, color)

    return output
