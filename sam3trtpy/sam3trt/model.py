import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sam3trt.engines import Sam3ImageEncoder, Sam3MaskDecoder, Sam3TextEncoder
from sam3trt.gpu import GpuArray, device, grid_2d, run_kernel
from sam3trt.processor import Sam3Processor
from sam3trt.result import Detection, Detections, FeatureMap, Sam3Result


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


class Sam3Model:
    _FEATURE_IDX = 2
    _BLOCK = (32, 16, 1)

    @dataclass
    class _Config:
        presence_threshold: float = 0.5
        confidence_threshold: float = 0.3
        image_mean: tuple = (0.485, 0.456, 0.406)
        image_std: tuple = (0.229, 0.224, 0.225)
        image_size: int = 644
        text_padding: int = 32
        max_img_h: int = 1536
        max_img_w: int = 2048

    def __init__(self, models_path, merges_path, vocab_path, config_path):
        models_path = Path(models_path)
        self._config = self._load_config(config_path)
        self._stream = device().create_stream()
        self._image_encoder = Sam3ImageEncoder(models_path / "image_encoder_fp16.engine")
        self._text_encoder = Sam3TextEncoder(models_path / "text_encoder_fp16.engine")
        self._mask_decoder = Sam3MaskDecoder(models_path / "mask_decoder_fp16.engine")
        self._processor = Sam3Processor(merges_path, vocab_path, self._processor_params())

    @classmethod
    def _load_config(cls, path):
        with open(path) as f:
            values = json.load(f)
        return cls._Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})

    def _processor_params(self):
        return Sam3Processor.Params(
            image_size=self._config.image_size,
            text_padding=self._config.text_padding,
            image_mean=self._config.image_mean,
            image_std=self._config.image_std,
            image_dtype=self._image_encoder.input_dtype,
            max_img_h=self._config.max_img_h,
            max_img_w=self._config.max_img_w,
        )

    def infer(self, image, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            raise ValueError("Sam3Model.infer: texts must not be empty")

        image_input = self._processor.preprocess_image(image)
        image_features = self._image_encoder.encode(image_input.image, self._stream)
        features = FeatureMap(image_features.fpn[self._FEATURE_IDX].download(self._stream))

        detections = []
        for class_id, text in enumerate(texts):
            detections += self._forward(image_features, text, class_id, *image_input.original_size)

        return Sam3Result(Detections(detections), features)

    def _forward(self, image_features, text, class_id, orig_h, orig_w):
        text_input = self._processor.preprocess_text(text)
        text_features = self._text_encoder.encode(
            text_input.input_ids,
            text_input.attention_mask,
            self._stream,
        )
        output = self._mask_decoder.decode(
            image_features,
            text_features,
            text_input.attention_mask,
            text_input.attention_mask_f,
            self._stream,
        )

        presence = _sigmoid(output.presence_logits.download(self._stream)).item()
        if presence < self._config.presence_threshold:
            return []

        confidences = _sigmoid(output.predicted_logits.download(self._stream).reshape(-1))
        indices = np.flatnonzero(confidences >= self._config.confidence_threshold).astype(np.int32)
        if indices.size == 0:
            return []

        masks, boxes = self._upsample(output.predicted_masks, indices, orig_h, orig_w)
        return [
            Detection(self._to_xywh(box), mask, text, class_id, float(confidences[index]))
            for index, mask, box in zip(indices, masks, boxes)
        ]

    def _upsample(self, predicted_masks, indices, orig_h, orig_w):
        count = indices.size
        src_h, src_w = predicted_masks.shape[-2:]

        d_indices = GpuArray(indices.shape, np.int32, self._stream)
        d_masks = GpuArray((count, orig_h, orig_w), np.uint8, self._stream)
        d_boxes = GpuArray((count, 4), np.int32, self._stream)
        d_indices.upload(indices, self._stream)
        d_boxes.upload(np.tile(np.array([orig_w, orig_h, -1, -1], np.int32), (count, 1)), self._stream)

        run_kernel(
            "upsampleMasks",
            grid_2d(orig_w, orig_h, self._BLOCK, count),
            self._BLOCK,
            self._stream,
            predicted_masks.ptr,
            d_indices.ptr,
            d_masks.ptr,
            d_boxes.ptr,
            *(np.int32(v) for v in (src_h, src_w, orig_h, orig_w)),
        )
        return d_masks.download(self._stream), d_boxes.download(self._stream)

    @staticmethod
    def _to_xywh(box):
        x0, y0, x1, y1 = (int(v) for v in box)
        if x1 < 0:
            return [0, 0, 0, 0]
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
