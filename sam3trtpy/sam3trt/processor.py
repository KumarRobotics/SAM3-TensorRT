from dataclasses import dataclass

import numpy as np

from sam3trt.gpu import GpuArray, PinnedArray, device, grid_2d, run_kernel
from sam3trt.tokenizer import ClipTokenizer


@dataclass
class Sam3ImageInput:
    image: GpuArray
    original_size: tuple


@dataclass
class Sam3TextInput:
    input_ids: GpuArray
    attention_mask: GpuArray
    attention_mask_f: GpuArray


@dataclass
class Sam3Input:
    image: Sam3ImageInput
    text: list


class Sam3Processor:
    _BLOCK = (32, 16, 1)
    _RESIZE_KERNELS = {"float32": "resizeNormalizeFloat", "float16": "resizeNormalizeHalf"}

    @dataclass
    class Params:
        image_size: int = 644
        text_padding: int = 32
        image_mean: tuple = (0.485, 0.456, 0.406)
        image_std: tuple = (0.229, 0.224, 0.225)
        image_dtype: type = np.float32
        max_img_h: int = 1536
        max_img_w: int = 2048

    def __init__(self, merges_path, vocab_path, params=None):
        self._params = params or self.Params()
        self._tokenizer = ClipTokenizer(merges_path, vocab_path)
        self._stream = device().create_stream()
        self._allocate_buffers()

    def _allocate_buffers(self):
        size, length = self._params.image_size, self._params.text_padding
        self._staging = PinnedArray(self._params.max_img_h * self._params.max_img_w * 3)
        self._image = GpuArray((1, 3, size, size), self._params.image_dtype)
        self._input_ids = GpuArray((1, length), np.int64)
        self._attention_mask = GpuArray((1, length), np.bool_)
        self._attention_mask_f = GpuArray((1, length), np.float16)

    def preprocess_image(self, image):
        if image is None or image.size == 0:
            raise ValueError("Sam3Processor.preprocess_image: empty image")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Sam3Processor.preprocess_image: expected HxWx3 uint8 BGR image")

        src_h, src_w = image.shape[:2]
        size = self._params.image_size
        mean = [np.float32(m) for m in self._params.image_mean]
        inv_std = [np.float32(1.0 / s) for s in self._params.image_std]

        source = GpuArray(image.shape, np.uint8, self._stream)
        source.upload(self._staging.stage(image), self._stream)
        run_kernel(
            self._RESIZE_KERNELS[np.dtype(self._params.image_dtype).name],
            grid_2d(size, size, self._BLOCK),
            self._BLOCK,
            self._stream,
            source.ptr,
            self._image.ptr,
            *(np.int32(v) for v in (src_h, src_w, size, size)),
            *mean,
            *inv_std,
        )
        self._stream.sync()
        return Sam3ImageInput(self._image, (src_h, src_w))

    def preprocess_text(self, text):
        if not text:
            raise ValueError("Sam3Processor.preprocess_text: empty string")

        length = self._params.text_padding
        tokens = self._tokenizer.tokenize(text)[:length]

        input_ids = np.zeros((1, length), np.int64)
        input_ids[0, : len(tokens)] = tokens
        attention_mask = np.ones((1, length), np.bool_)
        attention_mask[0, : len(tokens)] = False

        self._input_ids.upload(input_ids, self._stream)
        self._attention_mask.upload(attention_mask, self._stream)
        self._attention_mask_f.upload(attention_mask.astype(np.float16), self._stream)
        self._stream.sync()
        return Sam3TextInput(self._input_ids, self._attention_mask, self._attention_mask_f)

    def preprocess(self, image, texts):
        return Sam3Input(self.preprocess_image(image), [self.preprocess_text(t) for t in texts])
