from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from sam3trt.gpu import GpuArray


@dataclass
class Sam3ImageFeatures:
    fpn: list
    fpn_pos: list


@dataclass
class Sam3TextFeatures:
    text_features: GpuArray
    text_embeddings: GpuArray


@dataclass
class Sam3DecoderOutput:
    predicted_logits: GpuArray
    predicted_boxes: GpuArray
    predicted_boxes_xyxy: GpuArray
    presence_logits: GpuArray
    predicted_masks: GpuArray


class Sam3ModelBase:
    _logger = trt.Logger(trt.Logger.WARNING)

    def __init__(self, plan_path):
        trt.init_libnvinfer_plugins(self._logger, "")
        self._runtime = trt.Runtime(self._logger)
        with open(plan_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"{type(self).__name__}: failed to deserialize engine: {plan_path}")
        self._context = self._engine.create_execution_context()
        self._outputs = {name: self._allocate(name) for name in self._tensor_names(trt.TensorIOMode.OUTPUT)}

    def _tensor_names(self, mode):
        names = (self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors))
        return [name for name in names if self._engine.get_tensor_mode(name) == mode]

    def _tensor_dtype(self, name):
        return np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))

    def _allocate(self, name):
        shape = self._engine.get_tensor_shape(name)
        array = GpuArray(shape, self._tensor_dtype(name))
        self._bind(name, array)
        return array

    def _bind(self, name, array):
        self._context.set_tensor_address(name, int(array.ptr))

    def _enqueue(self, stream):
        if not self._context.execute_async_v3(int(stream.handle)):
            raise RuntimeError(f"{type(self).__name__}: execute_async_v3 failed")


class Sam3ImageEncoder(Sam3ModelBase):
    _INPUT = "pixel_values"

    @property
    def input_dtype(self):
        return self._tensor_dtype(self._INPUT)

    def encode(self, image, stream):
        self._bind(self._INPUT, image)
        self._enqueue(stream)
        outputs = [self._outputs[f"output{i}"] for i in range(6)]
        return Sam3ImageFeatures(fpn=outputs[:3], fpn_pos=outputs[3:])


class Sam3TextEncoder(Sam3ModelBase):
    def encode(self, input_ids, attention_mask, stream):
        self._bind("input_ids", input_ids)
        self._bind("output1", attention_mask)
        self._enqueue(stream)
        return Sam3TextFeatures(self._outputs["output0"], self._outputs["output2"])


class Sam3MaskDecoder(Sam3ModelBase):
    def decode(self, image_features, text_features, attention_mask, attention_mask_f, stream):
        for i in range(3):
            self._bind(f"fpn{i}", image_features.fpn[i])
            self._bind(f"pos{i}", image_features.fpn_pos[i])
        self._bind("txt_feats", text_features.text_features)
        self._bind("txt_masks", attention_mask)
        self._bind("txt_masks_f", attention_mask_f)
        self._enqueue(stream)
        return Sam3DecoderOutput(*(self._outputs[f"output{i}"] for i in range(5)))
