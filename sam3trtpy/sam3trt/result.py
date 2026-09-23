from dataclasses import dataclass

import numpy as np


class FeatureMap:
    _C = 256
    _H = 46
    _W = 46

    def __init__(self, data):
        data = np.asarray(data, dtype=np.float32)
        if data.size != self._C * self._H * self._W:
            raise ValueError(f"FeatureMap: expected {self._C * self._H * self._W} floats, got {data.size}")
        self._data = data.reshape(self._C, self._H, self._W)

    @property
    def data(self):
        return self._data

    def patch(self, i, j):
        if not (0 <= i < self._H and 0 <= j < self._W):
            raise IndexError(f"FeatureMap.patch: ({i}, {j}) out of range [0,{self._H}) x [0,{self._W})")
        return self._data[:, i, j].copy()


@dataclass
class Detection:
    bbox: list
    mask: np.ndarray
    class_label: str
    class_id: int
    confidence: float


class Detections:
    def __init__(self, detections):
        self._detections = list(detections)

    def __getitem__(self, index):
        return self._detections[index]

    def __len__(self):
        return len(self._detections)

    def __iter__(self):
        return iter(self._detections)

    def size(self):
        return len(self._detections)

    def empty(self):
        return not self._detections


@dataclass
class Sam3Result:
    detections: Detections
    features: FeatureMap
