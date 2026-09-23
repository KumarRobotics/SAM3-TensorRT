import ctypes
from functools import cache
from pathlib import Path

import numpy as np
from cuda.core import (
    Buffer,
    Device,
    LaunchConfig,
    LegacyPinnedMemoryResource,
    Program,
    ProgramOptions,
    launch,
)
from cuda.pathfinder import find_nvidia_header_directory

KERNELS_PATH = Path(__file__).parent / "kernels.cu"
KERNEL_NAMES = ("resizeNormalizeFloat", "resizeNormalizeHalf", "upsampleMasks")


@cache
def device():
    dev = Device()
    dev.set_current()
    return dev


@cache
def kernels():
    options = ProgramOptions(
        arch=f"sm_{device().arch}",
        std="c++17",
        include_path=find_nvidia_header_directory("cudart"),
    )
    program = Program(KERNELS_PATH.read_text(), "c++", options)
    module = program.compile("cubin")
    return {name: module.get_kernel(name) for name in KERNEL_NAMES}


def run_kernel(name, grid, block, stream, *args):
    launch(stream, LaunchConfig(grid=grid, block=block), kernels()[name], *args)


def grid_2d(width, height, block, depth=1):
    return (-(-width // block[0]), -(-height // block[1]), depth)


def _host_buffer(array):
    return Buffer.from_handle(array.ctypes.data, array.nbytes)


class GpuArray:
    def __init__(self, shape, dtype, stream=None):
        self._shape = tuple(int(s) for s in shape)
        self._dtype = np.dtype(dtype)
        stream = stream or device().default_stream
        self._buffer = device().allocate(max(self.nbytes, 1), stream=stream)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return int(np.prod(self._shape)) * self._dtype.itemsize

    @property
    def ptr(self):
        return np.uintp(int(self._buffer.handle))

    def upload(self, array, stream):
        array = np.ascontiguousarray(array, dtype=self._dtype)
        self._buffer.copy_from(_host_buffer(array), stream=stream)

    def download(self, stream):
        array = np.empty(self._shape, self._dtype)
        _host_buffer(array).copy_from(self._buffer, stream=stream)
        stream.sync()
        return array


class PinnedArray:
    def __init__(self, nbytes):
        self._buffer = LegacyPinnedMemoryResource().allocate(nbytes)
        pointer = ctypes.cast(int(self._buffer.handle), ctypes.POINTER(ctypes.c_uint8))
        self._array = np.ctypeslib.as_array(pointer, shape=(nbytes,))

    def stage(self, array):
        if array.nbytes > self._array.size:
            raise ValueError(f"PinnedArray: {array.nbytes} bytes exceeds capacity {self._array.size}")
        view = self._array[: array.nbytes].view(array.dtype).reshape(array.shape)
        view[:] = array
        return view
