import os
import cv2
import sys
import torch
import numpy as np
import torch_tensorrt
from pathlib import Path
from typing import Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import Sam3Model, Sam3Processor, Sam3Config
    from huggingface_hub import login
except ImportError:
    print("Error: transformers>=4.47.0 is required for SAM3 support.")
    sys.exit(1)

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"

# TensorRT engines are built for a specific GPU target. Tracing/building on CPU
# will either fail or silently produce a CPU-shaped graph that isn't what gets
# deployed. Build on the same GPU (or same GPU architecture) you'll deploy on.
DEVICE = torch.device("cuda")

OUTPUT_NAMES = [f"fpn_{i}" for i in range(4)] + [f"pos_{i}" for i in range(4)]
INPUT_NAME = "pixel_values"


class ImageEncoderWrapper(torch.nn.Module):
    """
    Wraps Sam3VisionModel, ViT and FPN
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.vision_encoder = model.vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = self.vision_encoder(pixel_values)
        fpn = output.fpn_hidden_states      # tuple of 4 (B, C, H, W)
        pos = output.fpn_position_encoding  # tuple of 4 (B, C', H, W)
        return fpn[0], fpn[1], fpn[2], fpn[3], pos[0], pos[1], pos[2], pos[3]


def _report(name, out, ref, atol):
    out = out.detach().float().cpu()
    ref = ref.detach().float().cpu()
    if out.shape != ref.shape:
        print(f"  {name}: SHAPE MISMATCH wrapper={tuple(out.shape)} ref={tuple(ref.shape)}")
        return False
    d = (out - ref).abs().max().item()
    print(f"  {name}: max|diff|={d:.3e}  {'OK' if d < atol else 'FAIL'}")
    return d < atol


def validate_wrapper(model, input, atol: float = 1e-2) -> bool:
    wrapper = ImageEncoderWrapper(model).to(DEVICE).eval()
    with torch.inference_mode():
        ref = model.get_vision_features(pixel_values=input.pixel_values.to(DEVICE))
        wrapper_output = wrapper(input.pixel_values.to(DEVICE))

    ref_all = list(ref.fpn_hidden_states) + list(ref.fpn_position_encoding)
    ok = True
    for name, w, r in zip(OUTPUT_NAMES, wrapper_output, ref_all):
        ok &= _report(name, w, r, atol)
    print(f"[ImageEncoder] {'PASS' if ok else 'FAIL'}")
    return ok


def check_sdpa_in_graph(exported_program: torch.export.ExportedProgram) -> None:
    """
    Diagnostic: confirms attention actually shows up as a single fused SDPA
    node in the traced graph, rather than being decomposed into separate
    matmul/softmax/matmul ops. If SAM3's attention layers don't have an
    sdpa/flash attention implementation wired up yet, this will come back 0 —
    in which case no export strategy will give you a fused attention kernel,
    because the decomposition already happened in eager PyTorch before export
    ever saw the graph.
    """
    nodes = exported_program.graph_module.graph.nodes
    sdpa_count = sum(
        1 for n in nodes
        if n.op == "call_function" and "scaled_dot_product_attention" in str(n.target)
    )
    matmul_count = sum(
        1 for n in nodes
        if n.op == "call_function" and ("matmul" in str(n.target) or "bmm" in str(n.target))
    )
    print(f"[Diagnostic] scaled_dot_product_attention nodes in graph: {sdpa_count}")
    print(f"[Diagnostic] raw matmul/bmm nodes in graph: {matmul_count}")
    if sdpa_count == 0:
        print(
            "[WARNING] No fused SDPA op found in the traced graph. Attention is "
            "already decomposed into separate matmul/softmax ops at the PyTorch "
            "level (likely 'eager' attention). TensorRT will not get a fused "
            "attention kernel from this graph regardless of export path. Check "
            "whether Sam3Model supports attn_implementation='sdpa' and pass it "
            "explicitly in from_pretrained()."
        )


def trace_and_export_image_encoder(model: torch.nn.Module, input: Any, engine_path: Path) -> torch.Tensor:
    wrapper = ImageEncoderWrapper(model).to(DEVICE).eval()

    print("[ImageEncoderExport] Tracing Image Encoder Model")
    input = input.to(DEVICE)
    with torch.inference_mode():
        torch_output = wrapper(input)

    print("[ImageEncoderExport] Running torch.export (non-strict, fp16)")
    # enabled_precisions on the TRT builder only controls which kernels it's
    # ALLOWED to pick for internal compute -- it does not change the dtype of
    # the traced graph's inputs/outputs. torch.export records whatever dtype
    # the tensor actually was at trace time, so to get a true fp16 I/O
    # boundary (not just fp16-eligible internals with fp32 in/out casts),
    # the model and example input must themselves be half before exporting.
    wrapper_fp16 = wrapper.half()
    input_fp16 = input.half()
    with torch.inference_mode():
        exp_program = torch.export.export(wrapper_fp16, (input_fp16,), strict=False)

    check_sdpa_in_graph(exp_program)

    print("[ImageEncoderExport] Building TensorRT engine (fp16)")
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exp_program,
        arg_inputs=[input_fp16],
        use_explicit_typing=True,
        optimization_level=5,   # search harder for tactics; one-time build cost
        device=torch_tensorrt.Device(f"cuda:{input.device.index or 0}"),
    )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"[ImageEncoderExport] Wrote TensorRT engine to {engine_path}")

    return torch_output


def load_and_run_engine_standalone(engine_path: Path, input_np: np.ndarray) -> list:
    """
    Reference loader using ONLY the plain TensorRT runtime — no torch, no
    torch_tensorrt, no libtorch. This is what your deployment target should
    actually use.
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initializes a CUDA context)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    stream = cuda.Stream()
    bindings = []
    host_outputs, device_outputs = [], []
    device_input = cuda.mem_alloc(input_np.nbytes)
    cuda.memcpy_htod_async(device_input, np.ascontiguousarray(input_np), stream)

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, int(device_input))
        else:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            host_out = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device_out = cuda.mem_alloc(host_out.nbytes)
            context.set_tensor_address(name, int(device_out))
            host_outputs.append(host_out)
            device_outputs.append((device_out, shape))

    context.execute_async_v3(stream_handle=stream.handle)
    for host_out, (device_out, _) in zip(host_outputs, device_outputs):
        cuda.memcpy_dtoh_async(host_out, device_out, stream)
    stream.synchronize()

    return [h.reshape(shape) for h, (_, shape) in zip(host_outputs, device_outputs)]


if __name__ == "__main__":
    login(token=HF_KEY)
    print(f"[ImageEncoderExport] Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    config = Sam3Config.from_pretrained(MODEL_ID, token=HF_KEY)
    config.image_size = 644
    # Explicitly request SDPA. If SAM3's attention layers don't support it,
    # this will either raise or silently fall back — the diagnostic in
    # trace_and_export_image_encoder will tell you which.
    model = Sam3Model.from_pretrained(
        MODEL_ID, config=config, token=HF_KEY, attn_implementation="sdpa"
    )
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")

    processor.image_processor.size = {"height": 644, "width": 644}
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)
    validate_wrapper(model, inp)
    print(f"[ImageEncoderExport] Tracing with image size {inp.pixel_values.shape}")
    trace_and_export_image_encoder(model, inp.pixel_values, MODELS_DIR / "image_encoder.engine")
