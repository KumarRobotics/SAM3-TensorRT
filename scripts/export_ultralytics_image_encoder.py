import os
import cv2
import copy
import torch
import argparse
import torch_tensorrt
from ultralytics.models.sam import SAM3SemanticPredictor

from typing import Tuple, Any

import tensorrt as trt

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageEncoderWrapper(torch.nn.Module):
    """
    Wraps SAM3VLBackbone, ViT and FPN
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor]:
        output = self.backbone.forward_image(pixel_values)
        fpn = output["backbone_fpn"]     # list of 4 (B, C, H, W)
        pos = output["vision_pos_enc"]   # list of 4 (B, C', H, W)
        return fpn[0], fpn[1], fpn[2],  pos[0], pos[1], pos[2]

def _apply_rope_real(self, q, k):
    def rope(x):
        orig_dtype = x.dtype
        x = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_r, x_i = x[..., 0], x[..., 1]
        o_r = x_r * self.freqs_real - x_i * self.freqs_imag
        o_i = x_r * self.freqs_imag + x_i * self.freqs_real
        return torch.stack((o_r, o_i), dim=-1).flatten(-2).to(orig_dtype)
    return rope(q), rope(k)


def patch_rope(model):
    """Replace complex-arithmetic RoPE with the real equivalent for TRT export."""
    blocks = model.backbone.vision_backbone.trunk.blocks
    for blk in blocks:
        f = blk.attn.freqs_cis
        blk.attn.register_buffer("freqs_real", f.real.contiguous(), persistent=False)
        blk.attn.register_buffer("freqs_imag", f.imag.contiguous(), persistent=False)
    type(blocks[0].attn)._apply_rope = _apply_rope_real



@torch.inference_mode
def trace_and_export_image_encoder(model : torch.nn.Module, input : Any, engine_path : str, fp16 : bool) -> Tuple[torch.Tensor]: 
    wrapper = ImageEncoderWrapper(model).to(DEVICE).eval()

    predictor.setup_source(input)

    for batch in predictor.dataset:
        input = predictor.preprocess(batch[1])
        break
    
    print("[ImageEncoderExport] Tracing Image Encoder Model")
    torch_output = wrapper(input) # warm up
    exp_program = torch.export.export(wrapper, (input,), strict=False)

    precision = "fp16" if fp16 else "fp32"
    print(f"[ImageEncoderExport] Building TensorRT engine ({precision})")
    dtype = torch.float16 if fp16 else torch.float
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exp_program,
        arg_inputs=[input],
        optimization_level=5,
        #use_explicit_typing=True,
        enabled_precisions={dtype, torch.float32},
        device=torch_tensorrt.Device(f"cuda:{input.device.index or 0}"),
    )

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    return torch_output
    
@torch.inference_mode()
def _verify_wrapper(predictor, image):
    """Compare ImageEncoderWrapper outputs against predictor.set_image features."""
    print("[ImageEncoderExport] Verifying ultralytics wrapper")
    wrapper = ImageEncoderWrapper(predictor.model).to(DEVICE).eval()

    predictor.set_image(image)
    ref = predictor.features
    ref_tensors = list(ref["backbone_fpn"]) + list(ref["vision_pos_enc"])

    for batch in predictor.dataset:
        im = predictor.preprocess(batch[1])
        break
    got = wrapper(im)

    assert len(got) == len(ref_tensors), f"{len(got)} outputs, expected {len(ref_tensors)}"
    for i, (r, g) in enumerate(zip(ref_tensors, got)):
        assert r.shape == g.shape, f"[{i}] shape {tuple(g.shape)} != {tuple(r.shape)}"
        print(f"[{i}] {tuple(g.shape)}  max_abs={(r.float() - g.float()).abs().max().item():.3e}")

    del wrapper
    return got

@torch.inference_mode()
def _verify_engine(engine_path: str, input: torch.Tensor, torch_output: Tuple[torch.Tensor], min_cos: float = 0.9999) -> bool:
    """Run the serialized engine and compare its outputs against the eager wrapper."""
    print("[ImageEncoderExport] Verifying TRT Engine")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    assert len(inputs) == 1, f"expected 1 engine input, got {inputs}"
    assert len(outputs) == len(torch_output), (
        f"engine has {len(outputs)} outputs, wrapper returned {len(torch_output)}"
    )

    # bind input
    x = input.to(_TRT_TO_TORCH[engine.get_tensor_dtype(inputs[0])]).contiguous()
    ctx.set_input_shape(inputs[0], tuple(x.shape))
    ctx.set_tensor_address(inputs[0], x.data_ptr())

    # allocate + bind outputs
    trt_out = []
    for name in outputs:
        buf = torch.empty(
            tuple(ctx.get_tensor_shape(name)),
            dtype=_TRT_TO_TORCH[engine.get_tensor_dtype(name)],
            device=x.device,
        ).contiguous()
        ctx.set_tensor_address(name, buf.data_ptr())
        trt_out.append(buf)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert ctx.execute_async_v3(stream.cuda_stream), "engine execution failed"
    stream.synchronize()
    ok = True
    for i, (name, ref, got) in enumerate(zip(outputs, torch_output, trt_out)):
        if ref.shape != got.shape:
            print(f"[{i}] {name}  SHAPE {tuple(got.shape)} != {tuple(ref.shape)}")
            ok = False
            continue
        a, b = ref.float(), got.float()
        max_abs = (a - b).abs().max().item()
        scale = a.abs().max().item()
        max_rel = max_abs / scale if scale > 0 else 0.0
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        passed = cos >= min_cos
        ok &= passed
        print(f"[{i}] {name:<10} {str(tuple(got.shape)):<22} "
              f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  cos={cos:.6f}  "
              f"{'OK' if passed else 'FAIL'}")
    for name, t in zip(outputs, trt_out):
        t = t.float()
        print(name, "nan:", torch.isnan(t).sum().item(), "inf:", torch.isinf(t).sum().item(),
              "absmax:", t.abs().max().item())
    for i, t in enumerate(torch_output):
        print("torch", i, "absmax:", t.float().abs().max().item())
    print(f"engine {'MATCHES' if ok else 'DIFFERS FROM'} torch within min_cos={min_cos}")

    model_fp32 = copy.deepcopy(predictor.model).float()
    ref32 = ImageEncoderWrapper(model_fp32).eval()(im.float())

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

    for i in range(3):
        print(f"[{i}] torch_fp16 vs fp32: {cos(ref32[i], torch_output[i]):.6f}   "
              f"trt_fp16 vs fp32: {cos(ref32[i], trt_out[i]):.6f}")

    return ok

def _construct_model(fp16 : bool) -> SAM3SemanticPredictor:
    half = False
    quantize = 32
    if fp16:
        half = True
        quantize = 16

    overrides = dict(
        conf=0.2,
        task="segment",
        mode="predict",
        model=os.path.join(os.environ["HOME"],"models","sam3.pt"),
        half=half,
        quantize=quantize,
        compile=False,
        save=False,
        device="cuda",
    )

    return SAM3SemanticPredictor(overrides=overrides)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="build the engine with fp16 enabled")
    args = parser.parse_args()

    predictor = _construct_model(args.fp16)

    predictor.setup_model(model=None, verbose=False)
    img = cv2.imread("test.png")
    predictor.setup_source(img)
    
    patch_rope(predictor.model)
    torch_output = _verify_wrapper(predictor, img)

    precision = "fp16" if args.fp16 else "fp32"
    engine_path = os.path.join(os.environ["HOME"], "models", f"image_encoder_{precision}.engine")
    #_ = trace_and_export_image_encoder(
    #    predictor.model,
    #    img,
    #    engine_path,
    #    args.fp16
    #)

    for batch in predictor.dataset:
        im = predictor.preprocess(batch[1])
        break
    _verify_engine(engine_path, im, torch_output, min_cos=0.999 if args.fp16 else 0.9999) 

