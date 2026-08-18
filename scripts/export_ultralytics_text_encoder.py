
import os
import cv2
import torch
import argparse
import torch_tensorrt
import tensorrt as trt
from ultralytics.models.sam import SAM3SemanticPredictor

from typing import Tuple, Any

_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEncoderWrapper(torch.nn.Module):
    """
    Wraps VETextEncoder (TextTransformer + resizer), tokenizer split off.
    """
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.backbone.language_backbone  # VETextEncoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor]:
        encoder = self.text_encoder.encoder

        inputs_embeds = encoder.token_embedding(input_ids)          
        _, text_memory = encoder(input_ids)                         
        text_memory = text_memory.transpose(0, 1)                   
        text_memory_resized = self.text_encoder.resizer(text_memory) 

        return (
            text_memory_resized,            # language_features
            attention_mask,                 # language_mask
            inputs_embeds.transpose(0, 1),  # language_embeds
        )

@torch.inference_mode()
def _verify_wrapper(predictor, captions):
    """Compare TextEncoderWrapper outputs against backbone.forward_text."""
    wrapper = TextEncoderWrapper(predictor.model).to(DEVICE).eval()
    ref = predictor.model.backbone.forward_text(captions)
    ref_tensors = [ref["language_features"], ref["language_mask"], ref["language_embeds"]]

    tokenizer = predictor.model.backbone.language_backbone.tokenizer
    ctx = predictor.model.backbone.language_backbone.context_length
    input_ids = tokenizer(captions, context_length=ctx).to(predictor.device)
    attention_mask = (input_ids != 0).bool().ne(1)
    got = wrapper(input_ids, attention_mask)

    assert len(got) == len(ref_tensors), f"{len(got)} outputs, expected {len(ref_tensors)}"
    for i, (r, g) in enumerate(zip(ref_tensors, got)):
        assert r.shape == g.shape, f"[{i}] shape {tuple(g.shape)} != {tuple(r.shape)}"
        print(f"[{i}] {tuple(g.shape)}  max_abs={(r.float() - g.float()).abs().max().item():.3e}")
    
    del wrapper

    return got

@torch.inference_mode
def trace_and_export_text_encoder(model : torch.nn.Module, input : Any, engine_path : str, fp16: bool) -> Tuple[torch.Tensor]:
    wrapper = TextEncoderWrapper(model).to(DEVICE).eval()

    print("[TextEncoderExport] Tracing Text Encoder Model")
    torch_output = wrapper(input[0].to(DEVICE), input[1].to(DEVICE))

    exp_program = torch.export.export(wrapper, (input[0], input[1]), strict=False)

    precision = "fp16" if fp16 else "fp32"
    print(f"[TextEncoderExport] Building TensorRT engine ({precision})") 

    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exp_program,
        arg_inputs=[input[0].to(DEVICE), input[1].to(DEVICE)],
        optimization_level=5,
        use_explicit_typeing=True,
        device=torch_tensorrt.Device(f"cuda:0"),
    )

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    return torch_output

@torch.inference_mode()
def _verify_engine(engine_path: str, inputs_t: Tuple[torch.Tensor], torch_output: Tuple[torch.Tensor],
                   min_cos: float = 0.9999) -> bool:
    """Run the serialized engine and compare its outputs against the eager wrapper."""
    print("[TextEncoderExport] Verifying TRT Engine")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    assert len(in_names) == len(inputs_t), f"engine has {len(in_names)} inputs, got {len(inputs_t)}"
    assert len(out_names) == len(torch_output), (
        f"engine has {len(out_names)} outputs, wrapper returned {len(torch_output)}"
    )

    # bind inputs — keep refs alive so the buffers aren't freed before execute
    bound = []
    for name, t in zip(in_names, inputs_t):
        x = t.to(_TRT_TO_TORCH[engine.get_tensor_dtype(name)]).contiguous()
        ctx.set_input_shape(name, tuple(x.shape))
        ctx.set_tensor_address(name, x.data_ptr())
        bound.append(x)
        print(f"  in  {name:<16} {str(tuple(x.shape)):<18} {x.dtype}")

    # allocate + bind outputs
    device = bound[0].device
    trt_out = []
    for name in out_names:
        buf = torch.empty(
            tuple(ctx.get_tensor_shape(name)),
            dtype=_TRT_TO_TORCH[engine.get_tensor_dtype(name)],
            device=device,
        ).contiguous()
        ctx.set_tensor_address(name, buf.data_ptr())
        trt_out.append(buf)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert ctx.execute_async_v3(stream.cuda_stream), "engine execution failed"
    stream.synchronize()

    ok = True
    for i, (name, ref, got) in enumerate(zip(out_names, torch_output, trt_out)):
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
              f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  cos={cos:.8f}  "
              f"{'OK' if passed else 'FAIL'}")

    print(f"engine {'MATCHES' if ok else 'DIFFERS FROM'} torch within min_cos={min_cos}")
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
        device=DEVICE,
    )

    return SAM3SemanticPredictor(overrides=overrides)

if __name__ == "__main__":
    captions = ["road"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="build the engine with fp16 enabled")
    args = parser.parse_args()

    predictor = _construct_model(args.fp16)
    predictor.setup_model(model=None, verbose=False)

    tokenizer = predictor.model.backbone.language_backbone.tokenizer
    ctx = predictor.model.backbone.language_backbone.context_length

    input_ids = tokenizer(captions, context_length=ctx).to(DEVICE)
    attention_mask = (input_ids != 0).bool().ne(1)

    torch_output = _verify_wrapper(predictor, captions)

    precision = "fp16" if args.fp16 else "fp32"
    engine_path = os.path.join(os.environ["HOME"], "models", f"text_encoder_{precision}.engine")
    trace_and_export_text_encoder(predictor.model, (input_ids, attention_mask), engine_path, args.fp16)

    _verify_engine(engine_path, (input_ids, attention_mask), (torch_output[0], torch_output[2]), 0.999 if args.fp16 else 0.9999)
