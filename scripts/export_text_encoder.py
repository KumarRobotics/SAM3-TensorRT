import os
import cv2
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any

from transformers import Sam3Model, Sam3Processor, Sam3Config
from huggingface_hub import login
import onnx
import onnxruntime as ort

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

DEVICE = torch.device("cpu")

OUTPUT_NAMES=["text_embeddings"]
INPUT_NAME=["input_ids", "attention_mask"]

OPSET = 18

class TextEncoderWrapper(torch.nn.Module):
    """
    Wraps CLIPTextModelWithProjection and SAM3 text_projection.
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.text_encoder    = model.text_encoder    # CLIPTextModelWithProjection
        self.text_projection = model.text_projection  # Linear(1024, 256)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, T, 1024) -> project all tokens -> (B, T, 2V56)
        text_embeds = self.text_projection(output.last_hidden_state)
        return text_embeds

def _report(name, out, ref, atol):
    out = out.detach().float().cpu()
    ref = ref.detach().float().cpu()
    if out.shape != ref.shape:
        print(f"  {name}: SHAPE MISMATCH wrapper={tuple(out.shape)} ref={tuple(ref.shape)}")
        return False
    d = (out - ref).abs().max().item()
    print(f"  {name}: max|diff|={d:.3e}  {'OK' if d < atol else 'FAIL'}")
    return d < atol

def validate_wrapper(model, input, atol : float = 1e-2) -> bool:
    model_cpu = model.to('cpu')
    wrapper = TextEncoderWrapper(model_cpu).to('cpu')
    input_cpu = input.to('cpu')
    captured = {}
    h = model.text_projection.register_forward_hook(
        lambda mod, inp, out: captured.update(y=out)
    )
    try:
        with torch.inference_mode():
            _ = model(**input_cpu)
    finally:
        h.remove()

    ref = captured["y"].unsqueeze(0)

    with torch.inference_mode():
        wrapper_output = wrapper(input.input_ids.to('cpu'), input.attention_mask.to('cpu')) 

    ref_embeds = ref.text_embeds if hasattr(ref, "text_embeds") else ref[0]
    print(f"  wrapper={tuple(wrapper_output.shape)} ref={tuple(ref_embeds.shape)}")
    ok = _report("text_embeddings", wrapper_output, ref_embeds, atol)
    print(f"[TextEncoder] {'PASS' if ok else 'FAIL'}")
    return ok

def validate_onnx(wrapper: torch.nn.Module, input: Any, onnx_path: Path, atol: float = 1e-2) -> bool:
    print("[TextEncoderWrapper] Validating onnx export")
    wrapper = wrapper.to('cpu')
    ids, mask = input[0].cpu(), input[1].cpu()
    with torch.inference_mode():
        golden = wrapper(ids, mask)

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda" else ["CPUExecutionProvider"],
    )
    ort_outs = sess.run(OUTPUT_NAMES, {
        "input_ids": ids.numpy(),
        "attention_mask": mask.numpy(),
    })

    ok = True
    for name, g, o in zip(OUTPUT_NAMES, golden, ort_outs):
        g = g.detach().float().cpu().numpy()
        max_diff = float(np.abs(g - o.astype(np.float32)).max())
        close = max_diff < atol
        ok &= close
        print(f"  {name}: max|diff|={max_diff:.3e}  {'OK' if close else 'FAIL'}")

    print(f"[TextEncoderExport] Passed Tests: {ok}")

    return ok

def text_encoder_forward(model : torch.nn.Module, input : Any) -> torch.Tensor:
    wrapper = TextEncoderWrapper(model).to(input[0].device).eval()
    with torch.inference_mode():
        torch_output = wrapper(input[0], input[1])
    return torch_output

def trace_and_export_text_encoder(model : torch.nn.Module, input : Any, onnx_path) -> torch.Tensor:
    wrapper = TextEncoderWrapper(model).to(DEVICE).half().eval()
    print("[TextEncoderExport] Tracing Text Encoder Model")
    with torch.inference_mode():
        torch_output = wrapper(input[0].half(), input[1].half())

    input = (input[0].half(), input[1].half())

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            input,
            str(onnx_path),
            input_names=INPUT_NAME,
            output_names=OUTPUT_NAMES,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamp=True,
            dynamic_axes=None,
        )
 
    onnx.checker.check_model(str(onnx_path))
    
    assert validate_onnx(wrapper, input, onnx_path)

    return torch_output
    

if __name__ == "__main__": 
    login(token=HF_KEY)
    print(f"[TextEncoderExport] Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    config = Sam3Config.from_pretrained(MODEL_ID, token=HF_KEY)
    config.image_size = 644
    model = Sam3Model.from_pretrained(MODEL_ID, config=config, token=HF_KEY)
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")

    processor.image_processor.size = {"height": 644, "width": 644}
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)
    validate_wrapper(model, inp)
    print(f"[ImageEncoderExport] Tracing with input_ids size {inp.input_ids.shape}, and attention_mask size {inp.attention_mask.shape}")
    
    trace_and_export_text_encoder(model, (inp.input_ids, inp.attention_mask), MODELS_DIR / "text_encoder.onnx") 


