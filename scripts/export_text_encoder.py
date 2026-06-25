import os
import cv2
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any

from transformers import Sam3Model, Sam3Processor
from huggingface_hub import login
import onnx
import onnxruntime as ort

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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

def validate(wrapper: torch.nn.Module, input: Any, onnx_path: Path, atol: float = 1e-2) -> bool:
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

def trace_and_export_text_encoder(model : torch.nn.Module, input : Any, onnx_path) -> torch.Tensor:
    wrapper = TextEncoderWrapper(model)
    print("[TextEncoderExport] Tracing Text Encoder Model")
    with torch.inference_mode():
        torch_output = wrapper(input[0], input[1])

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            input,
            str(onnx_path),
            input_names=INPUT_NAME,
            output_names=OUTPUT_NAMES,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
        )
 
    onnx.checker.check_model(str(onnx_path))
    
    assert validate(wrapper, input, onnx_path)
    

if __name__ == "__main__": 
    login(token=HF_KEY)
    print(f"Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    
    model = Sam3Model.from_pretrained(MODEL_ID, token=HF_KEY)
    processor = Sam3Processor.from_pretrained(MODEL_ID, target_size=644, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)

    trace_and_export_text_encoder(model, (inp.input_ids, inp.attention_mask), MODELS_DIR / "text_encoder.onnx") 


