import os
import cv2
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import Sam3Model, Sam3Processor, Sam3Config
    import onnx
    import onnxruntime as ort
    from huggingface_hub import login
except ImportError:
    print("Error: transformers>=4.47.0 and onnx and onnxruntime is required for SAM3 support.")
    print("We recommend using the sam3-tensorrt:trace dockerfile.")
    sys.exit(1)

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
EXPORT_DTYPE = torch.float32

OUTPUT_NAMES = [f"fpn_{i}" for i in range(4)] + [f"pos_{i}" for i in range(4)]
INPUT_NAME = "pixel_values"

OPSET = 18

class ImageEncoderWrapper(torch.nn.Module):
    """
    Wraps Sam3VisionModel, ViT and FPN
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.vision_encoder = model.vision_encoder

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor]:
        output = self.vision_encoder(pixel_values)
        fpn = output.fpn_hidden_states      # tuple of 4 (B, C, H, W)
        pos = output.fpn_position_encoding  # tuple of 4 (B, C', H, W)
        return fpn[0], fpn[1], fpn[2], fpn[3], pos[0], pos[1], pos[2], pos[3]

def validate(wrapper: torch.nn.Module, input: Any, onnx_path: Path, atol: float = 1e-2) -> bool:
    print("[ImageEncoderExport] Validating onnx export")
    wrapper = wrapper.to('cpu')
    with torch.inference_mode():
        golden = wrapper(input.to('cpu'))

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda" else ["CPUExecutionProvider"],
    )
    ort_outs = sess.run(OUTPUT_NAMES, {INPUT_NAME: input.detach().cpu().numpy()})

    ok = True
    for name, g, o in zip(OUTPUT_NAMES, golden, ort_outs):
        g = g.detach().float().cpu().numpy()
        max_diff = float(np.abs(g - o.astype(np.float32)).max())
        close = max_diff < atol
        ok &= close
        print(f"  {name}: max|diff|={max_diff:.3e}  {'OK' if close else 'FAIL'}")

    print(f"[ImageEncoderExport] Passed Tests: {ok}")
    return ok

def trace_and_export_image_encoder(model : torch.nn.Module, input : Any, onnx_path : Path) -> torch.Tensor:
    wrapper = ImageEncoderWrapper(model).to(DEVICE).eval()
    print("[ImageEncoderExport] Tracing Image Encoder Model")
    # do an initial forward pass of the model 
    with torch.inference_mode():
        torch_output = wrapper(input)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input,),
            str(onnx_path),
            input_names=[INPUT_NAME],
            output_names=OUTPUT_NAMES,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,
        )
 
    onnx.checker.check_model(str(onnx_path))

    assert validate(wrapper, input, onnx_path)

    return torch_output

if __name__ == "__main__":
    login(token=HF_KEY)
    print(f"[ImageEncoderExport] Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    config = Sam3Config.from_pretrained(MODEL_ID, token=HF_KEY)
    config.image_size = 644
    model = Sam3Model.from_pretrained(MODEL_ID, config=config, token=HF_KEY)
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")

    processor.image_processor.size = {"height": 644, "width": 644}
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)
    print(f"[ImageEncoderExport] Tracing with image size {inp.pixel_values.shape}")
    trace_and_export_image_encoder(model, inp.pixel_values, MODELS_DIR / "image_encoder.onnx")


