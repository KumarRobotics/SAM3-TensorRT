
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
    from transformers import Sam3Model, Sam3Processor
    import onnx
    import onnxruntime as ort
    from huggingface_hub import login
except ImportError:
    print("Error: transformers>=4.47.0, onnx and onnxruntime is required for SAM3 export.")
    print("We recommend using the sam3-tensorrt:trace dockerfile.")
    sys.exit(1)

from sam3trt.export_image_encoder import trace_and_export_image_encoder
from sam3trt.export_text_encoder import trace_and_export_text_encoder 

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


if __name__ == "__main__":
    login(token=HF_KEY)
    print(f"Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    
    model = Sam3Model.from_pretrained(MODEL_ID, token=HF_KEY)
    processor = Sam3Processor.from_pretrained(MODEL_ID, target_size=644, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)

    image_output = trace_and_export_image_encoder(model, inp.pixel_values, MODELS_DIR / "image_encoder.onnx")
