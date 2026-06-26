
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
    print("Error: transformers>=4.47.0, onnx and onnxruntime is required for SAM3 export.")
    print("We recommend using the sam3-tensorrt:trace dockerfile.")
    sys.exit(1)

from export_image_encoder import trace_and_export_image_encoder
from export_text_encoder import trace_and_export_text_encoder 
from export_mask_decoder import trace_and_export_mask_decoder

MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

if __name__ == "__main__":
    login(token=HF_KEY)
    print(f"[ExportAll] Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    
    config = Sam3Config.from_pretrained(MODEL_ID, token=HF_KEY)
    config.image_size = 644
    model = Sam3Model.from_pretrained(MODEL_ID, config=config, token=HF_KEY).to(DEVICE)
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_KEY)
    
    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")
    processor.image_processor.size = {"height": 644, "width": 644}
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)

    image_output = trace_and_export_image_encoder(model, inp.pixel_values, MODELS_DIR / "image_encoder.onnx")
    text_output = trace_and_export_text_encoder(model, (inp.input_ids, inp.attention_mask), MODELS_DIR / "text_encoder.onnx")
    
    mask_inputs = image_output + (text_output, inp.attention_mask, inp.original_sizes)
    mask_output = trace_and_export_mask_decoder(model, mask_inputs, MODELS_DIR / "mask_decoder.onnx")
