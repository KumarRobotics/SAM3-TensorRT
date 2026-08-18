
import os
import cv2
import sys
import json
import yaml
import torch
import numpy as np
import argparse
from typing import Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.environ["HOME"], "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

from export_ultralytics_image_encoder import export_and_verify_image_encoder, _construct_model
from export_ultralytics_text_encoder import export_and_verify_text_encoder
from export_ultralytics_mask_decoder import export_and_verify_mask_decoder

if __name__ == "__main__":

    captions = ["road"]
    img = cv2.imread("test.png")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="build the engine with fp16 enabled")
    args = parser.parse_args()

    predictor = _construct_model(args.fp16)
    predictor.setup_model(model=None, verbose=False)

    img_feats = export_and_verify_image_encoder(predictor, args.fp16, img)
    txt_feats = export_and_verify_text_encoder(predictor, args.fp16, captions)

    txt_mask_f = torch.zeros_like(txt_feats[1], dtype=txt_feats[0].dtype).masked_fill_(txt_feats[1], -65504.0)
    input = img_feats + (txt_feats[0], txt_feats[1], txt_mask_f,)

    export_and_verify_mask_decoder(predictor, input, args.fp16, img, captions)


