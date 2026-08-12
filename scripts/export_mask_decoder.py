"""

"""
import torch 

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

from transformers import Sam3Model, Sam3Processor, Sam3Config
import onnx
import onnxruntime as ort
from huggingface_hub import login

from export_image_encoder import image_encoder_forward
from export_text_encoder import text_encoder_forward


MODEL_ID   = "facebook/sam3"
HF_KEY     = os.environ.get("HF_KEY")
REPO_ROOT  = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "models"
CONFIG_DIR = REPO_ROOT / "config"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

INPUT_NAMES = [f"fpn_{i}" for i in range(4)] + [f"fpn_pos_{j}" for j in range(4)] + ["text_embeds", "attention_mask", "original_sizes"]
OUTPUT_NAMES = ["predicted_masks", "predicted_logits", "presence_logits"]
OPSET = 18

class MaskDecoderWrapper(torch.nn.Module):
    """
    Wraps DETR encoder + DETR decoder + mask head + dot-product scoring.
    Designed for text-only inference (no geometry prompts).
    """
    def __init__(self, model: Sam3Model):
        super().__init__()
        self.detr_encoder      = model.detr_encoder
        self.detr_decoder      = model.detr_decoder
        self.mask_decoder      = model.mask_decoder
        self.dot_product_scoring = model.dot_product_scoring

    def forward(
        self,
        fpn_0:    torch.Tensor,
        fpn_1:    torch.Tensor,
        fpn_2:    torch.Tensor,
        fpn_3:    torch.Tensor,
        fpn_pos_0: torch.Tensor,
        fpn_pos_1: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        fpn_pos_3: torch.Tensor,
        text_embeds:    torch.Tensor,
        attention_mask: torch.Tensor,
        original_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        enc_out = self.detr_encoder(
            vision_features=[fpn_2],
            text_features=text_embeds,
            vision_pos_embeds=[fpn_pos_2],
            text_mask=attention_mask.bool(),
            original_sizes=original_sizes,
        )

        dec_out = self.detr_decoder(
            vision_features=enc_out.last_hidden_state,
            text_features=text_embeds,
            vision_pos_encoding=enc_out.pos_embeds_flattened,
            text_mask=attention_mask.bool(),
            spatial_shapes=enc_out.spatial_shapes,
            original_sizes=original_sizes
        )

        scores = self.dot_product_scoring(
            decoder_hidden_states=dec_out.intermediate_hidden_states,
            text_features=text_embeds,
            text_mask=attention_mask.bool(),
        )
        n_layers    = scores.shape[0]
        pred_logits = scores[n_layers - 1, :, :, 0]

        presence_logits = dec_out.presence_logits
        if presence_logits.dim() == 3:
            presence_logits = presence_logits[-1]

        n_int        = dec_out.intermediate_hidden_states.shape[0]
        last_queries = dec_out.intermediate_hidden_states[n_int - 1]  # (B, Q, D)

        mask_out = self.mask_decoder(
            decoder_queries=last_queries,
            backbone_features=[fpn_0, fpn_1, fpn_2],
            encoder_hidden_states=enc_out.last_hidden_state,
            prompt_features=text_embeds,
            prompt_mask=attention_mask.bool(),
            original_sizes=original_sizes,
        )

        return mask_out.pred_masks, pred_logits, presence_logits

def _report(name, out, ref, atol):
    out = out.detach().float().cpu()
    ref = ref.detach().float().cpu()
    if out.shape != ref.shape:
        print(f"  {name}: SHAPE MISMATCH wrapper={tuple(out.shape)} ref={tuple(ref.shape)}")
        return False
    d = (out - ref).abs().max().item()
    print(f"  {name}: max|diff|={d:.3e}  {'OK' if d < atol else 'FAIL'}")
    return d < atol

def validate_wrapper(model, inp, wrapper, wrapper_inputs, atol=1e-3) -> bool:
    model = model.to("cpu").eval()
    inp = {k: v.to("cpu") for k, v in inp.items()}
    wrapper = wrapper.to("cpu").eval()
    wrapper_inputs = tuple(x.to("cpu") for x in wrapper_inputs)

    with torch.inference_mode():
        ref = model(**inp)
        w_masks, w_logits, w_presence = wrapper(*wrapper_inputs)

    ok = True
    ok &= _report("pred_masks",   w_masks,    ref.pred_masks,      atol)
    ok &= _report("pred_logits",  w_logits,   ref.pred_logits,     atol)
    ok &= _report("presence",     w_presence, ref.presence_logits, atol)

    print(f"[MaskDecoderWrapper] Passed Tests: {ok}")
    return ok


def validate_onnx(wrapper : torch.nn.Module, input : Tuple[Any], onnx_path : Path, atol : float = 1e-2) -> bool:
    print("[MaskDecoderWrapper] Validating onnx export")
    wrapper = wrapper.to('cpu')
    with torch.inference_mode():
        golden = wrapper(input[0].to('cpu'),
                         input[1].to('cpu'),
                         input[2].to('cpu'),
                         input[3].to('cpu'),
                         input[4].to('cpu'),
                         input[5].to('cpu'),
                         input[6].to('cpu'),
                         input[7].to('cpu'),
                         input[8].to('cpu'),
                         input[9].to('cpu'),
                         input[10].to('cpu'))

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda" else ["CPUExecutionProvider"],
    )
    ort_outs = sess.run(OUTPUT_NAMES, {
        'fpn_0': input[0].to('cpu').numpy(),
        'fpn_1': input[1].to('cpu').numpy(),
        'fpn_2': input[2].to('cpu').numpy(),
        'fpn_3': input[3].to('cpu').numpy(),
        'fpn_pos_0': input[4].to('cpu').numpy(),
        'fpn_pos_1': input[5].to('cpu').numpy(),
        'fpn_pos_2': input[6].to('cpu').numpy(),
        'fpn_pos_3': input[7].to('cpu').numpy(),
        'text_embeds': input[8].to('cpu').numpy(),
        'attention_mask': input[9].to('cpu').numpy(),
        'original_sizes': input[10].to('cpu').numpy()
    })

    ok = True
    for name, g, o in zip(OUTPUT_NAMES, golden, ort_outs):
        g = g.detach().float().cpu().numpy()
        max_diff = float(np.abs(g - o.astype(np.float32)).max())
        close = max_diff < atol
        ok &= close
        print(f"  {name}: max|diff|={max_diff:.3e}  {'OK' if close else 'FAIL'}")

    print(f"[MaskDecoderExport] Passed Tests: {ok}")

    return ok

def trace_and_export_mask_decoder(model : torch.nn.Module, input : Any, onnx_path) -> Tuple[torch.Tensor]:
    wrapper = MaskDecoderWrapper(model).to(DEVICE).half().eval()
    print("[TextEncoderExport] Tracing Text Encoder Model")
    with torch.inference_mode():
        torch_output = wrapper(input[0].half(),
                               input[1].half(),
                               input[2].half(),
                               input[3].half(),
                               input[4].half(),
                               input[5].half(),
                               input[6].half(),
                               input[7].half(),
                               input[8].half(),
                               input[9].half(),
                               input[10].half())

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            input,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=True,
            dynamic_axes=None,
        )
 
    onnx.checker.check_model(str(onnx_path))
    
    assert validate_onnx(wrapper, input, onnx_path)

    return torch_output


if __name__ == "__main__": 
    login(token=HF_KEY)
    print(f"[MaskDecoderExport] Loading SAM3 ({MODEL_ID}) on {DEVICE}...")
    config = Sam3Config.from_pretrained(MODEL_ID, token=HF_KEY)
    config.image_size = 644
    model = Sam3Model.from_pretrained(MODEL_ID, config=config, token=HF_KEY)
    processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_KEY)

    model = model.to(device=DEVICE).eval()
    img = cv2.imread("test.png")

    processor.image_processor.size = {"height": 644, "width": 644}
    inp = processor(images=img, text="road", return_tensors="pt").to(DEVICE)
    
    img_embeds = image_encoder_forward(model.to('cpu'), inp.pixel_values.to('cpu'))
    txt_embeds = text_encoder_forward(model.to('cpu'), (inp.input_ids.to('cpu'), inp.attention_mask.to('cpu')))

    dec_inputs = img_embeds + (txt_embeds, inp.attention_mask, inp.original_sizes)
    wrapper = MaskDecoderWrapper(model).to('cpu')
    validate_wrapper(model, inp, wrapper, dec_inputs)

    output = trace_and_export_mask_decoder(model.to(DEVICE), dec_inputs, MODELS_DIR / "mask_decoder.onnx")
