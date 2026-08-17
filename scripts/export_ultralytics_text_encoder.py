
import os
import cv2
import torch
import torch_tensorrt

from ultralytics.models.sam import SAM3SemanticPredictor

from typing import Tuple, Any

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
def _verify_wrapper(predictor, wrapper, captions):
    """Compare TextEncoderWrapper outputs against backbone.forward_text."""
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

@torch.inference_mode
def trace_and_export_text_encoder(model : torch.nn.Module, input : Any, engine_path : str) -> Tuple[torch.Tensor]:
    wrapper = TextEncoderWrapper(model).to(DEVICE).eval()

    print("[TextEncoderExport] Tracing Text Encoder Model")
    torch_output = wrapper(input[0].to(DEVICE), input[1].to(DEVICE))

    exp_program = torch.export.export(wrapper, (input[0], input[1]), strict=False)

    print("[TextEncoderExport] Building TensorRT engine (fp16, fp8)")
    
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
        exp_program,
        arg_inputs=[input[0].to(DEVICE), input[1].to(DEVICE)],
        optimization_level=5,
        enabled_precisions={torch.float8_e4m3fn,torch.float16},
        device=torch_tensorrt.Device(f"cuda:0"),
    )

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    return torch_output

def _construct_model() -> SAM3SemanticPredictor:
    overrides = dict(
        conf=0.2,
        task="segment",
        mode="predict",
        model=os.path.join(os.environ["HOME"],"models","sam3.pt"),
        half=True,
        quantize=16,
        compile=False,
        save=False,
        device=DEVICE,
    )

    return SAM3SemanticPredictor(overrides=overrides)

if __name__ == "__main__":
    captions = ["road"]

    predictor = _construct_model()

    predictor.setup_model(model=None, verbose=False)
    #wrapper = TextEncoderWrapper(predictor.model)

    tokenizer = predictor.model.backbone.language_backbone.tokenizer
    ctx = predictor.model.backbone.language_backbone.context_length

    input_ids = tokenizer(captions, context_length=ctx).to(DEVICE)
    attention_mask = (input_ids != 0).bool().ne(1)

    #_verify_wrapper(predictor, wrapper, captions)

    engine_path = os.path.join(os.environ["HOME"], "models", "text_encoder.engine")
    trace_and_export_text_encoder(predictor.model, (input_ids, attention_mask), engine_path)
