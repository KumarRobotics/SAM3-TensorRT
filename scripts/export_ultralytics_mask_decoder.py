import os
import cv2
import time
import torch
import argparse
import torch_tensorrt
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.utils.ops import xywh2xyxy
from typing import Tuple, List, Any
import tensorrt as trt
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


def _prepare_backbone_features(backbone_out, num_feature_levels, batch=1):
    """Prepare and flatten visual features from the image backbone output for further processing."""
    if batch > 1:  # expand features if there's more than one prompt
        backbone_out = {
            **backbone_out,
            "backbone_fpn": [feat.expand(batch, -1, -1, -1) for feat in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [pos.expand(batch, -1, -1, -1) for pos in backbone_out["vision_pos_enc"]],
        }
    assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
    assert len(backbone_out["backbone_fpn"]) >= num_feature_levels

    feature_maps = backbone_out["backbone_fpn"][-num_feature_levels :]
    vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels :]

    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    # flatten NxCxHxW to HWxNxC
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
    return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


class FusionEncoderWrapper(torch.nn.Module):
    """Stage 1: backbone feature prep + transformer encoder (text/image fusion)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fpn0, fpn1, fpn2, pos0, pos1, pos2, txt_feats, txt_masks, txt_masks_f):
        # txt_feats: (L, nc, C) already indexed by text_ids; txt_masks: (nc, L) True == pad
        backbone_out = {
            "vision_features": fpn2,
            "vision_pos_enc": [pos0, pos1, pos2],
            "backbone_fpn": [fpn0, fpn1, fpn2],
            "sam2_backbone_out": None,
        }
        nc = txt_masks.shape[0]
        _, img_feats, img_pos_embeds, vis_feat_sizes = _prepare_backbone_features(
            backbone_out, self.model.num_feature_levels, batch=nc
        )
        encoder_out = self.model._run_encoder(
            img_feats, img_pos_embeds, vis_feat_sizes, txt_feats, txt_masks
        )
        return (
            encoder_out["encoder_hidden_states"],  # memory
            encoder_out["pos_embed"],
            encoder_out["spatial_shapes"],
            encoder_out["valid_ratios"],
        )


class DetectionDecoderWrapper(torch.nn.Module):
    """Stage 2: query decoder + score/box heads."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory, pos_embed, spatial_shapes, valid_ratios, txt_feats, txt_masks):
        out, hs = self.model._run_decoder(
            pos_embed=pos_embed,
            memory=memory,
            src_mask=None,
            out={},
            prompt=txt_feats,
            prompt_mask=txt_masks,
            encoder_out={"spatial_shapes": spatial_shapes, "valid_ratios": valid_ratios},
        )
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_boxes_xyxy"],
            out["presence_logit_dec"],
            hs, 
        )


class SegmentationHeadWrapper(torch.nn.Module):
    """Stage 3: mask head over the full FPN."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, fpn0, fpn1, fpn2, hs, memory, txt_feats, txt_masks):
        num_o2o = hs.size(2)
        obj_queries = hs if self.model.o2m_mask_predict else hs[:, :, :num_o2o]
        seg_out = self.model.segmentation_head(
            backbone_feats=[fpn0, fpn1, fpn2],
            obj_queries=obj_queries,
            encoder_hidden_states=memory,
            prompt=txt_feats,
            prompt_mask=txt_masks,
        )
        return seg_out["pred_masks"][:, :num_o2o]

class MaskDecoderWrapper(torch.nn.Module):
    """
    Composes the three stages of SAM3SemanticModel.forward_grounding:
    fusion encoder -> detection decoder -> segmentation head.
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = FusionEncoderWrapper(model)
        self.detector = DetectionDecoderWrapper(model)
        self.segmenter = SegmentationHeadWrapper(model)

    def forward(
        self,
        fpn0: torch.Tensor, fpn1: torch.Tensor, fpn2: torch.Tensor,
        pos0: torch.Tensor, pos1: torch.Tensor, pos2: torch.Tensor,
        txt_feats: torch.Tensor,
        txt_masks: torch.Tensor,
        txt_masks_f: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        memory, pos_embed, spatial_shapes, valid_ratios = self.encoder(
            fpn0, fpn1, fpn2, pos0, pos1, pos2, txt_feats, txt_masks, txt_masks_f
        )
        pred_logits, pred_boxes, pred_boxes_xyxy, presence_logit_dec, hs = self.detector(
            memory, pos_embed, spatial_shapes, valid_ratios, txt_feats, txt_masks
        )
        pred_masks = self.segmenter(fpn0, fpn1, fpn2, hs, memory, txt_feats, txt_masks_f)

        return pred_logits, pred_boxes, pred_boxes_xyxy, presence_logit_dec, pred_masks

def time_model(predictor, image):
    for i in range(10):
        start = time.perf_counter()
        predictor.set_image(image)
        end = time.perf_counter()
        print("time: ", end - start)

@torch.inference_mode()
def _verify_wrapper(predictor : SAM3SemanticPredictor, input : Tuple[torch.Tensor], img : np.ndarray, captions : List[str]):
    """Compare DecoderWrapper against the stock predictor Results."""
    decoder_wrapper = MaskDecoderWrapper(predictor.model)
    predictor.set_image(img)
    ref = predictor(text=captions)[0] 
    for batch in predictor.dataset:
        im = predictor.preprocess(batch[1])
        break

    pred_logits, pred_boxes, _, presence, pred_masks = decoder_wrapper(*input)

    preds = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
        "presence_logit_dec": presence,
    }
    got = predictor.postprocess(preds, im, [img])[0]
    assert got.boxes.data.shape == ref.boxes.data.shape, (
        f"boxes {tuple(got.boxes.data.shape)} != {tuple(ref.boxes.data.shape)}"
    )
    print(f"boxes {tuple(got.boxes.data.shape)}  "
          f"max_abs={(ref.boxes.data - got.boxes.data).abs().max().item():.3e}")

    if ref.masks is not None:
        assert got.masks.data.shape == ref.masks.data.shape, (
            f"masks {tuple(got.masks.data.shape)} != {tuple(ref.masks.data.shape)}"
        )
        diff = (ref.masks.data.bool() ^ got.masks.data.bool()).sum().item()
        print(f"masks {tuple(got.masks.data.shape)}  differing_px={diff}")

    return got 

def _get_rpb_matrix_static(self, reference_boxes, feat_size):
    """feat_size ignored; H/W baked in from compilable_stored_size."""
    coords_h, coords_w = self.compilable_cord_cache
    zero = reference_boxes.sum() * 0
    coords_h = coords_h + zero
    coords_w = coords_w + zero
    boxes_xyxy = xywh2xyxy(reference_boxes).transpose(0, 1)
    bs, num_queries, _ = boxes_xyxy.shape

    deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
    deltas_y = deltas_y.view(bs, num_queries, -1, 2)
    deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
    deltas_x = deltas_x.view(bs, num_queries, -1, 2)

    if self.boxRPB in ("log", "both"):
        dxl = deltas_x * 8
        dxl = torch.sign(dxl) * torch.log2(torch.abs(dxl) + 1.0) / np.log2(8)
        dyl = deltas_y * 8
        dyl = torch.sign(dyl) * torch.log2(torch.abs(dyl) + 1.0) / np.log2(8)
        if self.boxRPB == "log":
            deltas_x, deltas_y = dxl, dyl
        else:
            deltas_x = torch.cat([deltas_x, dxl], dim=-1)
            deltas_y = torch.cat([deltas_y, dyl], dim=-1)

    deltas_x = self.boxRPB_embed_x(x=deltas_x)
    deltas_y = self.boxRPB_embed_y(x=deltas_y)
    B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
    return B.flatten(2, 3).permute(0, 3, 1, 2).contiguous()


def patch_rpb(model, H, W):
    dec = model.transformer.decoder
    p = next(model.parameters())
    dec.compilable_cord_cache = dec._get_coords(H, W, p.device, p.dtype)  # real ints
    dec.compilable_stored_size = (H, W)
    type(dec)._get_rpb_matrix = _get_rpb_matrix_static

@torch.inference_mode
def trace_and_export_mask_deocder(model : torch.nn.Module, input : Any, engine_path : str, fp16 : bool) -> Tuple[torch.Tensor]:

    wrapper = MaskDecoderWrapper(model).to(DEVICE).eval()
    
    fusion_wrapper = FusionEncoderWrapper(model).to(DEVICE).eval()
    print("[MaskDecoderExport] Patching Mask Decoder Model")
    _, _, spatial_shapes, _ = fusion_wrapper(*input)
    H, W = int(spatial_shapes[0, 0]), int(spatial_shapes[0, 1])
    patch_rpb(model, H, W)
   
    torch_output = wrapper(*input)

    print("[MaskDecoderExport] Tracing Mask Decoder Model")
    exp_program = torch.export.export(wrapper, input, strict=False)
    precision = "fp16" if fp16 else "fp32"
    print(f"[MaskDecoderExport] Building TensortRT Engine ({precision})")
    del wrapper, fusion_wrapper
    engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            exp_program,
            arg_inputs=input,
            #optimization_level=5,
            enable_experimental_decompositions=True,
            use_explicit_typing=True,
            offload_module_to_cpu=True,
            device=torch_tensorrt.Device("cuda:0"),
    )
    model.to(DEVICE)  # offload_module_to_cpu leaves the weights on the CPU

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    return torch_output 

@torch.inference_mode()
def _verify_engine(engine_path: str, inputs_t: Tuple[torch.Tensor], predictor,
                   im: torch.Tensor, img: np.ndarray, captions: List[str]) -> bool:
    """Run the engine, postprocess its outputs, and compare Results against stock ultralytics."""
    print("[MaskDecoderExport] Verifying TRT Engine")

    # reference first — postprocess reads self.batch, which predictor(...) populates
    predictor.set_image(img)
    ref = predictor(text=captions)[0]

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    assert len(in_names) == len(inputs_t), (
        f"engine has {len(in_names)} inputs {in_names}, got {len(inputs_t)} tensors"
    )
    assert len(out_names) == 5, f"expected 5 engine outputs, got {out_names}"

    bound = []
    for name, t in zip(in_names, inputs_t):
        x = t.to(_TRT_TO_TORCH[engine.get_tensor_dtype(name)]).contiguous()
        ctx.set_input_shape(name, tuple(x.shape))
        ctx.set_tensor_address(name, x.data_ptr())
        bound.append(x)

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

    # engine output order matches MaskDecoderWrapper.forward's return order
    pred_logits, pred_boxes, _, presence, pred_masks = trt_out
    preds = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
        "presence_logit_dec": presence,
    }
    got = predictor.postprocess(preds, im, [img])[0]

    ok = True
    if got.boxes.data.shape != ref.boxes.data.shape:
        print(f"boxes  {tuple(got.boxes.data.shape)} != {tuple(ref.boxes.data.shape)}  "
              f"(detection count changed — likely a conf/NMS threshold flip)")
        ok = False
    else:
        d = (ref.boxes.data - got.boxes.data).abs()
        print(f"boxes  {tuple(got.boxes.data.shape)}  max_abs={d.max().item():.3e}")
        print(f"  conf ref={ref.boxes.conf.tolist()}")
        print(f"  conf got={got.boxes.conf.tolist()}")

    if ref.masks is not None and got.masks is not None:
        if got.masks.data.shape != ref.masks.data.shape:
            print(f"masks  {tuple(got.masks.data.shape)} != {tuple(ref.masks.data.shape)}")
            ok = False
        else:
            a, b = ref.masks.data.bool(), got.masks.data.bool()
            diff = (a ^ b).sum().item()
            inter = (a & b).sum().item()
            union = (a | b).sum().item()
            iou = inter / union if union else 1.0
            print(f"masks  {tuple(got.masks.data.shape)}  differing_px={diff}  IoU={iou:.6f}")
            ok &= iou > 0.99
    elif (ref.masks is None) != (got.masks is None):
        print(f"masks  ref={'None' if ref.masks is None else 'set'}  "
              f"got={'None' if got.masks is None else 'set'}")
        ok = False

    print(f"engine {'MATCHES' if ok else 'DIFFERS FROM'} ultralytics after postprocess")
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

def export_and_verify_mask_decoder(predictor : SAM3SemanticPredictor, input : Tuple[torch.Tensor], fp16 : bool, img : np.ndarray, captions : List[str]) -> None:

    torch_output = _verify_wrapper(predictor, input, img, captions)

    precision =  "fp16" if fp16 else "fp32"
    engine_path = os.path.join(os.environ["HOME"], "models", f"mask_decoder_{precision}.engine")
    trace_and_export_mask_deocder(
        predictor.model, 
        input,
        engine_path,
        fp16
    )
    predictor.set_image(img)
    ref = predictor(text=captions)[0] 
    for batch in predictor.dataset:
        im = predictor.preprocess(batch[1])
        break
    _verify_engine(engine_path, input, predictor, im, img, captions)


if __name__ == "__main__":
    from export_ultralytics_image_encoder import ImageEncoderWrapper
    from export_ultralytics_text_encoder import TextEncoderWrapper
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action="store_true", help="build the engine with fp16 enabled")
    args = parser.parse_args()
    
    captions = ["road"]

    predictor = _construct_model(args.fp16)
    predictor.setup_model(model=None, verbose=False)

    image_encoder_wrapper = ImageEncoderWrapper(predictor.model)
    text_encoder_wrapper = TextEncoderWrapper(predictor.model)

    # setup inputs
    img = cv2.imread("test.png")
    predictor.setup_source(img)

    for batch in predictor.dataset:
        im = predictor.preprocess(batch[1])
        break

    img_encoder_input = image_encoder_wrapper(im)
    
    tokenizer = predictor.model.backbone.language_backbone.tokenizer
    ctx = predictor.model.backbone.language_backbone.context_length
    input_ids = tokenizer(captions, context_length=ctx).to(predictor.device)
    attention_mask = (input_ids != 0).bool().ne(1)
    txt_feats, txt_masks, _ = text_encoder_wrapper(input_ids, attention_mask)

    NEG = -65504.0
    txt_masks_f = torch.zeros_like(txt_masks, dtype=txt_feats.dtype).masked_fill_(txt_masks, NEG)

    input = img_encoder_input + (txt_feats, txt_masks, txt_masks_f)
    torch_output = _verify_wrapper(predictor, input, img, captions)

    del image_encoder_wrapper, text_encoder_wrapper
    precision =  "fp16" if args.fp16 else "fp32"
    engine_path = os.path.join(os.environ["HOME"], "models", f"mask_decoder_{precision}.engine")
    trace_and_export_mask_deocder(
        predictor.model, 
        input,
        engine_path,
        args.fp16
    )

    _verify_engine(engine_path, input, predictor, im, img, captions)
