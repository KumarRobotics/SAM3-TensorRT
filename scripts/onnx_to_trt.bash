#!/bin/bash

trtexec --onnx=$HOME/models/image_encoder.onnx --saveEngine=$HOME/models/image_encoder.plan --fp16
trtexec --onnx=$HOME/models/text_encoder.onnx --saveEngine=$HOME/models/text_encoder.plan --fp16
trtexec --onnx=$HOME/models/mask_decoder.onnx --saveEngine=$HOME/models/mask_decoder.plan --fp16
