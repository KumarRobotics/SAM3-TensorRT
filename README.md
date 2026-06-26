# SAM3 Tensor RT

This repo is a partial Tensor RT implementation of [SAM 3](https://huggingface.co/facebook/sam3), designed with multi-class inference in mind. This implementation reuses the image encoding when there are multiple text inputs.

The TensorRT export must be run on the type of hardware it will be deployed on, i.e. you must export for a jetson on a jetson device.

### Tracing

The model currently traces the image at a fixed `(1, 3, 644, 644)` size. You can trace with batched images if you want, but the tracing function would have to be changed to handle a fixed size or to have that dimension be dynamic. 

By default I export the text encoder with 32 tokens.

Since SAM 3 is a gated model you will need to trace the model yourself once you get access to it on huggingface. The easiest thing is to use the `trace` docker image:
```[bash]
cd docker
./build-trace.bash
cd .. 
./run-trace.bash
```
This will bring you inside the docker image, now just run the tracking scripts.
