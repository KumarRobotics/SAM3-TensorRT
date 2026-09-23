# SAM3 Tensor RT

This repo is a partial TensorRT implementation of [SAM 3](https://huggingface.co/facebook/sam3), designed with multi-class inference in mind. This implementation reuses the image encoding when there are multiple text inputs.

The TensorRT export must be run on the type of hardware it will be deployed on, i.e. you must export for a jetson on a jetson device.

### Tracing

The model currently traces the image at a fixed `(1, 3, 644, 644)` size. You can trace with batched images if you want, but the tracing function would have to be changed to handle a fixed size or to have that dimension be dynamic. 
By default I export the text encoder with 32 tokens.

Since SAM 3 is a gated model you will need to trace the model yourself once you get access to it on huggingface. Once you are granted access download the `sam3.pt` file and put it in the `models` directory.

Trace the model suing the `.trace` docker image. You will need an up to date nvidia driver on your device.
```[bash]
cd docker
./build-trt.bash
./run-trt.bash
```
This will bring you inside the docker image, then run: `export_all.py`

### Building CPP

To build the model in cpp use:
```[bash]
cd sam3trt
cmake -S . -B build
cmake --build build
```
this will give you a `sam3` executable which you can run with `./sam3 <image_path> <label>`. It writes `<image_name>_viz.png` next to the input image with the detections drawn on the original image.

To skip building the unit tests (removes the GTest requirement) add `-DSAM3TRT_BUILD_TESTS=OFF` when configuring.

### Installing CPP

After building, install with:
```[bash]
sudo cmake --install build --prefix /usr/local
```
This installs the `sam3` executable to `/usr/local/bin`, the static libraries to `/usr/local/lib`, the headers to `/usr/local/include/sam3trt`, the config files to `/usr/local/share/sam3trt/config` and the cmake package files to `/usr/local/lib/cmake/sam3trt`.

To use the installed library in another cmake project:
```[cmake]
find_package(sam3trt REQUIRED)
target_link_libraries(my_app PRIVATE sam3trt::sam3trt)
```
`find_package` also sets `SAM3TRT_CONFIG_DIR` to the installed config directory.

### API Usage

Initialize the model with:
```[cpp]
#include <sam3trt/Sam3Model.hpp>
#include <sam3trt/Visualize.hpp>

const std::string models = std::string(getenv("HOME")) + "/models";
Sam3Model model(models, CONFIG_DIR + "/merges.txt", CONFIG_DIR + "/vocab.json", CONFIG_DIR + "/params.json");
```
where `CONFIG_DIR` is `/usr/local/share/sam3trt/config` when installed. Then infer with:
```[cpp]
cv::Mat image = cv::imread("..."); // in BGR format
std::string label = "...";
// or 
std::vector<std::string> label = {..., };
Sam3Result result = model.infer(image, label);

for (const Detection& det : result.detections) {
    // det.bbox, det.mask, det.class_label, det.class_id, det.confidence
}
std::vector<float> feature = result.features.patch(i, j);
cv::Mat viz = visualize(image, result);
```

### Installing Python

The python version lives in `sam3trtpy` and has the same API as the cpp version. Install it with:
```[bash]
pip install ./sam3trtpy
```
The CUDA kernels are compiled at runtime with NVRTC when the model is first created, so there is no build step.

### Python Usage

The `sam3.py` script mirrors the cpp executable:
```[bash]
python3 sam3trtpy/sam3.py <image_path> <label>
```
which writes `<image_name>_viz.png` next to the input image.

Initialize the model with:
```[python]
from pathlib import Path
import cv2
from sam3trt import CONFIG_DIR, Sam3Model, visualize

model = Sam3Model(Path.home() / "models", CONFIG_DIR / "merges.txt", CONFIG_DIR / "vocab.json", CONFIG_DIR / "params.json")
```
and infer with:
```[python]
image = cv2.imread("...") # in BGR format
label = "..."
# or
label = [..., ]
result = model.infer(image, label)

for det in result.detections:
    print(det.bbox, det.class_label, det.class_id, det.confidence, det.mask.shape)
feature = result.features.patch(i, j)
viz = visualize(image, result)
```

### Unit Tests

The model tests need the fp16 engines in `~/models`. The cpp tests use GTest and are run with ctest after building:
```[bash]
cd sam3trt
ctest --test-dir build
```
The timing benchmark takes about 35 seconds, to skip it use `ctest --test-dir build -E Timing`. A single test suite can be run directly, e.g. `./build/test/test_processor`.

To run the python tests:
```[bash]
cd sam3trtpy
python3 -m unittest discover -s tests -v
```

### Performance

The model is traced from the [ultralytics](https://docs.ultralytics.com/models/sam-3) of SAM 3. This is a scalped version of the original version. It is traced at `float16` precision and benchmarked against it. 
![RTX 2000ada](assets/trt_vs_ult_fp16.png)

The model is compared the ultralytics version it is traced from at the same precision. Most of the speed up comes on the decoding side, so this implementation is better for inferencing with many classes.
