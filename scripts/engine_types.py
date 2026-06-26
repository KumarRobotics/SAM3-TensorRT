import tensorrt as trt

def inspect_plan(path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    print(f"\n{path}")
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        mode  = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"  {'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} "
              f"{dtype} {str(shape):<20} {name}")

inspect_plan("../models/image_encoder.plan")
inspect_plan("../models/text_encoder.plan")
