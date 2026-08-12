import onnx
model = onnx.load("image_encoder.onnx")
attn_ops = [n.op_type for n in model.graph.node 
            if any(x in n.op_type.lower() 
                   for x in ["attention", "sdpa", "scaled_dot"])]
print("Attention ops found:", attn_ops)
print("Total nodes:", len(model.graph.node))
