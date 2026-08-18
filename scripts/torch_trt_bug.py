import torch
import torch.nn.functional as F
import torch_tensorrt


class ConstMaskAttention(torch.nn.Module):
    def __init__(self, heads=8, q_len=16, kv_len=32):
        super().__init__()
        self.register_buffer("coords", torch.linspace(-1, 1, kv_len))
        self.proj = torch.nn.Linear(1, heads)

    def forward(self, q, k, v):
        m = self.proj(self.coords.view(1, -1, 1))
        m = m.permute(0, 2, 1).unsqueeze(2).expand(1, m.shape[-1], q.shape[-2], -1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m.contiguous())

if __name__ == "__main__":
    dev, dtype = "cuda", torch.float16
    b, h, ql, kvl, d = 1, 8, 16, 32, 64

    m = ConstMaskAttention(h, ql, kvl).to(dev, dtype).eval()
    q = torch.randn(b, h, ql, d, device=dev, dtype=dtype)
    k = torch.randn(b, h, kvl, d, device=dev, dtype=dtype)
    v = torch.randn(b, h, kvl, d, device=dev, dtype=dtype)

    with torch.inference_mode():
        ep = torch.export.export(m, (q, k, v), strict=False)
        print([n.target for n in ep.graph.nodes if "attention" in str(n.target)])

        torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            ep,
            arg_inputs=[q, k, v],
            use_explicit_typing=True,
            enable_experimental_decompositions=True,
            device=torch_tensorrt.Device("cuda:0"),
        )
    print("converted OK")
