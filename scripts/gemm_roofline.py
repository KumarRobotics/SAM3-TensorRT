"""
Is TensorRT actually leaving GEMM performance on the table?

TRT's profile reports the SAM3 mlp_fc1 GEMM (M=2116, K=1024, N=4736) at
0.748 ms = 27.4 TFLOPS. This runs the identical shape through cuBLAS and a
few variants to find out whether that is the card's ceiling or TRT's.

Interpretation:
  cuBLAS >> 27.4 TFLOPS  -> TRT tactic selection is bad. Tune the builder.
  cuBLAS ~= 27.4 TFLOPS  -> the card is the wall. Only FP8 changes it.
"""
import torch
import numpy as np

# Exact SAM3 ViT-L shapes at 644x644 (46x46 = 2116 tokens)
M, K, N_FF, D = 2116, 1024, 4736, 1024
TRT_FC1_MS = 0.748          # measured, mlp_fc1
TRT_QKV_MS = 0.462          # measured, fused q+k+v
TRT_TOTAL_GEMM_TFLOP = 1.88  # whole encoder


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def gemm(m, k, n, dtype=torch.float16, label="", ref_ms=None):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    ms = bench(lambda: torch.mm(a, b))
    tflops = (2 * m * k * n) / (ms * 1e-3) / 1e12
    line = f"  {label:<44} {ms:7.3f} ms  {tflops:6.1f} TFLOPS"
    if ref_ms:
        line += f"   (TRT {ref_ms:.3f} ms, {(ref_ms/ms - 1)*100:+.0f}%)"
    print(line)
    return tflops


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs  "
          f"{p.total_memory/1e9:.1f} GB")
    print(f"torch {torch.__version__}\n")

    print("fp16 reduced-precision reduction (fp16 accumulate):",
          torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    print()

    print("=== The exact shapes TRT is running ===")
    fc1 = gemm(M, K, N_FF, label="mlp_fc1   2116x1024 @ 1024x4736",
               ref_ms=TRT_FC1_MS)
    gemm(M, N_FF, D, label="mlp_fc2   2116x4736 @ 4736x1024")
    gemm(M, K, 3 * D, label="qkv       2116x1024 @ 1024x3072",
         ref_ms=TRT_QKV_MS)
    gemm(M, K, D, label="o_proj    2116x1024 @ 1024x1024")

    print("\n=== fp16 accumulate vs fp32 accumulate ===")
    print("  (Ada restricts fp32-accumulate tensor throughput; if these differ")
    print("   a lot, accumulation type is a real lever)")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N_FF, device="cuda", dtype=torch.float16)
    for flag in (True, False):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
        ms = bench(lambda: torch.mm(a, b))
        tf = (2 * M * K * N_FF) / (ms * 1e-3) / 1e12
        print(f"  reduced_precision_reduction={str(flag):<5} "
              f"{ms:7.3f} ms  {tf:6.1f} TFLOPS")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    print("\n=== bf16 and tf32, for reference ===")
    gemm(M, K, N_FF, torch.bfloat16, "mlp_fc1 bf16")
    torch.backends.cuda.matmul.allow_tf32 = True
    gemm(M, K, N_FF, torch.float32, "mlp_fc1 fp32/tf32")

    print("\n=== larger M, to see if you are tile-starved at 2116 ===")
    for m in (2116, 4096, 8192):
        gemm(m, K, N_FF, label=f"mlp_fc1 shape at M={m}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  TRT achieved on mlp_fc1 : 27.4 TFLOPS")
    print(f"  cuBLAS achieved         : {fc1:.1f} TFLOPS")
    ratio = fc1 / 27.4
    if ratio > 1.15:
        floor = TRT_TOTAL_GEMM_TFLOP / fc1 * 1e3
        print(f"\n  cuBLAS is {(ratio-1)*100:.0f}% FASTER. TensorRT picked bad")
        print(f"  tactics. Rebuild with a fresh timing cache, locked clocks,")
        print(f"  builder_optimization_level=5, and a larger workspace.")
        print(f"  Achievable GEMM floor would be ~{floor:.0f} ms, not 69 ms.")
    elif ratio < 0.87:
        print(f"\n  TRT is {(1/ratio-1)*100:.0f}% faster than cuBLAS. The engine")
        print(f"  is fine; the remaining gap is elsewhere.")
    else:
        print(f"\n  Within {abs(ratio-1)*100:.0f}%. Both hit the same wall.")
        print(f"  This is the card's dense fp16 GEMM ceiling. 1.88 TFLOP of")
        print(f"  encoder GEMM cannot go below "
              f"{TRT_TOTAL_GEMM_TFLOP/fc1*1e3:.0f} ms in fp16, in ANY runtime.")
        print(f"  FP8 (sm_89 supports it, 2x rate) is the only real lever.")


if __name__ == "__main__":
    main()
