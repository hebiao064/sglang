#!/usr/bin/env python3
"""
Simple reproduction: hidden_dim=1368 + per token quant
Test if cutlass_scaled_mm works but fp8_scaled_mm fails
"""

import torch
from vllm import _custom_ops as ops
from sgl_kernel import fp8_scaled_mm
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8


def create_fp8_tensor(shape, dtype=torch.float8_e4m3fn, device="cuda"):
    float_tensor = torch.randn(shape, dtype=torch.float16, device=device)
    return float_tensor.to(dtype)


def create_scale_tensor(shape, device="cuda"):
    return torch.rand(shape, dtype=torch.float32, device=device) * 0.1 + 0.01


def main():
    print("🚀 Testing hidden_dim=1368 + per token quant")
    
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    
    # https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
    # "intermediate_size": 10944, when tp = 8, hidden_dim = 1368
    batch_size, seq_len, hidden_dim, output_dim = 4, 512, 1368, 2048
    device = "cuda"
    
    print(f"\n🧪 Testing: {batch_size}x{seq_len}x{hidden_dim} -> {output_dim}")
    
    input_2d = torch.randn((batch_size * seq_len, hidden_dim), dtype=torch.float16, device=device)
    

    weight_data = torch.randn(hidden_dim * output_dim, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
    weight = torch.as_strided(weight_data, (hidden_dim, output_dim), (1, hidden_dim))
    
    qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
    
    
    weight_scale = create_scale_tensor((output_dim,), device=device)  # [2048]
    bias = torch.randn(output_dim, dtype=torch.float16, device=device)

    
    # Test 1: cutlass_scaled_mm (should work)
    print(f"\n1️⃣ cutlass_scaled_mm:")
    try:
        output1 = ops.cutlass_scaled_mm(qinput, weight, out_dtype=input_2d.dtype, 
                                        scale_a=x_scale, scale_b=weight_scale, bias=bias)
        print(f"   ✅ SUCCESS")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 2: fp8_scaled_mm (should fail on 16-byte alignment)
    print(f"\n2️⃣ fp8_scaled_mm:")
    try:
        output2 = fp8_scaled_mm(qinput, weight, x_scale, weight_scale, 
                                out_dtype=input_2d.dtype, bias=bias)  
        print(f"   ✅ SUCCESS")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")



if __name__ == "__main__":
    main() 