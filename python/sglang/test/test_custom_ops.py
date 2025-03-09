# Adapted from https://github.com/vllm-project/vllm/blob/8ca7a71df787ad711ad3ac70a5bd2eb2bb398938/tests/quantization/test_fp8.py

import pytest
import torch

from sglang.srt.custom_op import scaled_fp8_quant


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_fp8_quant_per_tensor(dtype) -> None:

    def quantize_ref_per_tensor(tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        qweight = (tensor.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
        qweight = qweight.to(torch.float8_e4m3fn)
        return qweight

    def dequantize_per_tensor(tensor, inv_scale, dtype):
        fake_qweight = tensor.to(dtype)
        dq_weight = fake_qweight * inv_scale
        return dq_weight

    # Note that we use a shape % 8 != 0 to cover edge cases,
    # because scaled_fp8_quant is vectorized by 8.
    x = (torch.randn(size=(11, 11), device="cuda") * 13).to(dtype)

    # Test Per Tensor Dynamic quantization
    # scale max(abs(x)) / FP8_E4M3_MAX
    y, scale = scaled_fp8_quant(x, None)
    ref_y = quantize_ref_per_tensor(x, scale)
    torch.testing.assert_close(y, ref_y)
    torch.testing.assert_close(
        dequantize_per_tensor(y, scale, dtype),
        dequantize_per_tensor(ref_y, scale, dtype),
    )

    # Test Per Tensor Static quantization
    y, _ = scaled_fp8_quant(x, scale)
    ref_y = quantize_ref_per_tensor(x, scale)
    torch.testing.assert_close(y, ref_y)
    torch.testing.assert_close(
        dequantize_per_tensor(y, scale, dtype),
        dequantize_per_tensor(ref_y, scale, dtype),
    )


if __name__ == "__main__":
    # Run the specific test function directly
    pytest.main([__file__])
