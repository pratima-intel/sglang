import itertools
import unittest

# TODO: use interface in cpu.py
import sgl_kernel
import torch
import torch.nn as nn
from utils import (
    convert_weight,
    native_w8a8_per_token_matmul,
    per_token_quant_int8,
    precision,
    unpack_and_dequant_awq,
)

from sglang.test.test_utils import CustomTestCase
import time
torch.manual_seed(1234)
a = torch.ones(10240,10240)
b = torch.ones(10240,10240)
def flush_cache():
    c = a+b
    return c

class Mod(nn.Module):
    def __init__(self, input_channel, output_channel, has_bias):
        super(Mod, self).__init__()
        self.linear = torch.nn.Linear(input_channel, output_channel, has_bias)

    def forward(self, x):
        return self.linear(x)


class TestGemm(CustomTestCase):
    M = [1024]
    N = [4096]
    K = [4096]
    has_bias = [False]

    M_int8 = [2, 128]
    N_int8 = [32 * 12]
    K_int8 = [32 * 17]

    M_fp8 = [1, 11]
    N_fp8 = [128, 224]
    K_fp8 = [512, 576]

    M_awq = [1, 32]
    N_awq = [4096]
    K_awq = [4096]

    def _bf16_gemm(self, M, N, K, has_bias):

        mat1 = torch.randn(M, K, dtype=torch.bfloat16)
        mat2 = torch.randn(N, K, dtype=torch.bfloat16)

        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)

        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
        out2 = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, packed_mat2, bias if has_bias else None, True
        )

        e = 0
        for _ in range(1000):
            _ = flush_cache()
            s = time.time()
            out2 = torch.ops.sgl_kernel.weight_packed_linear(
                mat1, packed_mat2, bias if has_bias else None, True
            )
            e = e + time.time() - s
        print("bf16 (ms): ", e/1000*1000)

    def xxtest_bf16_gemm(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._bf16_gemm(*params)

    def _int8_gemm(self, M, N, K, has_bias):
        dtype = torch.bfloat16
        A = torch.randn((M, K), dtype=dtype) / 10
        Aq, As = per_token_quant_int8(A)

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = torch.rand(N) * factor_for_scale

        bias = torch.randn(N) if has_bias else None

        # test the fused version
        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(Bq)
        fused_out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A, packed_mat2, Bs, bias if has_bias else None, torch.bfloat16, True
        )
        e = 0
        for _ in range(1000):
            _ = flush_cache()
            s = time.time()
            fused_out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                A, packed_mat2, Bs, bias if has_bias else None, torch.bfloat16, True
            )
            e = e + time.time() - s
        print("w8a8 (ms): ", e/1000*1000)

    def xxtest_int8_gemm(self):
        for params in itertools.product(
            self.M_int8,
            self.N_int8,
            self.K_int8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._int8_gemm(*params)

    def _fp8_gemm(self, M, N, K, has_bias):
        prepack = True
        chunk = False
        scale_block_size_N = 64
        scale_block_size_K = 128
        assert scale_block_size_N <= N
        assert scale_block_size_K <= K
        A_dtype = torch.bfloat16

        model = Mod(K, N, has_bias).eval()
        if chunk:
            data = torch.randn(M, K + 6, dtype=A_dtype).narrow(1, 0, K)
        else:
            data = torch.randn(M, K, dtype=A_dtype)

        weight = model.linear.weight  # (N, K)

        if has_bias:
            bias = model.linear.bias

        fp8_weight, scales, dq_weight = convert_weight(
            weight, [scale_block_size_N, scale_block_size_K], A_dtype
        )

        if has_bias:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T) + bias.to(A_dtype)
        else:
            ref = torch.matmul(data.to(A_dtype), dq_weight.T)

        if prepack:
            fp8_weight = torch.ops.sgl_kernel.convert_weight_packed(fp8_weight)

        opt = torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
            data,
            fp8_weight,
            scales,
            [scale_block_size_N, scale_block_size_K],
            bias if has_bias else None,
            data.dtype,
            prepack,
        )
        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, opt, atol=atol, rtol=rtol)

    def xxtest_fp8_gemm(self):
        for params in itertools.product(
            self.M_fp8,
            self.N_fp8,
            self.K_fp8,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._fp8_gemm(*params)

    def _int4_awq_gemm(self, M, N, K, group_size, has_bias, w4a8):
        awq_weight = torch.randint(-128, 128, (K, N // 8)).to(torch.int)
        awq_zero = torch.randint(0, 10, (K // group_size, N // 8)).to(torch.int)
        awq_scales = torch.rand(int(K // group_size), N).to(torch.bfloat16)
        bf16_weight, _ = unpack_and_dequant_awq(
            awq_weight, awq_zero, awq_scales, 4, 128
        )
        if has_bias:
            bias = torch.rand(bf16_weight.shape[0]).to(torch.float)
        else:
            bias = None
        x = torch.rand(M, bf16_weight.size(-1)).to(torch.bfloat16)

        if w4a8:
            packed_weight, packed_zero, packed_scales = (
                torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                    awq_weight, awq_zero, awq_scales, True
                )
            )
            target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                x,
                packed_weight,
                packed_zero,
                packed_scales,
                bias,
                w4a8,
            )
            e = 0
            for _ in range(1000):
                _ = flush_cache()
                s = time.time()
                target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                    x,
                    packed_weight,
                    packed_zero,
                    packed_scales,
                    bias,
                    w4a8,
                )
                e = e + time.time() - s
            print("w4a8 (ms): ", e/1000*1000)
        else:
            packed_weight, packed_zero, packed_scales = (
                torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                    awq_weight, awq_zero, awq_scales, False
                )
            )
            target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                x, packed_weight, packed_zero, packed_scales, bias, w4a8
            )
            e=0
            for _ in range(1000):
                _ = flush_cache()
                s = time.time()
                target_res = torch.ops.sgl_kernel.int4_scaled_mm_cpu(
                    x,
                    packed_weight,
                    packed_zero,
                    packed_scales,
                    bias,
                    w4a8,
                )
                e = e + time.time() - s
            print("w4a16 (ms): ", e/1000*1000)

    def test_int4_awq_gemm(self):
        # for params in itertools.product(
        #     self.M,
        #     self.N,
        #     self.K,
        #     self.has_bias,
        # ):
        #     with self.subTest(
        #         M=params[0],
        #         N=params[1],
        #         K=params[2],
        #         has_bias=params[3],
        #     ):
        #         self._bf16_gemm(*params)
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
            ):
                self._int8_gemm(*params)
        for params in itertools.product(
            self.M, self.N, self.K, [128], self.has_bias, [False, True]
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                group_size=params[3],
                has_bias=params[4],
                w4a8=params[5],
            ):
                self._int4_awq_gemm(*params)


if __name__ == "__main__":
    unittest.main()
