import itertools

import unittest
# import cuda sgl-kernel
import sgl_kernel
import torch
# import cpu amx sgl-kernel
from sgl_kernel_cpu import common_ops

kernel = torch.ops.sgl_kernel
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
from sglang.srt.layers.moe.topk import fused_topk
torch.manual_seed(1234)

from utils import (
    precision,
)

from sglang.test.test_utils import CustomTestCase


def fused_moe(a, w1, w2, score, topk, renormalize, prepack):

    G = 1
    topk_group = 1

    B, D = a.shape
    topk_weights = torch.empty(B, topk, dtype=torch.float32)
    topk_ids = torch.empty(B, topk, dtype=torch.int32)
    topk_weights, topk_ids = kernel.grouped_topk_cpu(
        a, score, topk, renormalize, G, topk_group, 0, None, None
    )

    packed_w1 = kernel.convert_weight_packed(w1) if prepack else w1
    packed_w2 = kernel.convert_weight_packed(w2) if prepack else w2

    inplace = True
    return kernel.fused_experts_cpu(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        inplace,
        False,
        False,
        None,
        None,
        None,
        None,
        None,
        prepack,
    )

def torch_cuda_fused_moe(a, w1, w2, score, topk, renormalize):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    return outplace_fused_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
    )

class TestFusedExperts(CustomTestCase):
    M = [2, 114]
    N = [32]
    K = [32]
    E = [4]
    topk = [2]
    renormalize = [False, True]

    def _bf16_moe(self, m, n, k, e, topk, renormalize):
        dtype = torch.bfloat16
        prepack = True

        a = torch.randn((m, k), device="cpu", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cpu", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cpu", dtype=dtype) / 10
        score = torch.randn((m, e), device="cpu", dtype=dtype)

        # calling cuda kernel (triton)
        torch_output = torch_cuda_fused_moe(a.cuda(), w1.cuda(), w2.cuda(), score.cuda(), topk, renormalize)
        # caling cpu amx kernel
        fused_output = fused_moe(a, w1, w2, score, topk, renormalize, prepack)


    def test_bf16_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
            self.renormalize,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                e=params[3],
                topk=params[4],
                renormalize=params[5],
            ):
                self._bf16_moe(*params)

class TestTopK(CustomTestCase):
    def _run_single_test(self, M, E, topk, renormalize, dtype):
        torch.manual_seed(1998)

        # expand gating_output by M, otherwise bfloat16 fall into same value aftering truncating
        hidden_states = torch.randn(M, 100, dtype=dtype)
        gating_output = torch.randn(M, E, dtype=dtype) * 2 * M
        # calling cuda kernel (cuda sgl-kernel)
        ref_topk_weights, ref_topk_ids = fused_topk(
            hidden_states.float().cuda(),
            gating_output.float().cuda(),
            topk,
            renormalize,
        )

        # caling cpu amx kernel
        topk_weights, topk_ids = torch.ops.sgl_kernel.topk_softmax_cpu(
            hidden_states, gating_output, topk, renormalize
        )


    def test_topk(self):
        for renormalize in [True, False]:
            self._run_single_test(123, 8, 2, renormalize, torch.bfloat16)
            self._run_single_test(123, 16, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 32, 3, renormalize, torch.bfloat16)
            self._run_single_test(123, 64, 6, renormalize, torch.bfloat16)
            self._run_single_test(123, 256, 4, renormalize, torch.bfloat16)
            self._run_single_test(123, 160, 6, renormalize, torch.bfloat16)

if __name__ == "__main__":
    unittest.main()

