from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import add_prefix, align, is_cuda, is_hip, is_npu, is_cpu

if is_cuda():
    import deep_gemm

from sglang.srt.layers.attention.nsa.utils import NSA_DUAL_STREAM, NSA_USE_REAL_INDEXER
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    # from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    import scipy
    scale=hidden_size**-0.5
    return F.linear(x, torch.tensor(scipy.linalg.hadamard(hidden_size)).to(torch.bfloat16)) * scale
    # return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)

import math

# def act_quant_py(
#     X: torch.Tensor,            # shape (M, N)
#     group_size: int,
#     round_scale: bool,
#     fp8_max_inv: float,
#     fp8_min: float,
#     fp8_max: float,
#     blk_m: int = 32,
# ):  
#     print(X.shape)
#     M, N = X.shape
#     device = X.device
#     dtype = X.dtype

#     n_groups = math.ceil(N / group_size)
#     Y = torch.empty_like(X)
#     S = torch.empty((M, n_groups), dtype=torch.float32, device=device)

#     for pid_m in range(math.ceil(M / blk_m)):
#         row_start = pid_m * blk_m
#         row_end = min((pid_m + 1) * blk_m, M)
#         cur_blk_m = row_end - row_start  # 当前这个 block 的行数

#         for pid_n in range(n_groups):
#             col_start = pid_n * group_size
#             col_end = min((pid_n + 1) * group_size, N)
#             cur_grp = col_end - col_start  # 当前这个组的列数

#             # 切出子块
#             X_block = X[row_start:row_end, col_start:col_end]  # shape (cur_blk_m, cur_grp)

#             # 计算每行绝对值最大值
#             amax = X_block.abs().max(dim=1).values  # shape (cur_blk_m,)
#             amax = torch.maximum(amax, torch.full_like(amax, 1e-4))

#             # 计算 scale
#             if round_scale:
#                 s_local = torch.round(amax * fp8_max_inv) / fp8_max_inv
#             else:
#                 s_local = amax * fp8_max_inv

#             # 存 S
#             # s_local 的形状是 (cur_blk_m,)
#             S[row_start:row_end, pid_n] = s_local.to(torch.float32)

#             # 量化
#             # 扩展 s_local 到 (cur_blk_m, cur_grp)
#             s_expand = s_local.unsqueeze(1).expand(cur_blk_m, cur_grp)
#             y_block = X_block / s_expand
#             y_block = torch.clamp(y_block, fp8_min, fp8_max)

#             # 写回 Y
#             # 注意：Y[row_start:row_end, col_start:col_end] 的子矩阵的形状就是 (cur_blk_m, cur_grp)
#             Y[row_start:row_end, col_start:col_end] = y_block.to(dtype)

#     return Y, S
def act_quant_py(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise FP8 quantization.

    Args:
        x (torch.Tensor): Input tensor, must be contiguous and last dim divisible by block_size.
        block_size (int): Block size for quantization (default: 128).
        scale_fmt (Optional[str]): If not None, enables rounding of scale (simulated).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Quantized tensor in torch.float8_e4m3fn.
            - Scale tensor in torch.float32, shape [..., N // block_size].
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, f"Last dim {x.size(-1)} not divisible by block_size={block_size}"

    # Constants for FP8 E4M3
    FP8_MAX = 448.0
    FP8_MIN = -448.0
    EPS = 1e-4  # to avoid division by zero

    # Reshape to [..., num_blocks, block_size]
    original_shape = x.shape
    M = x.numel() // x.size(-1)
    N = x.size(-1)
    x_reshaped = x.view(M, N)
    num_blocks = N // block_size
    x_blocks = x_reshaped.view(M, num_blocks, block_size)  # [M, num_blocks, block_size]

    # Compute absolute max per block (along last dim)
    amax = torch.amax(torch.abs(x_blocks), dim=-1)  # [M, num_blocks]
    amax = torch.clamp(amax, min=EPS)

    # Compute scale: s = amax / FP8_MAX  --> so that x / s fits in [-FP8_MAX, FP8_MAX]
    scale = amax / FP8_MAX  # [M, num_blocks]

    if scale_fmt is not None:
        # Simulate "fast_round_scale": round scale to nearest representable?
        # Since we don't have the exact function, we round to nearest multiple of a small epsilon.
        # Alternatively, you can quantize scale itself, but here we just round to 6 decimals as approximation.
        scale = torch.round(scale * 1e6) / 1e6

    # Avoid in-place modification; expand scale to match x_blocks
    scale_expanded = scale.unsqueeze(-1)  # [M, num_blocks, 1]

    # Quantize: x_quant = clamp(x / scale, FP8_MIN, FP8_MAX)
    x_dequantized = x_blocks / scale_expanded  # [M, num_blocks, block_size]
    x_clamped = torch.clamp(x_dequantized, min=FP8_MIN, max=FP8_MAX)

    # Convert to FP8
    # Note: PyTorch requires conversion from float32 to float8
    y_fp8 = x_clamped.to(torch.float8_e4m3fn)

    # Reshape back
    y = y_fp8.view(original_shape)
    s = scale.view(*original_shape[:-1], num_blocks).to(torch.float32)

    return y, s

def fp8_index_py(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
) -> torch.Tensor:
    """
    Perform index score computation using FP8-like quantized inputs.

    Args:
        q (torch.Tensor): FP8 query tensor of shape [B, M, H, D]
        q_s (torch.Tensor): Query scales of shape [B, M, H] (float32)
        k (torch.Tensor): FP8 key tensor of shape [B, N, D]
        k_s (torch.Tensor): Key scales of shape [B, N] (float32)

    Returns:
        o (torch.Tensor): Index scores of shape [B, M, N] (float32)
    """
    # Validate shapes and dtypes
    assert q.dtype == torch.float8_e4m3fn, "q must be in torch.float8_e4m3fn"
    assert k.dtype == torch.float8_e4m3fn, "k must be in torch.float8_e4m3fn"
    assert q.is_contiguous() and k.is_contiguous()
    assert q_s.is_contiguous() and k_s.is_contiguous()

    B, M, H, D = q.shape
    B_k, N, D_k = k.shape
    assert (B_k, D_k) == (B, D), f"Shape mismatch: q has D={D}, k has D={D_k}"
    assert q_s.shape == (B, M, H), f"q_s shape {q_s.shape} != (B, M, H) = ({B}, {M}, {H})"
    assert k_s.shape == (B, N), f"k_s shape {k_s.shape} != (B, N) = ({B}, {N})"

    # Step 1: Dequantize q and k to float32 for computation
    # Since FP8 tensors cannot be directly used in matmul in most PyTorch versions,
    # we convert to float32 first.
    q_f32 = q.to(torch.float32)  # [B, M, H, D]
    k_f32 = k.to(torch.float32)  # [B, N, D]

    # Step 2: Compute logits = k @ q^T  --> [B, N, H] for each (m)
    # We need: for each (b, m), compute k[b] @ q[b, m].T  --> [N, H]
    # So: q reshaped to [B, M, D, H] for batch matmul with k [B, N, D]
    q_T = q_f32.transpose(-1, -2)  # [B, M, D, H]
    # Use einsum or bmm: (B, N, D) x (B, M, D, H) -> we need per m
    # Better: expand k to [B, M, N, D] and q to [B, M, D, H], then matmul
    k_exp = k_f32.unsqueeze(1).expand(-1, M, -1, -1)        # [B, M, N, D]
    q_exp = q_f32                                           # [B, M, H, D]
    # Compute: [B, M, N, D] @ [B, M, D, H] -> [B, M, N, H]
    logits = torch.matmul(k_exp, q_exp.transpose(-1, -2))   # [B, M, N, H]

    # Step 3: ReLU (max(x, 0))
    logits = torch.relu(logits)  # [B, M, N, H]

    # Step 4: Multiply by q_s per head: [B, M, N, H] * [B, M, H] -> broadcast
    q_s_exp = q_s.unsqueeze(2)  # [B, M, 1, H]
    logits = logits * q_s_exp   # [B, M, N, H]

    # Step 5: Sum over heads (H)
    logits_sum = logits.sum(dim=-1)  # [B, M, N]

    # Step 6: Multiply by k_s: [B, M, N] * [B, N] -> broadcast
    k_s_exp = k_s.unsqueeze(1)  # [B, 1, N]
    o = logits_sum * k_s_exp    # [B, M, N]

    return o

class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
        # torch.Size([7, 64, 64])
        # torch.Size([7, 64])
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope.unsqueeze(1))
        k_rope = k_rope.view([k_rope.size(0), -1])

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        return query, key

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        assert page_size == 64, "only support page size 64"

        # NOTE(dark): this support extend/decode/decode+graph
        block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        seqlens_32 = metadata.get_seqlens_int32()
        # NOTE(dark): 132 is SM count on H200/B200, not magic number
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_32, blocksize, self.sm_count
        )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            seqlens_32,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        offset = 0

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full((extend_seq_len,), offset, dtype=torch.int32, device="cuda")
            k_fp8_list.append(k_fp8)
            k_scale_list.append(k_scale)
            ks_list.append(ks)
            offset += extend_seq_len

        k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
        k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)
        ks = torch.cat(ks_list, dim=0)
        seq_lens_expanded = metadata.get_seqlens_expanded()
        ke = ks + seq_lens_expanded

        logits = deep_gemm.fp8_mqa_logits(
            q_fp8,
            kv_fp8,
            weights,
            ks,
            ke,
            clean_logits=False,
        )

        assert logits.shape[0] == len(seq_lens_expanded)
        topk_result = metadata.topk_transform(logits, self.index_topk)

        return topk_result

    def forward_indexer_bs_1(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        # if not is_npu():
        #     from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cpu"
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index_py(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)

        return topk_indices

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self.forward_indexer_bs_1(q_fp8, weights, forward_batch, topk, layer_id)

    def _forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        # if not is_npu():
        #     from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        enable_dual_stream = (
            NSA_DUAL_STREAM
            and self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        if not NSA_USE_REAL_INDEXER:  # temporary
            return self._forward_fake(x, q_lora, positions, forward_batch, layer_id)

        query, key = self._get_q_k_bf16(q_lora, x, positions, enable_dual_stream)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            current_stream.wait_stream(self.alt_stream)
        else:
            # q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            # k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
            q_fp8, q_scale = act_quant_py(query, self.block_size, self.scale_fmt)
            k_fp8, k_scale = act_quant_py(key, self.block_size, self.scale_fmt)

        # k_fp8: (seq_len, head_dim) fp8_e4m3fn
        # k_buffer: (num_total_tokens + page_size, head_dim) fp8_e4m3fn
        # k_scale: (seq_len, head_dim // block_size = 1) fp8_e4m3fn
        # k_scale_cache: (num_total_tokens + page_size, head_dim // block_size = 1) fp8_e4m3fn
        forward_batch.token_to_kv_pool.set_index_k_and_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

        weights = self._get_logits_head_gate(x, q_scale)

        if is_cuda():
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return torch.full(
                    (x.shape[0], self.index_topk), -1, dtype=torch.int, device="cuda"
                )

            if forward_batch.forward_mode.is_decode_or_idle():
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                topk_result = self._get_topk_ragged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )

        return topk_result

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self._forward(x, q_lora, positions, forward_batch, layer_id)

    def forward_cpu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self._forward(x, q_lora, positions, forward_batch, layer_id)

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> torch.Tensor:
        import custom_ops
        import torch_npu

        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )
        from sglang.srt.utils import get_bool_env_var

        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        enable_index_cp = (
            get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA") and layer_id >= 4
        )
        is_prefill = forward_batch.forward_mode.is_extend()

        attention_tp_rank = get_attention_tp_rank()
        attention_tp_size = get_attention_tp_size()

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
        if is_prefill and enable_index_cp:
            slice_length = cos.shape[0] // attention_tp_size
            cos = cos[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]
            sin = sin[
                slice_length
                * attention_tp_rank : slice_length
                * (attention_tp_rank + 1)
            ]

        slot_mapping = forward_batch.out_cache_loc
        block_table = forward_batch.attn_backend.forward_metadata.block_tables

        bs = x.shape[0]

        q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
        q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
        q_pe, q_nope = torch.split(
            q,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64, 64 + 64]

        q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin).view(
            bs, self.n_heads, self.rope_head_dim
        )  # [bs, n, d]
        q = torch.cat([q_pe, q_nope], dim=-1)

        k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
        k = self.k_norm(k_proj)
        k_pe, k_nope = torch.split(
            k,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )  # [bs, 64 + 64]

        k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
        k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin).view(
            bs, 1, self.rope_head_dim
        )  # [bs, 1, d]
        k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        if is_prefill and enable_index_cp:
            k, local_k = (
                torch.empty(
                    (k.shape[0] * attention_tp_size, k.shape[1], k.shape[2]),
                    dtype=k.dtype,
                    device=k.device,
                ),
                k,
            )
            get_attention_tp_group().all_gather_into_tensor(k, local_k)

        forward_batch.token_to_kv_pool.set_index_k_buffer(layer_id, slot_mapping, k)

        indexer_input = {}
        if is_prefill:
            actual_seq_lengths_kv = forward_batch.seq_lens.to(device=q.device)
            actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0).to(
                device=q.device
            )
            if enable_index_cp:
                actual_seq_lengths_q -= bs * attention_tp_rank
                actual_seq_lengths_q = torch.max(
                    actual_seq_lengths_q,
                    torch.zeros_like(actual_seq_lengths_q).to(
                        device=actual_seq_lengths_q.device
                    ),
                )
                actual_seq_lengths_q = torch.min(
                    actual_seq_lengths_q,
                    torch.full(actual_seq_lengths_q.shape, bs).to(
                        device=actual_seq_lengths_q.device
                    ),
                )

        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                actual_seq_lengths_q = torch.tensor(
                    [1 + i * 1 for i in range(bs)], dtype=torch.int32, device=k.device
                )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        x = x.view(-1, self.hidden_size)
        weights = self.weights_proj(x)[0]
        block_table = (
            block_table[: actual_seq_lengths_q.size()[0]] if is_prefill else block_table
        )

        topk_indices = torch.ops.custom.npu_lightning_indexer(
            query=q.view(-1, self.n_heads, self.head_dim),
            key=past_key_states,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
            actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(torch.int32),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )

        if is_prefill and enable_index_cp:
            topk_indices, local_topk_indices = (
                torch.empty(
                    (
                        topk_indices.shape[0] * attention_tp_size,
                        topk_indices.shape[1],
                        topk_indices.shape[2],
                    ),
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                ),
                topk_indices,
            )
            get_attention_tp_group().all_gather_into_tensor(
                topk_indices, local_topk_indices
            )

        return topk_indices
