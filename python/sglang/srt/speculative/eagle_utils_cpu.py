
import copy
import logging
import os
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.utils import is_cuda, is_hip, next_power_of_2
from sgl_kernel import fast_topk

logger = logging.getLogger(__name__)

def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len,
    bs_upper,
):
    return

def assign_draft_cache_locs(
    req_pool_indices: torch.Tensor,   # [batch] (int64/int32)
    req_to_token: torch.Tensor,       # [max_batch, pool_len] or flattened 1D
    seq_lens: torch.Tensor,           # [batch] (int32/int64)
    extend_lens: torch.Tensor,        # [batch] (int32/int64)
    num_new_pages_per_topk: torch.Tensor,  # [batch]
    out_cache_loc: torch.Tensor,      # 1D flat buffer (tokens)
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
    page_size: int,
    bs_upper: int,
    iter_upper: int,
):
    return

# @torch.compile(dynamic=True)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        print("topk ", topk)
        topk_index = torch.tensor([[0]], dtype=torch.int32)
        topk_p = torch.tensor([[0.8]], dtype=torch.float)
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device="cpu")
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        print("topk ", topk)
        print("scores ", scores.size())
        print("topk_p ", topk_p.size())
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device="cpu"
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info
