# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.deepseek_v2 import (
    _bind_glm_wavefront_topk,
    _should_skip_index_topk,
)


def test_index_topk_pattern_only_skips_backbone_layers() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        index_topk_pattern="FSSF",
    )

    assert not _should_skip_index_topk(config, 0)
    assert _should_skip_index_topk(config, 1)
    assert _should_skip_index_topk(config, 2)
    assert not _should_skip_index_topk(config, 3)
    assert not _should_skip_index_topk(config, 4)
    assert not _should_skip_index_topk(config, 5)


def test_index_topk_frequency_does_not_skip_nextn_layer() -> None:
    config = SimpleNamespace(
        num_hidden_layers=4,
        index_topk_pattern=None,
        index_topk_freq=3,
        index_skip_topk_offset=2,
    )

    assert _should_skip_index_topk(config, 2)
    assert not _should_skip_index_topk(config, 4)


def test_wavefront_binds_indexer_and_attention_topk_storage() -> None:
    original_indices = torch.zeros(1, 1, dtype=torch.int32)
    original_scores = torch.zeros(1, 1)
    lane_indices = torch.ones(1, 1, dtype=torch.int32)
    lane_scores = torch.ones(1, 1)
    indexer_op = SimpleNamespace(
        topk_indices_buffer=original_indices,
        topk_scores_buffer=original_scores,
    )
    indexer = SimpleNamespace(
        indexer_op=indexer_op,
        topk_indices_buffer=original_indices,
        topk_scores_buffer=original_scores,
    )
    impl = SimpleNamespace(topk_indices_buffer=original_indices)
    mla_layer = SimpleNamespace(impl=impl)
    wrapper = SimpleNamespace(
        indexer=indexer,
        mla_attn=mla_layer,
        topk_indices_buffer=original_indices,
    )
    layer = SimpleNamespace(self_attn=SimpleNamespace(mla_attn=wrapper))

    with _bind_glm_wavefront_topk(layer, lane_indices, lane_scores):
        assert indexer_op.topk_indices_buffer is lane_indices
        assert indexer_op.topk_scores_buffer is lane_scores
        assert wrapper.topk_indices_buffer is lane_indices
        assert impl.topk_indices_buffer is lane_indices

    assert indexer_op.topk_indices_buffer is original_indices
    assert indexer_op.topk_scores_buffer is original_scores
    assert wrapper.topk_indices_buffer is original_indices
    assert impl.topk_indices_buffer is original_indices
