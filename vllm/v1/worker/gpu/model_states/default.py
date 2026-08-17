# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        self.execute_mm_encoder(scheduled_encoder_inputs)

        mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
        if self.mm_pruner is not None and mm_embeds:
            # EVS: recompute mrope positions for pruned media.
            mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
            # We must flush the staged rope updates for prepare_inputs() to pick up.
            self.apply_staged_writes()

        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if self.rope_state is None:
            return {}  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = (
            input_batch.max_req_tokens or input_batch.num_scheduled_tokens.max().item()
        )
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        is_prefilling = torch.from_numpy(input_batch.is_prefilling_np)
        if num_reqs != input_batch.num_reqs:
            padded_is_prefilling = torch.zeros(num_reqs, dtype=torch.bool)
            padded_is_prefilling[: input_batch.num_reqs] = is_prefilling
            is_prefilling = padded_is_prefilling
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len_upper_bound=input_batch.max_seq_len_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            is_prefilling=is_prefilling,
            max_req_tokens=input_batch.max_req_tokens or 0,
        )
        return attn_metadata

    def prepare_glm52_wavefront_attn(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        chunk_size: int,
        dcp_rank: int,
    ) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
        """Build metadata for the two strict GLM-5.2 wavefront lanes."""
        if (
            chunk_size <= 0
            or input_batch.num_reqs != 1
            or input_batch.num_tokens != 2 * chunk_size
            or input_batch.num_tokens_after_padding != input_batch.num_tokens
            or input_batch.num_draft_tokens != 0
            or not bool(input_batch.is_prefilling_np[0])
            or input_batch.num_scheduled_tokens.tolist() != [2 * chunk_size]
            or input_batch.query_start_loc_np[:2].tolist() != [0, 2 * chunk_size]
            or bool(input_batch.is_padding[: input_batch.num_tokens].any().item())
            or slot_mappings.shape[1] < input_batch.num_tokens
        ):
            raise RuntimeError(
                "GLM-5.2 wavefront metadata requires one unpadded text prefill "
                "request with exactly two chunks and no draft tokens."
            )

        parallel_config = self.vllm_config.parallel_config
        dcp_size = parallel_config.decode_context_parallel_size
        cp_interleave = parallel_config.cp_kv_cache_interleave_size
        query_start_loc_cpu = torch.tensor([0, chunk_size], dtype=torch.int32)
        query_start_loc_gpu = input_batch.query_start_loc.new_tensor([0, chunk_size])
        final_seq_lens = input_batch.seq_lens[:1]
        final_upper_bounds = input_batch.seq_lens_cpu_upper_bound[:1]
        lane_seq_lens = [final_seq_lens - chunk_size, final_seq_lens]
        lane_upper_bounds = [
            final_upper_bounds - chunk_size,
            final_upper_bounds,
        ]
        raw_slot_mappings = [
            slot_mappings[:, :chunk_size],
            slot_mappings[:, chunk_size : 2 * chunk_size],
        ]
        metadata: list[dict[str, Any]] = []
        for lane_idx in range(2):
            seq_lens = lane_seq_lens[lane_idx]
            upper_bounds = lane_upper_bounds[lane_idx]
            dcp_local_seq_lens = None
            if dcp_size > 1:
                width = dcp_size * cp_interleave
                remainder = torch.clamp(
                    seq_lens % width - dcp_rank * cp_interleave,
                    min=0,
                    max=cp_interleave,
                )
                dcp_local_seq_lens = seq_lens // width * cp_interleave + remainder
            token_start = lane_idx * chunk_size
            metadata.append(
                build_attn_metadata(
                    attn_groups=attn_groups,
                    num_reqs=1,
                    num_tokens=chunk_size,
                    query_start_loc_gpu=query_start_loc_gpu,
                    query_start_loc_cpu=query_start_loc_cpu,
                    max_query_len=chunk_size,
                    seq_lens=seq_lens,
                    max_seq_len=int(upper_bounds[0]),
                    block_tables=block_tables,
                    slot_mappings=raw_slot_mappings[lane_idx],
                    kv_cache_config=kv_cache_config,
                    seq_lens_cpu_upper_bound=upper_bounds,
                    max_seq_len_upper_bound=int(upper_bounds[0]),
                    dcp_local_seq_lens=dcp_local_seq_lens,
                    positions=input_batch.positions[
                        token_start : token_start + chunk_size
                    ],
                    is_prefilling=torch.ones(1, dtype=torch.bool),
                    rswa_prefix_lens=input_batch.prompt_lens,
                    max_req_tokens=chunk_size,
                    metadata_builder_idx=lane_idx,
                )
            )
        from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
            B12xMLASparseMetadata,
            link_glm52_wavefront_ckv_metadata,
        )

        linked: set[tuple[int, int]] = set()
        for layer_name, lane_0_metadata in metadata[0].items():
            lane_1_metadata = metadata[1][layer_name]
            pair = (id(lane_0_metadata), id(lane_1_metadata))
            if pair in linked or not isinstance(lane_0_metadata, B12xMLASparseMetadata):
                continue
            if not isinstance(lane_1_metadata, B12xMLASparseMetadata):
                raise RuntimeError("Wavefront lanes use different attention backends")
            link_glm52_wavefront_ckv_metadata(lane_0_metadata, lane_1_metadata)
            linked.add(pair)
        return metadata, raw_slot_mappings
