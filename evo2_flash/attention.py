"""Mamba2 metadata plus per-request computed-token counts, for Hyena rollback.

vLLM 0.26 passes ``num_accepted_tokens`` to Mamba2 builders only on steps where
at least one scheduled request carries draft tokens (``use_spec_decode`` in
``gpu_model_runner`` is evaluated per step). On a step without drafts the
metadata still holds the speculative slot matrix, but not the counts, so the
mixers cannot tell which snapshot of the previous verify step is live.

Hyena therefore stamps every cached snapshot with the number of tokens it
represents and resolves the live snapshot by comparing that stamp with the
request's computed-token count. This builder supplies that count on every step,
in a persistent buffer so that CUDA-graph replays read fresh values. It
subclasses the Mamba2 builder so the runner keeps passing acceptance counts when
it has them; the mixers no longer rely on them.
"""

from dataclasses import dataclass, replace

import torch
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionBackend,
    Mamba2AttentionMetadata,
    Mamba2AttentionMetadataBuilder,
)


@dataclass
class HyenaMamba2Metadata(Mamba2AttentionMetadata):
    # (num_decodes,) tokens already computed before this step, per decode request.
    num_computed_tokens_d: torch.Tensor | None = None
    # Prefix caching ("align" mode) only, prefill rows: tokens already computed,
    # and the block holding the state after exactly that many tokens (the block
    # of the last computed token). Lets the mixers repair the state vLLM copies
    # into the current block before the step -- see HyenaMixerBase.
    # _repair_resumed_prefill_state for the vLLM 0.26 defect this works around.
    # (Named apart from the base class's own num_computed_tokens_p, which is
    # only filled in some modes.)
    computed_before_p: torch.Tensor | None = None
    state_src_block_p: torch.Tensor | None = None


# Per-step, host-side (num_reqs,) view of computed + scheduled tokens, in the
# runner's row order -- which is also the order of the sampler's logits rows.
# Published from here because the metadata builder is the only component that
# sees it before sampling, and `evo2_flash.cfg` needs it to tell a logits row
# that is a real next-token distribution from one that is merely the tail of a
# mid-prompt chunk (vLLM samples for those too and discards the result,
# `gpu_model_runner.py:2216` taking one row per scheduled request). It is a CPU
# tensor the runner has already materialised, so reading it costs no sync.
# `(num_reqs, seq_lens)`, or None if the runner did not supply it.
STEP_SEQ_LENS: tuple[int, torch.Tensor] | None = None


class HyenaMamba2MetadataBuilder(Mamba2AttentionMetadataBuilder):
    metadata_cls = HyenaMamba2Metadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # CUDA graphs retain tensor addresses: refresh this buffer in place.
        self.decode_num_computed_tokens = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        global STEP_SEQ_LENS
        bound = common_attn_metadata.seq_lens_cpu_upper_bound
        # Precise for prefill rows and for every row outside async spec decode,
        # which CFG forbids; that is all `evo2_flash.cfg` reads it for.
        STEP_SEQ_LENS = (
            None if bound is None else (common_attn_metadata.num_reqs, bound)
        )
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs
        )
        nd, npf = metadata.num_decodes, metadata.num_prefills
        extra = {}
        if nd or (npf and self.vllm_config.cache_config.mamba_cache_mode == "align"):
            counts = common_attn_metadata.compute_num_computed_tokens()
        if nd:
            self.decode_num_computed_tokens[:nd].copy_(counts[:nd])
            extra["num_computed_tokens_d"] = self.decode_num_computed_tokens[:nd]
        if npf and self.vllm_config.cache_config.mamba_cache_mode == "align":
            cp = counts[nd : nd + npf].to(torch.int64)
            table = common_attn_metadata.block_table_tensor[nd : nd + npf]
            col = ((cp - 1).clamp_min(0) // self.kv_cache_spec.block_size).clamp_max(
                table.shape[1] - 1)
            src = table.gather(1, col[:, None]).squeeze(1)
            extra["computed_before_p"] = cp
            extra["state_src_block_p"] = torch.where(cp > 0, src, torch.zeros_like(src))
        return replace(metadata, **extra) if extra else metadata


class HyenaMamba2Backend(Mamba2AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "EVO2_HYENA_MAMBA2"

    @staticmethod
    def get_builder_cls() -> type[HyenaMamba2MetadataBuilder]:
        return HyenaMamba2MetadataBuilder
