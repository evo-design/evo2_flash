"""Evo 2 MTP draft head as a vLLM EAGLE draft model (no vLLM patches).

The head (reference semantics and training: evo2_layers/mtp_head.py,
scripts/mtp_train.py, MTP_DRAFTER.md) maps (token x_{i+1}, hidden for position i)
to a predicted *post-final-norm* hidden for position i+1:

    z  = fc(cat(norm_e(E[x_{i+1}]), norm_h(hid_i)))
    z += o_proj(Attn(rope(qkv(norm_a(z)))))          one causal attention layer
    z += down(silu(gate) * up)(norm_m(z))            SwiGLU
    ĥ  = norm_o(out(z))                              -> logits = E ĥ (tied LM head)

vLLM drives it through method "eagle": the target (StripedHyena2ForCausalLM,
which declares SupportsEagle) returns its post-norm hidden states; the proposer
runs the head over every newly verified position with those, then K-1 chained
single-token steps feeding ĥ back as `hidden_states`. Embedding and LM head are
not in the head checkpoint: vLLM shares the target's (`has_own_*` = False).

The head's attention uses the target attention's KV layout (32 KV heads x 128,
model dtype), so its cache joins the target's attention KV-cache group, which
vLLM requires of every drafting layer.

Usage:
    speculative_config={"method": "eagle", "model": <head dir>,
                        "num_speculative_tokens": K,
                        "draft_sample_method": "probabilistic"}
<head dir> holds config.json (architectures ["EagleEvo2MTPModel"],
model_type "evo2_mtp") and model.safetensors, written by
scripts/mtp_export_head.py.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import maybe_prefix

from .config import Evo2MTPConfig  # noqa: F401  (re-export)


_ACGT = tuple(ord(c) for c in "ACGT")


@support_torch_compile
class _Inner(nn.Module):
    """Compiled by vLLM (piecewise CUDA graphs split at the attention op): the EAGLE
    proposer only replays PIECEWISE graphs, so an uncompiled draft model runs
    eagerly on every drafting pass."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        # name the attention layer after the target's last layer: unique in the
        # shared static forward context
        layer_id = vllm_config.model_config.hf_config.num_hidden_layers
        H, d = cfg.hidden_size, cfg.head_size
        eps = cfg.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, H,
                                                   prefix=maybe_prefix(prefix, "embed_tokens"))
        self.norm_e = RMSNorm(H, eps=eps)
        self.norm_h = RMSNorm(H, eps=eps)
        self.fc = ReplicatedLinear(2 * H, d, bias=False, return_bias=False,
                                   prefix=maybe_prefix(prefix, "fc"))
        lp = maybe_prefix(prefix, f"layers.{layer_id}")
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.norm_a = RMSNorm(d, eps=eps)
        self.qkv_proj = QKVParallelLinear(d, self.hd, self.n_heads, self.n_kv, bias=False,
                                          prefix=f"{lp}.self_attn.qkv_proj")
        self.o_proj = RowParallelLinear(self.n_heads * self.hd, d, bias=False,
                                        prefix=f"{lp}.self_attn.o_proj")
        # rope cache sized to the engine's context, not the config's 1M ceiling
        max_pos = max(int(vllm_config.model_config.max_model_len) + 64, 4096)
        self.rotary_emb = get_rope(self.hd, max_position=max_pos,
                                   rope_parameters={"rope_theta": cfg.rope_theta},
                                   is_neox_style=True)
        self.attn = Attention(self.n_heads, self.hd, self.hd ** -0.5, num_kv_heads=self.n_kv,
                              cache_config=vllm_config.cache_config,
                              prefix=f"{lp}.self_attn.attn")
        self.norm_m = RMSNorm(d, eps=eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            d, [cfg.intermediate_size] * 2, bias=False, prefix=f"{lp}.mlp.gate_up_proj")
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(cfg.intermediate_size, d, bias=False,
                                           prefix=f"{lp}.mlp.down_proj")
        self.out = ReplicatedLinear(d, H, bias=False, return_bias=False,
                                    prefix=maybe_prefix(prefix, "out"))
        self.norm_o = RMSNorm(H, eps=eps)
        self.q_size = self.n_heads * self.hd
        self.kv_size = self.n_kv * self.hd

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        e = self.embed_tokens(input_ids)
        z = self.fc(torch.cat([self.norm_e(e), self.norm_h(hidden_states)], -1))
        qkv, _ = self.qkv_proj(self.norm_a(z))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        a, _ = self.o_proj(self.attn(q, k, v))
        z = z + a
        m, _ = self.down_proj(self.act(self.gate_up_proj(self.norm_m(z))[0]))
        z = z + m
        return self.norm_o(self.out(z))


class _NullInner(nn.Module):
    """Null drafter (control): no computation; logits are constant (see
    EagleEvo2MTPModel.compute_logits). Not compiled: the compiled pass-through hit
    a CUDA illegal memory access at bs8 (H200, 2026-10-04; cause not chased, it is
    only a control). It holds
    one attention layer only because vLLM requires every EAGLE drafter to own one
    KV-cache-group layer."""

    kind = "uniform"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        layer_id = vllm_config.model_config.hf_config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.hidden_size,
                                                   prefix=maybe_prefix(prefix, "embed_tokens"))
        hd = cfg.head_dim
        self.attn = Attention(cfg.num_attention_heads, hd, hd ** -0.5,
                              num_kv_heads=cfg.num_key_value_heads,
                              cache_config=vllm_config.cache_config,
                              prefix=maybe_prefix(prefix, f"layers.{layer_id}.self_attn.attn"))

    def forward(self, input_ids, positions, hidden_states):
        return hidden_states * 1


class _Evo2MTPBase:
    """Shared by the EAGLE (final hidden only) and EAGLE-3 (multi-layer) heads."""

    has_own_lm_head = False
    has_own_embed_tokens = False

    def _build(self, vllm_config: VllmConfig, prefix: str):
        cfg = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = cfg
        inner = _NullInner if getattr(cfg, "kind", "attn") == "uniform" else _Inner
        self.model = inner(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        # Replaced by the target's (tied) LM head in the proposer's load_model.
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(cfg.vocab_size)
        # EAGLE-3 features: one RMSNorm per target feature (intermediate pre-norm
        # outputs + final hidden), concatenated and fused to hidden_size.
        n_feat = len(getattr(cfg, "eagle_aux_hidden_state_layer_ids", None) or [])
        self.n_feat = n_feat
        if n_feat:
            self.aux_norms = nn.ModuleList(RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
                                           for _ in range(n_feat))
            self.fc_aux = ReplicatedLinear(n_feat * cfg.hidden_size, cfg.hidden_size,
                                           bias=False, return_bias=False,
                                           prefix=maybe_prefix(prefix, "fc_aux"))

    def combine_hidden_states(self, hidden_states):
        """(T, n_feat*H) concatenated target features -> (T, H) depth-1 hidden."""
        chunks = hidden_states.chunk(self.n_feat, dim=-1)
        return self.fc_aux(torch.cat([n(c) for n, c in zip(self.aux_norms, chunks)], -1))

    def embed_input_ids(self, input_ids, *args, **kwargs):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None):
        if inputs_embeds is not None:
            raise NotImplementedError("the Evo 2 MTP head takes token ids only")
        h = self.model(input_ids, positions, hidden_states)
        return h, h

    def compute_logits(self, hidden_states):
        # kind "uniform_attn" (diagnostic): the full head runs -- its attention
        # writes KV like a real drafter -- but q is uniform over ACGT.
        if isinstance(self.model, _NullInner) or getattr(self.config, "kind", "") == "uniform_attn":
            # Null drafter (control): q = uniform over A, C, G, T whatever the context.
            out = torch.full((hidden_states.shape[0], self.config.vocab_size), -1e4,
                             device=hidden_states.device, dtype=torch.float32)
            out[:, list(_ACGT)] = 0.0
            return out
        return self.logits_processor(self.lm_head, hidden_states)

    # checkpoint name -> module parameter (qkv / gate_up are stored fused)
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        if isinstance(self.model, _NullInner):
            for _ in weights:
                pass
            return set(params)   # nothing is used: the logits are constant
        for name, w in weights:
            if name.startswith(("model.embed_tokens", "lm_head")):
                continue  # shared from the target
            if name not in params:
                raise KeyError(f"unexpected draft checkpoint key {name!r}")
            p = params[name]
            if name.endswith("qkv_proj.weight"):
                q, k, v = w.split([self.model.q_size, self.model.kv_size,
                                   self.model.kv_size], 0)
                for shard, t in (("q", q), ("k", k), ("v", v)):
                    p.weight_loader(p, t, shard)
            elif name.endswith("gate_up_proj.weight"):
                g, u = w.chunk(2, 0)
                p.weight_loader(p, g, 0)
                p.weight_loader(p, u, 1)
            else:
                getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(name)
        # embed_tokens / lm_head are filled by the proposer from the target; report
        # them as loaded so the strict loader check does not flag them.
        loaded.update(n for n in params if n.startswith(("model.embed_tokens", "lm_head")))
        return loaded


class EagleEvo2MTPModel(_Evo2MTPBase, nn.Module):
    """vLLM EAGLE draft model: the head fed the target's final post-norm hidden."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self._build(vllm_config, prefix)


class Eagle3Evo2MTPModel(_Evo2MTPBase, Eagle3LlamaForCausalLM):
    """vLLM EAGLE-3 draft model: depth 1 is fed several target layers' features.

    Subclasses Eagle3LlamaForCausalLM only because vLLM 0.26's proposer asserts
    `isinstance(model, (Eagle3LlamaForCausalLM, ...))` for method "eagle3"
    (v1/spec_decode/llm_base_proposer.py:526-537). Nothing of Llama is built or
    used: __init__ bypasses it and every method the proposer calls is ours.
    The features are chosen by `eagle_aux_hidden_state_layer_ids` in the head's
    config (StripedHyena2ForCausalLM.set_aux_hidden_state_layers documents the ids).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self._build(vllm_config, prefix)
        assert self.n_feat, "Eagle3Evo2MTPModel needs eagle_aux_hidden_state_layer_ids"


def speculative_config(head_dir: str, k: int) -> dict:
    """vLLM speculative_config for an exported head dir: method "eagle3" when the
    head takes multi-layer target features, else "eagle"; probabilistic drafting
    (the draft distribution goes to the rejection sampler; fixed-token drafts
    would cap acceptance at p(token))."""
    import json
    import os

    with open(os.path.join(head_dir, "config.json")) as f:
        aux = json.load(f).get("eagle_aux_hidden_state_layer_ids")
    return {"method": "eagle3" if aux else "eagle", "model": head_dir,
            "num_speculative_tokens": k, "draft_sample_method": "probabilistic"}
