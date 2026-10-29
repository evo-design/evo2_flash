"""StripedHyena2ForCausalLM — Evo 2 for vLLM V1.

Two model-level details deviate from vLLM's stock building blocks and are the
reason this cannot reuse a generic hybrid model (see PLAN.md §1.2, and the
golden tests in tests/):

1. `Evo2RMSNorm` — vortex divides by (‖x‖₂·H^-0.5 + eps), i.e. eps is added to
   the *norm*, not inside a sqrt. vLLM's RMSNorm uses rsqrt(mean(x²)+eps) and is
   not interchangeable here, given a residual stream that reaches 1e12–1e17.
2. The gated MLP applies its activation **only on layer 0**
   (`evo2_style_activations`); every later layer is a pure bilinear gate.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
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
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import maybe_prefix

from .config import StripedHyena2Config
from .mixers import HyenaFIRMixer, HyenaIIRMixer
from .triton_norm import rms_norm


class Evo2RMSNorm(nn.Module):
    """y = scale * x / (‖x‖₂ · H^-0.5 + eps).  Note eps outside the sqrt.

    `forward(x, residual)` normalises x + residual and returns (y, x + residual),
    y first so a forward hook still sees the norm's output. On CUDA it runs
    `triton_norm.rms_norm`, which is bit-identical to the torch code below;
    `EVO2_TORCH_NORM=1` forces the torch code (a diagnostic, read per call).
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if x.is_cuda and os.environ.get("EVO2_TORCH_NORM") != "1":
            out = rms_norm(x, self.weight, self.eps, residual)
            if out is not None:
                return out
        if residual is not None:
            x = x + residual
            return self.forward_torch(x), x
        return self.forward_torch(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size**-0.5 + self.eps)
        return self.weight * y


class Evo2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, use_act, prefix,
                 quant_config=None):
        super().__init__()
        self.l1 = ReplicatedLinear(
            hidden_size, intermediate_size, bias=False, prefix=f"{prefix}.l1",
            quant_config=quant_config,
        )
        self.l2 = ReplicatedLinear(
            hidden_size, intermediate_size, bias=False, prefix=f"{prefix}.l2",
            quant_config=quant_config,
        )
        self.l3 = ReplicatedLinear(
            intermediate_size, hidden_size, bias=False, prefix=f"{prefix}.l3",
            quant_config=quant_config,
        )
        self.use_act = use_act

    def forward(self, x):
        z1, _ = self.l1(x)
        z2, _ = self.l2(x)
        if self.use_act:
            z1 = nn.functional.gelu(z1)
        out, _ = self.l3(z1 * z2)
        return out


class Evo2Attention(nn.Module):
    def __init__(self, config, layer_idx, cache_config, prefix, quant_config=None):
        super().__init__()
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = h // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=h,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=h, output_size=h, bias=True, prefix=f"{prefix}.o_proj",
            quant_config=quant_config,
        )
        # Always derive rotary from the config: long-context checkpoints carry a
        # STALE inv_freq buffer for base 1e4 (PLAN.md §1.9).
        rope_parameters = {"rope_theta": config.rope_theta}
        if config.rope_scaling:
            rope_parameters.update(config.rope_scaling)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out, _ = self.o_proj(self.attn(q, k, v))
        return out


class Evo2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, model_config, cache_config, prefix,
                 quant_config=None):
        super().__init__()
        kind = config.layer_kind(layer_idx)
        self.kind = kind
        self.pre_norm = Evo2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = Evo2RMSNorm(config.hidden_size, config.rms_norm_eps)

        if kind == "attn":
            self.mixer = Evo2Attention(
                config, layer_idx, cache_config, f"{prefix}.mixer", quant_config
            )
        elif kind == "hcl":
            self.mixer = HyenaIIRMixer(
                config, layer_idx, model_config, cache_config, f"{prefix}.mixer",
                quant_config=quant_config,
            )
        else:
            self.mixer = HyenaFIRMixer(
                config, layer_idx, model_config, cache_config, f"{prefix}.mixer",
                k_inner=config.inner_filter_length(layer_idx),
                quant_config=quant_config,
            )

        self.mlp = Evo2MLP(
            config.hidden_size,
            config.intermediate_size,
            use_act=(layer_idx == 0),  # evo2_style_activations
            prefix=f"{prefix}.mlp",
            quant_config=quant_config,
        )

    def forward(self, positions, hidden_states, pre=None, input_ids=None):
        # `pre` lets Evo2Model reuse a pre-norm output it already computed (EAGLE-3
        # features) instead of running the norm twice. `input_ids` reach a mixer
        # served from an fp8-surgery token table.
        if pre is None:
            pre = self.pre_norm(hidden_states)
        if self.kind == "attn":
            m = self.mixer(positions, pre)
        else:
            m = self.mixer(pre, token_ids=input_ids)
        post, x = self.post_norm(m, hidden_states)  # x = m + hidden_states
        return self.mlp(post) + x


class Evo2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: StripedHyena2Config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleList(
            [
                Evo2DecoderLayer(
                    config, i, vllm_config.model_config, vllm_config.cache_config,
                    f"{prefix}.layers.{i}",
                    # Embeddings, norms and the Hyena filters stay full precision:
                    # only Linear layers see the quant config. Nothing happens
                    # unless the caller passes `quantization=...`, so bf16
                    # checkpoints behave exactly as before.
                    quant_config=vllm_config.quant_config,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Evo2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.aux_layers: tuple[int, ...] = ()   # EAGLE-3 features, off by default

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    get_input_embeddings = embed_input_ids

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        if not self.aux_layers:
            for layer in self.layers:
                x = layer(positions, x, input_ids=input_ids)
            return self.norm(x)
        aux = []
        for i, layer in enumerate(self.layers):
            if i in self.aux_layers:
                pre = layer.pre_norm(x)
                aux.append(pre)
                x = layer(positions, x, pre, input_ids=input_ids)
            else:
                x = layer(positions, x, input_ids=input_ids)
        h = self.norm(x)
        if len(self.layers) in self.aux_layers:
            aux.append(h)
        return h, aux


class StripedHyena2ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP, SupportsEagle,
                               SupportsEagle3):
    """Evo 2 for vLLM.

    `SupportsEagle`: forward returns the post-final-norm hidden states, which vLLM
    hands to an EAGLE draft model (evo2_flash/mtp_draft.py). `SupportsEagle3`: with
    aux layers set, forward also returns the listed layers' features (see
    `set_aux_hidden_state_layers`) for an EAGLE-3 draft model.
    """

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """EAGLE-3 features. Id i < num_layers: layer i's pre-norm output (its
        RMS-normalised input, the residual stream after block i-1). Id ==
        num_layers: the final post-norm hidden state."""
        n = self.config.num_hidden_layers
        if any(not 0 <= i <= n for i in layers):
            raise ValueError(f"aux layer ids must be in [0, {n}], got {layers}")
        self.model.aux_layers = tuple(int(i) for i in layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        n = self.config.num_hidden_layers
        return (2, n // 2, n - 3, n)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config) -> tuple[torch.dtype, ...]:
        # Must match HyenaMixerBase.unified_state_dtype. The inner-FIR and LI
        # temporal slices are both fp32, and every Hyena layer shares one spec.
        from .mixers import HyenaMixerBase

        return HyenaMixerBase.unified_state_dtype(vllm_config.model_config.dtype)

    @classmethod
    def get_mamba_state_copy_func(cls):
        """One state-copy function per declared state slice.

        Required only on the prefix-caching path: the model runner calls this to
        build the buffers that move cached state between blocks
        (`v1/worker/gpu_model_runner.py::_get_mamba_bufs`). Without it, vLLM
        fails with `TypeError: object of type 'NoneType' has no len()`.

        Outer FIR uses vLLM's extended sliding-convolution layout. Inner FIR and
        LI each store one complete temporal snapshot per verifier block. Their
        separate slices avoid assigning incompatible copy rules to shared state
        during prefix-cache ``align`` postprocessing.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            get_conv_copy_spec,
            get_temporal_copy_spec,
        )

        return (get_conv_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> tuple[tuple[int, ...], ...]:
        """Report the **largest** per-layer layout (Hyena-MR).

        vLLM sizes one mamba page from this and pads every layer's spec up to it
        via `mamba_page_size_padded`. Evo 2 has three different Hyena state
        layouts; MR's 127-tap inner window dominates, so reporting it makes the
        common page large enough for SE and LI too. Cost at 7B: 28.1 vs 12.9
        MiB/seq — negligible against 80 KiB/token of KV.
        """
        from .mixers import HyenaMixerBase

        cfg = vllm_config.model_config.hf_config
        h = cfg.hidden_size
        spec_config = vllm_config.speculative_config
        num_spec = (
            spec_config.num_speculative_tokens if spec_config is not None else 0
        )
        # One source of truth with the layers' declared spec, including the
        # speculative stamp column; a mismatch would size the shared mamba page
        # from a different layout than the layers use.
        return HyenaMixerBase.unified_state_shape(
            h,
            cfg.short_filter_length,
            cfg.hcm_filter_length,
            cfg.state_size,
            num_spec,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = Evo2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    get_input_embeddings = embed_input_ids

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            # depthwise filters are bare Parameters, not Linear modules
            if name.endswith("mixer.short_filter.weight"):
                name = name[: -len(".weight")]
            elif name.endswith("mixer.conv.weight"):
                name = name[: -len(".weight")]
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue
            if name not in params:
                raise KeyError(f"unexpected checkpoint key {name!r}")
            p = params[name]
            loader = getattr(p, "weight_loader", default_weight_loader)
            loader(p, w)
            loaded.add(name)
        return loaded
