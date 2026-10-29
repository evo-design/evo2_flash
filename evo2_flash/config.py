"""HF-style config for Evo 2 / StripedHyena2, as emitted by the converter."""

from transformers import PretrainedConfig


class StripedHyena2Config(PretrainedConfig):
    model_type = "striped_hyena2"

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 1920,
        num_hidden_layers: int = 25,
        num_attention_heads: int = 15,
        num_key_value_heads: int | None = None,
        intermediate_size: int = 5120,
        attn_layer_idxs: tuple[int, ...] = (),
        hcs_layer_idxs: tuple[int, ...] = (),
        hcm_layer_idxs: tuple[int, ...] = (),
        hcl_layer_idxs: tuple[int, ...] = (),
        short_filter_length: int = 3,
        hcs_filter_length: int = 7,
        hcm_filter_length: int = 128,
        state_size: int = 16,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,
        interleave: bool = True,
        column_split_hyena: bool = False,
        hyena_flip_x1x2: bool = False,
        conv_kernels_pre_flipped: bool = True,
        mlp_activation: str = "gelu",
        use_fp8_input_projections: bool = False,
        hyena_in_proj_fp8: dict | None = None,
        hyena_token_table_layers: tuple[int, ...] = (),
        hyena_fp8_emulated_layers: tuple[int, ...] = (),
        hyena_fp8_emulated_channels: dict | None = None,
        fp8_surgery: dict | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.intermediate_size = intermediate_size
        self.attn_layer_idxs = list(attn_layer_idxs)
        self.hcs_layer_idxs = list(hcs_layer_idxs)
        self.hcm_layer_idxs = list(hcm_layer_idxs)
        self.hcl_layer_idxs = list(hcl_layer_idxs)
        self.short_filter_length = short_filter_length
        self.hcs_filter_length = hcs_filter_length
        self.hcm_filter_length = hcm_filter_length
        self.state_size = state_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.interleave = interleave
        self.column_split_hyena = column_split_hyena
        self.hyena_flip_x1x2 = hyena_flip_x1x2
        self.conv_kernels_pre_flipped = conv_kernels_pre_flipped
        self.mlp_activation = mlp_activation
        self.use_fp8_input_projections = use_fp8_input_projections
        self.hyena_in_proj_fp8 = hyena_in_proj_fp8
        # FP8 surgery (FP8_SURGERY.md, scripts/fp8_surgery.py): the listed Hyena
        # layers replace in_proj by a (vocab, 3H) table holding the FP8
        # fast-accumulation projection of each token. Only layer 0 qualifies:
        # its in_proj input is pre_norm(embedding[token]), a function of the
        # token alone, so the table is exact rather than an approximation.
        self.hyena_token_table_layers = list(hyena_token_table_layers or ())
        self.fp8_surgery = fp8_surgery
        # Layers whose in_proj keeps its E4M3 operands and runs the software model
        # of Hopper fast accumulation (fp8_emul_triton.py) on any GPU.
        self.hyena_fp8_emulated_layers = list(hyena_fp8_emulated_layers or ())
        if self.hyena_fp8_emulated_layers:
            hy = set(hcs_layer_idxs) | set(hcm_layer_idxs) | set(hcl_layer_idxs)
            bad = [i for i in self.hyena_fp8_emulated_layers if i not in hy]
            if bad or 0 in self.hyena_token_table_layers and 0 in self.hyena_fp8_emulated_layers:
                raise ValueError(f"invalid hyena_fp8_emulated_layers {self.hyena_fp8_emulated_layers}")
            if use_fp8_input_projections or not isinstance(fp8_surgery, dict):
                raise ValueError("hyena_fp8_emulated_layers is FP8-surgery output: needs "
                                 "use_fp8_input_projections=False and fp8_surgery provenance")
        # Hybrid layers: a bf16 in_proj whose calibrated worst output channels are
        # recomputed by the emulated FP8 GEMM. {layer: number of channels}.
        self.hyena_fp8_emulated_channels = {
            int(k): int(v) for k, v in (hyena_fp8_emulated_channels or {}).items()}
        if self.hyena_fp8_emulated_channels:
            hy = set(hcs_layer_idxs) | set(hcm_layer_idxs) | set(hcl_layer_idxs)
            bad = {i: n for i, n in self.hyena_fp8_emulated_channels.items()
                   if i not in hy or i in self.hyena_fp8_emulated_layers
                   or i in self.hyena_token_table_layers or not 0 < n < 3 * hidden_size}
            if bad or use_fp8_input_projections or not isinstance(fp8_surgery, dict):
                raise ValueError(f"invalid hyena_fp8_emulated_channels {bad or ''}: needs "
                                 "FP8-surgery provenance, Hyena layers, 0 < n < 3H")
        if self.hyena_token_table_layers:
            if self.hyena_token_table_layers != [0]:
                raise ValueError(
                    "hyena_token_table_layers may only be [0]: only layer 0's "
                    "projection input depends on the token alone, got "
                    f"{self.hyena_token_table_layers}"
                )
            if 0 in self.attn_layer_idxs:
                raise ValueError("layer 0 is attention; a token table needs Hyena")
            if use_fp8_input_projections:
                raise ValueError(
                    "hyena_token_table_layers is the output of FP8 surgery and "
                    "cannot be combined with use_fp8_input_projections=True"
                )
            if not isinstance(fp8_surgery, dict):
                raise ValueError("token-table checkpoints must carry fp8_surgery "
                                 "provenance; regenerate with scripts/fp8_surgery.py")

        # Released checkpoints that opt into this path were trained with
        # Transformer Engine E4M3 input projections.  Treating their stored bf16
        # master weights as ordinary linears is catastrophically wrong (the 1B
        # upstream NLL changes from 0.5026 to 1.388).  New converter output uses
        # vLLM's serialized, static W8A8 contract.  Refuse partial/manual configs
        # so a missing scale can never degrade correctness silently.
        quant = kwargs.get("quantization_config")
        if use_fp8_input_projections:
            if not isinstance(hyena_in_proj_fp8, dict):
                raise ValueError(
                    "use_fp8_input_projections=True requires converter-exported "
                    "hyena_in_proj_fp8 metadata; reconvert the source checkpoint"
                )
            expected_marker = {
                "required": True,
                "runtime": "vllm_fp8_static_w8a8_v1",
                "format": "e4m3fn",
                "fp8_max": 448,
                "scale_scheme": "per_tensor_static",
                "scale_convention": "dequantization",
                "input_scale_source": "scale_inv_fwd[0]",
                "weight_scale_source": "scale_inv_fwd[1]",
                "metadata_file": "metadata/hyena_in_proj_fp8_meta.safetensors",
            }
            bad_marker = {
                key: hyena_in_proj_fp8.get(key)
                for key, value in expected_marker.items()
                if hyena_in_proj_fp8.get(key) != value
            }
            if bad_marker:
                raise ValueError(
                    "unsupported or incomplete hyena_in_proj_fp8 metadata: "
                    f"{bad_marker}"
                )
            hyena_layers = (
                set(self.hcs_layer_idxs)
                | set(self.hcm_layer_idxs)
                | set(self.hcl_layer_idxs)
            )
            if hyena_in_proj_fp8.get("num_layers") != len(hyena_layers):
                raise ValueError(
                    "hyena_in_proj_fp8 num_layers does not match the layer map: "
                    f"{hyena_in_proj_fp8.get('num_layers')} != {len(hyena_layers)}"
                )
            if quant is not None:
                raise ValueError(
                    "FP8 Hyena projections use the plugin's selective static-W8A8 "
                    "runtime; an embedded global quantization_config would also "
                    "quantize attention KV cache and is unsupported"
                )
        elif hyena_in_proj_fp8 is not None:
            raise ValueError(
                "hyena_in_proj_fp8 metadata is present while "
                "use_fp8_input_projections=False"
            )

        # `layer_types` is the convention vLLM uses to find the full-attention
        # layers of a hybrid model (Qwen3.5/Next style; the alternatives are
        # `layers_block_type` and `attn_type_list`). Derived here rather than
        # stored in config.json, so checkpoints converted before this existed pick
        # it up for free. Required by TurboQuant KV-cache quantisation, which
        # otherwise raises "requires identifiable full-attention layers".
        #
        # Nothing else in vLLM reads it for a non-Gemma4, non-spec-decode model,
        # so adding it does not change the default path.
        # Only the "full_attention" entries are consumed; transformers validates
        # the labels against a fixed vocabulary, so the Hyena layers borrow
        # "linear_attention" (the label vLLM's Mamba-family hybrids use).
        self.layer_types = [
            "full_attention" if i in self.attn_layer_idxs else "linear_attention"
            for i in range(num_hidden_layers)
        ]
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def layer_kind(self, i: int) -> str:
        if i in self.attn_layer_idxs:
            return "attn"
        if i in self.hcs_layer_idxs:
            return "hcs"
        if i in self.hcm_layer_idxs:
            return "hcm"
        if i in self.hcl_layer_idxs:
            return "hcl"
        raise ValueError(f"layer {i} is in no layer-type list")

    def inner_filter_length(self, i: int) -> int:
        return {
            "hcs": self.hcs_filter_length,
            "hcm": self.hcm_filter_length,
        }[self.layer_kind(i)]


class Evo2MTPConfig(PretrainedConfig):
    """Config of the MTP / EAGLE draft head (evo2_flash/mtp_draft.py, MTP_DRAFTER.md)."""

    model_type = "evo2_mtp"

    def __init__(self, hidden_size=4096, head_size=4096, num_attention_heads=32,
                 num_key_value_heads=32, intermediate_size=11008, vocab_size=512,
                 rope_theta=10000.0, rms_norm_eps=1e-6, num_hidden_layers=1,
                 max_position_embeddings=1_048_576, kind="attn", **kwargs):
        self.hidden_size = hidden_size            # target hidden (fc input / ĥ size)
        self.head_size = head_size                # the head's internal width d
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.kind = kind
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
