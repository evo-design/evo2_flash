"""`evo2_flash.Evo2` — a drop-in replacement for `evo2.Evo2` that runs on vLLM.

    # from evo2 import Evo2
    from evo2_flash import Evo2

    model = Evo2("evo2_7b")
    scores = model.score_sequences(seqs)                       # evo2 semantics
    out = model.generate(["ACGT..."], n_tokens=500)            # vortex GenerationOutput fields
    logits, emb = model(input_ids, return_embeddings=True,
                        layer_names=["blocks.28.mlp.l3"])      # vortex module names

The engine configuration is fixed and built in one place (`engine_kwargs`):
bf16 (mandatory, fp16 overflows), prefix caching on, CUDA graphs on, byte-level
tokenisation handled here (vLLM runs with `skip_tokenizer_init`), no speculative
decoding, no fp8 weight quantisation. `kv_cache_dtype="fp8"` is an optional,
validated opt-in. The engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
because `forward()` captures activations with module hooks on the live model.

Semantic differences from evo2/vortex are listed in README.md, "Drop-in
replacement for evo2.Evo2"; each is also noted at the method it affects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .convert import (
    PUBLISHED_CHECKPOINTS,
    PUBLISHED_PATTERNS,
    CONFIG_MAP,
    FP8_BLOCK0_SENSITIVE,
    HF_MODEL_NAME_MAP,
    MODEL_NAMES,
    build_token_table_checkpoint,
    convert_checkpoint,
    find_checkpoint,
    find_config,
)

__all__ = [
    "Evo2", "CharLevelTokenizer", "GenerationOutput", "MODEL_NAMES",
    "HF_MODEL_NAME_MAP", "CONFIG_MAP", "engine_kwargs", "resolve_checkpoint",
    "checkpoint_for_this_gpu", "reverse_complement", "layer_name_table",
]

DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


# ---------------------------------------------------------------------------
# Tokenizer (vortex CharLevelTokenizer semantics, no vortex dependency)
# ---------------------------------------------------------------------------

class CharLevelTokenizer:
    """Byte-level tokenizer with vortex's `CharLevelTokenizer(512)` semantics.

    `tokenize` returns UTF-8 bytes as a list of Python ints (vortex returns a list
    of numpy uint8 scalars; same values). `detokenize` maps each id through
    `chr(clamp(id, 32, vocab_size))`, so control ids, including EOD 0 and PAD 1,
    become spaces exactly as in vortex.
    """

    def __init__(self, vocab_size: int = 512):
        self.name = "CharLevelTokenizer"
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def eod(self):
        return self.eod_id

    @property
    def eos(self):
        return self.eod_id

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    def decode_token(self, token: int) -> str:
        return chr(self.clamp(int(token)))

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def tokenize_batch(self, text_batch):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        return self.tokenize(text_batch)

    def detokenize(self, token_ids) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(map(self.decode_token, token_ids))

    def detokenize_batch(self, token_ids):
        if isinstance(token_ids, list):
            return [self.detokenize(s) for s in token_ids]
        if isinstance(token_ids, torch.Tensor):
            return [self.detokenize(s) for s in token_ids.tolist()]
        return self.detokenize(token_ids)


# Bio.Seq.reverse_complement's DNA table (IUPAC ambiguity codes, case kept,
# U treated as T); anything else passes through unchanged.
_DNA_COMPLEMENT = dict(zip("ACGTMRWSYKVHDBXNU", "TGCAKYWSRMBDHVXNA"))
_RC_TABLE = str.maketrans(
    "".join(_DNA_COMPLEMENT) + "".join(_DNA_COMPLEMENT).lower(),
    "".join(_DNA_COMPLEMENT.values()) + "".join(_DNA_COMPLEMENT.values()).lower(),
)


def reverse_complement(seq: str) -> str:
    """`str(Bio.Seq.Seq(seq).reverse_complement())` without Biopython."""
    return seq.translate(_RC_TABLE)[::-1]


# ---------------------------------------------------------------------------
# Generation output (vortex.model.generation.GenerationOutput fields)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class GenerationOutput:
    """Same fields as vortex's `GenerationOutput`.

    `logits` is computed lazily on first access (one extra prefill over prompt +
    completion), because vLLM does not keep per-step full-vocabulary logits and
    most callers never read them. Layout matches vortex: one float32 tensor
    `(N, n_tokens, 512)` when all prompts share a length and `batched=True`,
    otherwise one `(1, n_tokens, 512)` tensor per prompt. Values are the raw
    (untempered) logits that predicted each generated token.
    """

    sequences: list[str]
    logprobs_mean: list[float]
    token_ids: list[list[int]] = field(default_factory=list, repr=False)
    # CFG only: the mean log-probability of each sampled token under the
    # *guided* distribution. `logprobs_mean` keeps its vortex-parity meaning
    # (the raw conditional distribution), because vLLM computes its logprobs
    # before the CFG processor runs. Empty for ordinary generation.
    guidance_logprobs_mean: list[float] = field(default_factory=list, repr=False)
    num_cached_tokens: list[int] = field(default_factory=list, repr=False)
    _logits_fn: Callable[[], list] | None = field(default=None, repr=False)
    _logits: list | None = field(default=None, repr=False)

    @property
    def logits(self) -> list[torch.Tensor]:
        if self._logits is None:
            if self._logits_fn is None:
                raise RuntimeError("logits are unavailable for this output")
            self._logits = self._logits_fn()
        return self._logits


# ---------------------------------------------------------------------------
# vortex module names -> this plugin's modules
# ---------------------------------------------------------------------------

# (vortex pattern, plugin module template, what to capture, which layer kinds)
# capture: "out" = module output (first element if a tuple),
#          "in"  = first positional input of the module.
_NAME_RULES = [
    (r"embedding_layer", "model.embed_tokens", "out", None),
    (r"norm", "model.norm", "out", None),
    (r"unembed", None, "logits", None),
    (r"blocks\.(\d+)", "model.layers.{i}", "out", None),
    (r"blocks\.(\d+)\.pre_norm", "model.layers.{i}.pre_norm", "out", None),
    (r"blocks\.(\d+)\.post_norm", "model.layers.{i}.post_norm", "out", None),
    (r"blocks\.(\d+)\.mlp", "model.layers.{i}.mlp", "out", None),
    (r"blocks\.(\d+)\.mlp\.(l1|l2|l3)", "model.layers.{i}.mlp.{sub}", "out", None),
    # Hyena blocks (ParallelGatedConvBlock)
    (r"blocks\.(\d+)\.projections", "model.layers.{i}.mixer.in_proj", "out", "hyena"),
    (r"blocks\.(\d+)\.filter", "model.layers.{i}.mixer.out_proj", "in", "hyena"),
    (r"blocks\.(\d+)\.out_filter_dense", "model.layers.{i}.mixer.out_proj", "out", "hyena"),
    # Attention blocks (AttentionBlock)
    (r"blocks\.(\d+)\.inner_mha_cls", "model.layers.{i}.mixer", "out", "attn"),
    (r"blocks\.(\d+)\.inner_mha_cls\.out_proj", "model.layers.{i}.mixer.o_proj", "out", "attn"),
]


def layer_name_table() -> str:
    """Human-readable mapping of supported vortex names (for docs and errors)."""
    rows = []
    for pat, target, what, kind in _NAME_RULES:
        name = pat.replace(r"(\d+)", "N").replace(r"(l1|l2|l3)", "{l1,l2,l3}").replace("\\.", ".")
        tgt = "lm_head logits (same as the returned logits)" if target is None else (
            target.replace("{i}", "N").replace("{sub}", "{l1,l2,l3}")
            + (" (input)" if what == "in" else ""))
        rows.append(f"  {name:<34} -> {tgt}" + (f"   [{kind} layers only]" if kind else ""))
    return "\n".join(rows)


def _resolve_layer_name(name: str, cfg) -> tuple[str | None, str]:
    """vortex module name (or a native `model.*` name) -> (plugin module, capture)."""
    if name.startswith("model."):
        return name, "out"  # native plugin module name, passed through
    for pat, target, what, kind in _NAME_RULES:
        m = re.fullmatch(pat, name)
        if not m:
            continue
        if target is None:
            return None, what
        i = int(m.group(1)) if m.groups() else None
        if i is not None:
            if not 0 <= i < cfg.num_hidden_layers:
                raise ValueError(f"{name!r}: layer {i} out of range (model has "
                                 f"{cfg.num_hidden_layers} blocks)")
            is_attn = i in cfg.attn_layer_idxs
            if kind == "hyena" and is_attn:
                raise ValueError(f"{name!r}: block {i} is an attention block; use "
                                 f"blocks.{i}.inner_mha_cls instead")
            if kind == "attn" and not is_attn:
                raise ValueError(f"{name!r}: block {i} is a Hyena block; use "
                                 f"blocks.{i}.filter / blocks.{i}.out_filter_dense instead")
        sub = m.group(2) if len(m.groups()) > 1 else None
        return target.format(i=i, sub=sub), what
    raise ValueError(
        f"unsupported layer name {name!r}. Supported vortex names:\n{layer_name_table()}\n"
        "Native plugin module names (e.g. 'model.layers.3.mixer') are also accepted. "
        "Not supported: inner_mha_cls.Wqkv (the converter permutes its rows), "
        "rotary/inner attention internals, and HyenaCascade internals other than "
        "its output ('blocks.N.filter')."
    )


# ---------------------------------------------------------------------------
# Checkpoint resolution and the engine configuration
# ---------------------------------------------------------------------------

def default_cache_dir() -> str:
    """$EVO2_VLLM_CACHE, else $HF_HOME/evo2_vllm (HF_HOME defaults to ~/.cache/huggingface)."""
    if os.environ.get("EVO2_VLLM_CACHE"):
        return os.environ["EVO2_VLLM_CACHE"]
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(home, "evo2_vllm")


def _is_converted(path: str) -> bool:
    cfg = os.path.join(path, "config.json")
    if not (os.path.isdir(path) and os.path.exists(cfg)):
        return False
    with open(cfg) as f:
        return json.load(f).get("model_type") == "striped_hyena2"


def resolve_checkpoint(model_name: str, local_path: str | None = None,
                       cache_dir: str | None = None, download: bool = True,
                       log=print) -> str:
    """Return a converted checkpoint directory, converting on first use.

    - `local_path` a converted directory, or `model_name` itself a path to one:
      used as is.
    - `local_path` a vortex `.pt` (evo2's meaning): converted once into the cache
      with the config of `model_name`.
    - otherwise `model_name` must be an evo2 name: its published converted
      checkpoint (PUBLISHED_CHECKPOINTS) is downloaded; for a model without one,
      `{cache}/{model_name}_hf` is converted from the vortex checkpoint
      (downloaded like evo2 does) if missing.
    """
    if local_path is None and _is_converted(model_name):
        return model_name
    if local_path is not None and _is_converted(local_path):
        return local_path
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Invalid model name {model_name!r}. Should be one of: "
            f"{', '.join(MODEL_NAMES)}, or a converted checkpoint directory."
        )
    cache = cache_dir or default_cache_dir()
    if local_path is not None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"local_path {local_path!r} is neither a converted "
                                    "checkpoint directory nor a vortex .pt file")
        st = os.stat(local_path)
        tag = hashlib.sha1(f"{os.path.abspath(local_path)}:{st.st_size}:{st.st_mtime_ns}"
                           .encode()).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(local_path))[0]
        out = os.path.join(cache, f"{stem}-{tag}_hf")
        src = local_path
    else:
        if download and model_name in PUBLISHED_CHECKPOINTS:
            from huggingface_hub import snapshot_download

            path = snapshot_download(repo_id=PUBLISHED_CHECKPOINTS[model_name],
                                     allow_patterns=PUBLISHED_PATTERNS)
            if not _is_converted(path):
                raise RuntimeError(f"{PUBLISHED_CHECKPOINTS[model_name]} is not a converted "
                                   "checkpoint (config.json model_type != striped_hyena2)")
            return path
        out = os.path.join(cache, f"{model_name}_hf")
        src = None
    if _is_converted(out):
        return out
    if src is None:
        src = find_checkpoint(model_name, download=download)
    log(f"[evo2_flash] converting {model_name} -> {out} (first use only)")
    convert_checkpoint(src, find_config(model_name), out, log=log)
    return out


def checkpoint_for_this_gpu(path: str, cache_dir: str | None = None, log=print,
                            fp8_mode: str = "auto") -> str:
    """The checkpoint this GPU runs correctly: `path`, or its FP8-surgery variant.

    The released 1B, 20B and 40B depend on Hopper's FP8 rounding in block 0
    (FP8_SURGERY.md). A GPU without FP8 tensor cores (sm < 8.9: A100, RTX 3090,
    A40) runs their FP8 in_proj through weight-only Marlin, and they score about
    ln 4 there: 0.805 for the stock 1B on an A100 and an RTX 3090
    (docs/GCP_GPU_MATRIX.md on branch ops/gcp-gpu-matrix). On such a GPU:

    - a checkpoint with a `surgery/` sidecar (the published 1B and 20B; surgery.py)
      uses its FP8-free form, built once into the cache;
    - otherwise the 1B is swapped for its token-table variant, built once into the
      cache (about 20 s; strict gate 6.6e-4);
    - the 20B and 40B raise, since their fix is not a one-step build;
    - everything else, including every 7B and any surgery checkpoint, is
      returned unchanged, as on sm >= 8.9.

    `fp8_mode="surgery"` uses the sidecar's FP8-free form on any GPU (to validate it
    on Hopper). `EVO2_FP8_FALLBACK=off` returns `path` unchanged on any GPU
    (diagnostics).
    """
    from . import surgery

    if fp8_mode not in ("auto", "surgery"):
        raise ValueError(f'fp8_mode must be "auto" or "surgery", got {fp8_mode!r}')
    if fp8_mode == "surgery":
        if not surgery.has_sidecar(path):
            raise ValueError(f'fp8_mode="surgery": {path} has no {surgery.SIDECAR_DIR}/ sidecar')
        return surgery.materialize_cached(path, cache_dir or default_cache_dir(), log=log)
    if os.environ.get("EVO2_FP8_FALLBACK", "auto") == "off" or not torch.cuda.is_available():
        return path
    cc = torch.cuda.get_device_capability()
    if cc >= (8, 9):
        return path
    if surgery.has_sidecar(path):
        log(f"[evo2_flash] {torch.cuda.get_device_name()} has no FP8 tensor cores; using "
            "this checkpoint's FP8-free form")
        return surgery.materialize_cached(path, cache_dir or default_cache_dir(), log=log)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    fix = FP8_BLOCK0_SENSITIVE.get(cfg.get("_source_config"))
    if not cfg.get("use_fp8_input_projections") or fix is None:
        return path
    gpu = f"{torch.cuda.get_device_name()} (sm_{cc[0]}{cc[1]})"
    if fix != "token_table":
        raise RuntimeError(
            f"{path} ({cfg['_source_config']}) needs FP8 tensor cores, which {gpu} lacks: "
            "its block 0 depends on Hopper's FP8 rounding, and through Marlin it scores "
            f"about ln 4. Without FP8 hardware it needs an FP8-surgery checkpoint: {fix} "
            "(FP8_SURGERY.md, scripts/fp8_surgery.py). Use an sm_89+ GPU (L4, L40S, "
            "H100, H200), or set EVO2_FP8_FALLBACK=off to run it anyway."
        )
    st = os.stat(os.path.join(path, "model.safetensors"))
    tag = hashlib.sha1(f"{os.path.abspath(path)}:{st.st_size}:{st.st_mtime_ns}"
                       .encode()).hexdigest()[:10]
    out = os.path.join(cache_dir or default_cache_dir(),
                       f"{os.path.basename(os.path.normpath(path))}-tokentable-{tag}")
    if not _is_converted(out):
        log(f"[evo2_flash] {gpu} has no FP8 tensor cores; building the FP8-surgery "
            f"token-table variant of {path} -> {out} (first use only, FP8_SURGERY.md)")
        build_token_table_checkpoint(path, out, log=log)
    else:
        log(f"[evo2_flash] {gpu} has no FP8 tensor cores; using the token-table "
            f"variant {out} (FP8_SURGERY.md)")
    return out


# Speculative decoding, opt-in, with a trained draft head. It drafts exactly K tokens
# per request, which keeps every verify step on vLLM's FULL CUDA-graph path (10.4 ms
# vs 28.5 ms for a non-uniform step at 7B/bs8; 7.85 ms plain decode). The source
# repo's n-gram drafter is left out: it helped only at batch 1 on 20B/40B.
MTP_DRAFT_TOKENS = 3   # K=4 only helps at batch 1 (MTP_DRAFTER.md)


_FORBIDDEN_OVERRIDES = {
    "dtype": "bfloat16 is mandatory: the residual stream reaches ~1e12 and fp16 gives NaN",
    "skip_tokenizer_init": "Evo 2's byte tokenizer is handled by this class",
    "tokenizer": "Evo 2's byte tokenizer is handled by this class",
    "model": "pass the model as the first argument",
}


def engine_kwargs(ckpt: str, max_model_len: int | None = None,
                  gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                  kv_cache_dtype: str = "auto", speculative: str | None = None,
                  cfg: bool = False, **overrides) -> dict:
    """The one validated engine configuration. Every Evo2 instance is built from this."""
    for key, why in _FORBIDDEN_OVERRIDES.items():
        if key in overrides and not (key == "dtype" and overrides[key] in ("bfloat16", torch.bfloat16)):
            raise ValueError(f"cannot override {key!r}: {why}")
    overrides.pop("dtype", None)
    if max_model_len is None:
        # vLLM auto-fit: the checkpoint's context (the converter writes the
        # trained one, 8192 x rope factor) or the longest single request the KV
        # pool can hold, whichever is smaller. A fixed default cannot be right
        # everywhere: 40B on an H200 has ~10 GiB of KV left after weights and
        # activation reserve, too little for one 131k request.
        max_model_len = -1
    kw = dict(
        model=ckpt,
        skip_tokenizer_init=True,        # byte-level tokenizer, handled here
        dtype="bfloat16",                # mandatory: activations reach ~1e12
        enable_prefix_caching=True,      # shared prompts reuse Hyena + KV state
        enforce_eager=False,             # CUDA graphs: ~1.4-1.65x faster decode
        max_model_len=max_model_len,
        # vLLM's default (1024) needs one Hyena state block per sequence and
        # refuses to start when memory is tight (20B on an H100 at 0.6: 653
        # blocks; 40B on an H200: 246). Throughput is flat well below 128.
        max_num_seqs=128,
        gpu_memory_utilization=gpu_memory_utilization,
        # "fp8" doubles KV capacity; with the fp8-projection checkpoints it moves
        # the upstream gate 1.2-1.8e-3 (7B/20B/40B): a lossy option, off by default.
        kv_cache_dtype=kv_cache_dtype,
    )
    if speculative is not None:
        if os.path.isdir(str(speculative)):
            # A trained MTP/EAGLE draft head (scripts/mtp_export_head.py; MTP_DRAFTER.md):
            # K=3, probabilistic drafting. 7B 1.3-1.7x, 20B 1.3-2.0x over plain decoding.
            from .mtp_draft import speculative_config as _mtp_config
            kw["speculative_config"] = _mtp_config(str(speculative), MTP_DRAFT_TOKENS)
        else:
            raise ValueError("speculative must be None or a trained draft-head "
                             f"directory, got {speculative!r}")
    if cfg:
        # Classifier-free guidance: pairs of requests mixed by a batch-level
        # logits processor (evo2_flash/cfg.py). Registered explicitly here, never
        # through the `vllm.logits_processors` entry point -- an entry point
        # would attach to every engine this plugin builds and so ban speculative
        # decoding globally, for users who never asked for CFG.
        from .cfg import Evo2CfgLogitsProcessor

        if speculative is not None:
            from vllm.v1.sample.logits_processor import (
                STR_SPEC_DEC_REJECTS_LOGITSPROCS,
            )
            raise ValueError(
                f"CFG cannot be combined with speculative decoding: vLLM rejects "
                f"custom logits processors when speculation is on "
                f"({STR_SPEC_DEC_REJECTS_LOGITSPROCS!r}). Build one engine for "
                "CFG and another for the draft head."
            )
        kw["logits_processors"] = [Evo2CfgLogitsProcessor]
        # Both tracks of a pair must be sampled on the same step. Preemption
        # would evict one of them, so leave headroom and keep the scheduler
        # from running ahead of the processor's view of the batch.
        kw["async_scheduling"] = False
        # The offline `LLM` API turns stat logging off, and `get_metrics()`
        # then asserts. `_generate_cfg` reads the preemption counter to prove
        # no pair was ever evicted mid-run, so CFG needs it on.
        kw["disable_log_stats"] = False

    unvalidated = sorted(set(overrides) - {"seed", "max_num_seqs", "max_num_batched_tokens"})
    if unvalidated:
        warnings.warn(f"evo2_flash.Evo2: engine overrides {unvalidated} are outside the "
                      "validated configuration", stacklevel=3)
    kw.update(overrides)
    return kw


def _require_unique_names(names):
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        raise ValueError(f"duplicate layer names: {dup}")


# ---------------------------------------------------------------------------
# The class
# ---------------------------------------------------------------------------

class Evo2:
    """Drop-in replacement for `evo2.Evo2` backed by the vLLM plugin.

    Args (evo2-compatible):
        model_name: an evo2 name (see `MODEL_NAMES`) or a converted checkpoint dir.
        local_path: a converted checkpoint dir, or a vortex `.pt` (converted once).
        use_kernels: accepted for compatibility; ignored.
    Engine args (this class only):
        max_model_len: context limit (default: auto-fit -- the trained context,
            8192 x rope factor, or what GPU memory holds if that is less).
        gpu_memory_utilization: fraction of GPU memory vLLM may use (default 0.85).
        kv_cache_dtype: "auto" (bf16) or "fp8" (2x KV capacity; lossy, ~1-2e-3 nats).
        speculative: None (default) or a trained draft-head directory
            (7B 1.3-1.7x, 20B 1.3-2.0x at batch 1-32). Never default.
        cfg: install the classifier-free-guidance logits processor, enabling
            `generate(..., guidance=...)`. Mutually exclusive with
            `speculative` (vLLM rejects custom logits processors under
            speculative decoding). Off by default; it costs nothing when on but
            unused, since the processor is a no-op for requests without a
            CfgSpec.
        cache_dir: where converted checkpoints live (default `default_cache_dir()`).
        fp8_mode: "auto" (default): FP8 weights on sm_89+, the checkpoint's
            FP8-free form below that; "surgery": the FP8-free form on any GPU.
        **engine_overrides: passed to `vllm.LLM` (warns outside the validated set).

    Attributes: `tokenizer` (CharLevelTokenizer), `llm` (the `vllm.LLM`),
    `model` (the in-process `StripedHyena2ForCausalLM`; plugin module names),
    `checkpoint` (converted directory), `max_model_len`.
    """

    def __init__(self, model_name: str = "evo2_7b", local_path: str | None = None,
                 use_kernels: bool = False, *, max_model_len: int | None = None,
                 gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                 kv_cache_dtype: str = "auto", speculative: str | None = None,
                 cfg: bool = False, cache_dir: str | None = None,
                 fp8_mode: str = "auto", **engine_overrides):
        self.checkpoint = checkpoint_for_this_gpu(
            resolve_checkpoint(model_name, local_path, cache_dir), cache_dir,
            fp8_mode=fp8_mode)
        self.model_name = model_name
        self.tokenizer = CharLevelTokenizer(512)

        # forward() hooks the live model, so the engine must run in this process.
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "0":
            warnings.warn("evo2_flash.Evo2 runs the engine in-process; overriding "
                          "VLLM_ENABLE_V1_MULTIPROCESSING=0", stacklevel=2)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        from . import register
        register()
        from vllm import LLM

        kw = engine_kwargs(self.checkpoint, max_model_len, gpu_memory_utilization,
                           kv_cache_dtype, speculative, cfg, **engine_overrides)
        self.cfg_enabled = cfg
        self.engine_config = dict(kw)
        self.llm = LLM(**kw)
        self.max_model_len = self.llm.llm_engine.vllm_config.model_config.max_model_len
        self._runner_ref = None

    # ---- engine internals --------------------------------------------------

    def _runner(self):
        if self._runner_ref is None:
            def get(worker):
                return getattr(worker, "worker", worker).model_runner
            self._runner_ref = self.llm.collective_rpc(get)[0]
        return self._runner_ref

    @property
    def model(self) -> torch.nn.Module:
        """The live `StripedHyena2ForCausalLM` (plugin module names, not vortex's)."""
        return self._runner().get_model()

    @property
    def block_size(self) -> int:
        return self.llm.llm_engine.vllm_config.cache_config.block_size

    def _check_len(self, n_tokens: int, what: str):
        if n_tokens > self.max_model_len:
            raise ValueError(
                f"{what} needs {n_tokens} positions but max_model_len={self.max_model_len}; "
                f"construct Evo2(..., max_model_len=...) with a larger value"
            )

    def _prompts(self, id_lists):
        from vllm.inputs import TokensPrompt

        return [TokensPrompt(prompt_token_ids=list(ids)) for ids in id_lists]

    # ---- scoring -------------------------------------------------------------

    def _token_logprobs(self, id_lists) -> list[np.ndarray]:
        """log p(x_t | x_<t) for t = 1..L-1 of each sequence, fp32, one vLLM call."""
        from vllm import SamplingParams

        for ids in id_lists:
            if len(ids) == 0:
                raise ValueError("cannot score an empty sequence")
            self._check_len(len(ids) + 1, "scoring a sequence")
        # prompt_logprobs=0 returns only the actual next token's logprob per
        # position; flat_logprobs keeps it in primitive lists (O(L), no dicts).
        # Requesting prompt logprobs makes vLLM skip the prefix-cache *read*
        # (every position must be computed), which is what scoring needs.
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0,
                            flat_logprobs=True, detokenize=False)
        outs = self.llm.generate(self._prompts(id_lists), sp, use_tqdm=False)
        res = []
        for ids, o in zip(id_lists, outs):
            plp = o.prompt_logprobs
            if len(plp) != len(ids):
                raise RuntimeError(f"vLLM returned {len(plp)} prompt positions for a "
                                   f"{len(ids)}-token sequence")
            starts = np.asarray(plp.start_indices[1:], dtype=np.int64)
            tok = np.asarray(plp.token_ids, dtype=np.int64)[starts]
            if not np.array_equal(tok, np.asarray(ids[1:], dtype=np.int64)):
                raise RuntimeError("prompt logprobs are not aligned with the input tokens")
            res.append(np.asarray(plp.logprobs, dtype=np.float32)[starts])
        return res

    def score_sequences(self, seqs: list[str], batch_size: int = 1,
                        prepend_bos: bool = False, reduce_method: str = "mean",
                        average_reverse_complement: bool = False) -> list[float]:
        """evo2 `score_sequences`: reduced log-likelihood of each sequence.

        Position t contributes log p(x_t | x_<t) for t >= 1; with `prepend_bos`,
        EOD (id 0) is prepended and every original token is scored. `reduce_method`
        is 'mean' or 'sum'. `average_reverse_complement` averages the score of the
        sequence and of its reverse complement. All sequences go to vLLM in one
        call; `batch_size` is accepted for compatibility and does not change
        results (in evo2, batch_size>1 with ragged lengths and prepend_bos=False
        also scores one spurious pad position per shorter sequence; this does not).
        Log-softmax is fp32 here; evo2 computes it in bf16.
        """
        if reduce_method == "mean":
            reduce = np.mean
        elif reduce_method == "sum":
            reduce = np.sum
        else:
            raise ValueError(f"Invalid reduce_method {reduce_method}")
        if isinstance(seqs, str):
            seqs = [seqs]
        seqs = list(seqs)
        if not seqs:
            return []
        variants = seqs + ([reverse_complement(s) for s in seqs]
                           if average_reverse_complement else [])
        bos = [self.tokenizer.eod_id] if prepend_bos else []
        lps = self._token_logprobs([bos + self.tokenizer.tokenize(s) for s in variants])
        with np.errstate(all="ignore"):  # a 1-token sequence has nothing to score
            scores = [float(reduce(lp)) for lp in lps]
        if average_reverse_complement:
            n = len(seqs)
            scores = [0.5 * (a + b) for a, b in zip(scores[:n], scores[n:])]
        return scores

    # ---- forward / embeddings -----------------------------------------------

    def _capture(self, id_lists, modules: dict[str, tuple[str | None, str]],
                 want_logits: bool = True):
        """Run each sequence as a prefill-only request; capture per-token tensors.

        `modules`: key -> (plugin module name or None for logits, "out"|"in"|"logits").
        Returns one dict per sequence: key -> (L, D) tensor in position order.

        vLLM packs every scheduled token of every request into one flat batch and
        may split a long prompt across steps (chunked prefill). A pre-hook on the
        decoder stack records, per step, which rows belong to which request and
        at which positions; tensors are re-assembled by (request, position), and
        every position of every sequence must be covered or this raises. CUDA
        graphs are switched off for these steps, since a replayed graph would not
        run Python hooks.
        """
        from vllm import SamplingParams
        from vllm.config import CUDAGraphMode

        for ids in id_lists:
            if len(ids) == 0:
                raise ValueError("cannot run forward on an empty sequence")
            if min(ids) < 0 or max(ids) >= self.tokenizer.vocab_size:
                raise ValueError("token ids must be in [0, 512)")
            self._check_len(len(ids) + 1, "forward over a sequence")

        runner = self._runner()
        core = runner.get_model()
        steps: list[dict] = []
        state = {"open": False}

        def pre_hook(module, args, kwargs):
            input_ids = args[0] if len(args) > 0 else kwargs.get("input_ids")
            positions = args[1] if len(args) > 1 else kwargs.get("positions")
            n = runner.input_batch.num_reqs
            qsl = runner.query_start_loc.np[: n + 1].copy()
            t = int(qsl[-1])
            steps.append({"req_ids": list(runner.input_batch.req_ids[:n]), "qsl": qsl,
                          "pos": positions[:t].clone(), "ids": input_ids[:t].clone(),
                          "t": {}})
            state["open"] = True

        def post_hook(module, args, output):
            state["open"] = False

        def make_hook(key, what):
            def hook(module, args, output):
                if not state["open"]:
                    return
                step = steps[-1]
                if key in step["t"]:
                    raise RuntimeError(f"module for {key!r} ran twice in one step")
                if what == "in":
                    x = args[0]
                elif what == "logits":
                    x = core.compute_logits(output[: len(step["pos"])])
                else:
                    x = output[0] if isinstance(output, tuple) else output
                step["t"][key] = x[: len(step["pos"])].detach().clone()
            return hook

        targets = dict(modules)
        if want_logits:
            targets["__logits__"] = ("model.norm", "logits")
        handles = [core.model.register_forward_pre_hook(pre_hook, with_kwargs=True),
                   core.model.register_forward_hook(post_hook)]
        dispatcher = runner.cudagraph_dispatcher
        saved_mode = dispatcher.cudagraph_mode
        try:
            for key, (mod_name, what) in targets.items():
                mod = core.get_submodule(mod_name or "model.norm")
                handles.append(mod.register_forward_hook(make_hook(key, what)))
            dispatcher.cudagraph_mode = CUDAGraphMode.NONE
            # Prefill only: max_tokens=1 ends each request after its prefill, and
            # skipping the prefix-cache read guarantees every position is computed.
            sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                                skip_reading_prefix_cache=True)
            outs = self.llm.generate(self._prompts(id_lists), sp, use_tqdm=False)
        finally:
            dispatcher.cudagraph_mode = saved_mode
            for h in handles:
                h.remove()

        # vLLM's internal request id is "<external id>-<8 random chars>".
        ext_ids = {o.request_id for o in outs}
        external = {}
        for s, step in enumerate(steps):
            for r, rid in enumerate(step["req_ids"]):
                ext = rid if rid in ext_ids else rid.rsplit("-", 1)[0]
                external.setdefault(ext, []).append((s, r))
        keys = list(targets)
        results = []
        for ids, o in zip(id_lists, outs):
            segs = external.get(o.request_id)
            if not segs:
                raise RuntimeError(f"no forward steps were captured for request {o.request_id}")
            L = len(ids)
            dev = steps[segs[0][0]]["pos"].device
            filled = torch.zeros(L, dtype=torch.bool, device=dev)
            out_t = {}
            for s, r in segs:  # step order: a recomputation (preemption) overwrites
                step = steps[s]
                a, b = int(step["qsl"][r]), int(step["qsl"][r + 1])
                pos = step["pos"][a:b].long()
                keep = pos < L  # ignore any rows past the prompt
                pos = pos[keep]
                if not torch.equal(step["ids"][a:b][keep].long(),
                                   torch.as_tensor(ids, device=dev)[pos]):
                    raise RuntimeError("captured rows do not match the request's tokens")
                for k in keys:
                    if k not in step["t"]:
                        raise RuntimeError(f"{k!r} was not captured (did a CUDA graph replay?)")
                    x = step["t"][k][a:b][keep]
                    if k not in out_t:
                        out_t[k] = torch.empty((L,) + tuple(x.shape[1:]), dtype=x.dtype, device=dev)
                    out_t[k][pos] = x
                filled[pos] = True
            if not bool(filled.all()):
                raise RuntimeError(f"forward capture covered {int(filled.sum())}/{L} positions")
            results.append(out_t)
        return results

    def forward(self, input_ids, return_embeddings: bool = False, layer_names=None):
        """evo2 `forward`: (logits (B, L, 512) bf16, embeddings dict or None).

        Each row of `input_ids` is an independent causal sequence (pad tokens are
        ordinary tokens, as in evo2). `layer_names` use vortex module names, e.g.
        "blocks.28.mlp.l3" (pre-residual MLP output), "blocks.N" (residual stream
        after block N), "norm"; see `layer_name_table()`. Each embedding has shape
        (B, L, D) and the hooked module's dtype. Tensors are on the model's GPU.
        """
        if return_embeddings and layer_names is None:
            raise ValueError(
                "layer_names must be specified when return_embeddings=True. "
                f"Supported vortex names:\n{layer_name_table()}"
            )
        if isinstance(input_ids, torch.Tensor):
            rows = input_ids.detach().long().cpu()
            if rows.dim() == 1:
                rows = rows[None]
            if rows.dim() != 2:
                raise ValueError(f"input_ids must be (B, L), got shape {tuple(input_ids.shape)}")
            id_lists = rows.tolist()
        else:
            id_lists = [list(map(int, r)) for r in input_ids]
            if id_lists and not isinstance(input_ids[0], (list, tuple, np.ndarray, torch.Tensor)):
                id_lists = [list(map(int, input_ids))]
        names = list(layer_names or []) if return_embeddings else []
        _require_unique_names(names)
        cfg = self.model.config
        modules = {n: _resolve_layer_name(n, cfg) for n in names}
        core = self.model
        for n, (mod, _) in modules.items():
            if mod is not None:
                try:
                    core.get_submodule(mod)
                except AttributeError:
                    raise ValueError(f"layer name {n!r}: no module {mod!r} in this model") from None
        modules = {n: v for n, v in modules.items() if v[1] != "logits"}
        caps = self._capture(id_lists, modules, want_logits=True)
        logits = torch.stack([c["__logits__"] for c in caps])
        if not return_embeddings:
            return logits, None
        emb = {}
        for n in names:
            emb[n] = logits if n == "unembed" else torch.stack([c[n] for c in caps])
        return logits, emb

    def __call__(self, input_ids, return_embeddings: bool = False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    # ---- generation ------------------------------------------------------------

    @staticmethod
    def sampling_kwargs(temperature: float, top_k: int, top_p: float) -> dict:
        """vortex `sample()` semantics as vLLM SamplingParams arguments.

        vortex: top_k == 1 is argmax (temperature ignored); otherwise the top-k
        logits are divided by the temperature and top-p filters the renormalised
        top-k distribution (top_k <= 0: whole vocabulary; top_p <= 0 or >= 1: no
        top-p). vLLM applies temperature, then top-k, then top-p over the masked
        distribution, which selects the same set and the same probabilities.
        temperature == 0 means greedy here; vortex divides by zero (NaN) unless
        top_k == 1.
        """
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_p > 1.0:
            raise ValueError("top-p should be in (0, 1].")
        if top_k == 1 or temperature == 0:
            return {"temperature": 0.0}
        return {
            "temperature": float(temperature),
            "top_k": int(min(top_k, 512)) if top_k > 0 else 0,
            "top_p": float(top_p) if 0.0 < top_p < 1.0 else 1.0,
        }

    def generate(self, prompt_seqs: list[str], n_tokens: int = 500,
                 temperature: float = 1.0, top_k: int = 4, top_p: float = 1.0,
                 batched: bool = True, cached_generation: bool = True,
                 verbose: int = 1, force_prompt_threshold: int | None = None,
                 *, seed: int | None = None, prime_prefix_cache: bool = False,
                 negative_prompts: list[str] | str | None = None,
                 guidance: float | list[float] | None = None,
                 cfg_type: str | None = None, dna_only: bool = False,
                 baseline_prompt: str = "ATGCATGA",
                 on_desync: str = "raise",
                 ) -> GenerationOutput:
        """evo2 `generate`: n_tokens per prompt, vortex sampling semantics.

        All prompts are submitted in one vLLM call (continuous batching; ragged
        lengths are fine). Exactly `n_tokens` are generated per prompt; like
        vortex's evo2 path, EOD does not stop generation. `logprobs_mean` is the
        mean log-probability of the sampled tokens under the raw (untempered,
        unfiltered) model distribution, as in vortex.

        `batched`, `cached_generation` and `force_prompt_threshold` are accepted
        for compatibility; vLLM always batches, caches state, and bounds prefill
        memory with chunked prefill. Extra (keyword-only): `seed` makes request i
        use seed+i. Repeated prompts already share prefill through the prefix
        cache: copies scheduled after the first prefill step read it. Only the
        copies admitted in that first step (at most max_num_batched_tokens /
        prompt length) recompute; `prime_prefix_cache=True` prefills each repeated
        prompt once beforehand to avoid even that. Measured on 32 identical
        4000-nt prompts it made no difference (1B and 7B), so it is off.
        `guidance` (and the CFG-only arguments beside it) switch to the
        classifier-free-guidance path, `_generate_cfg`:

            guided = l_cond + guidance * (l_cond - l_base)

        A general steering mechanism, specified entirely by two prompts and a
        scalar. `prompt_seqs[i]` is the conditional prompt; the baseline is
        `baseline_prompt` (`cfg_type="uncond"`, steer along whatever the
        conditional prompt adds over an unconditional one) or
        `negative_prompts[i]` (`cfg_type="negative"`, steer toward the prompt
        and away from the negative). Nothing here is specific to any
        application -- callers compose the prompts, and that composition is the
        only thing that differs between use cases. `dna_only=True` restricts
        sampling to A/C/G/T. CFG uses one seeded sampling stream per generate
        call (restarted by an explicit seed), advancing across admission waves.
        The stream depends on the active pair batch shape; equal seeds do not
        promise the same strings across engines or different batch sizes.

        `guidance=0.0` is a CFG run at w=0, not "no CFG": it must be requested
        explicitly and, by the convention this follows, reproduces plain
        conditional generation exactly (gate G1 in
        `tests/test_cfg_consistency.py`).
        """
        from vllm import SamplingParams

        # CFG dispatch. `guidance=0.0` is a legitimate request, so this tests
        # for None, never for truth. `cfg_type="none"` is accepted as "plain",
        # matching the reference's flag vocabulary.
        cfg_requested = [
            name
            for name, value in (("guidance", guidance),
                                ("negative_prompts", negative_prompts),
                                ("cfg_type", cfg_type),
                                ("dna_only", dna_only or None))
            if value is not None
        ]
        if cfg_type == "none":
            # The reference's name for "not a CFG run" (CFG_REFERENCE.md §5).
            if cfg_requested != ["cfg_type"]:
                raise ValueError(
                    "cfg_type='none' is plain generation; drop the other CFG "
                    f"arguments {[n for n in cfg_requested if n != 'cfg_type']}"
                )
            cfg_requested = []
        if cfg_requested:
            if guidance is None:
                raise ValueError(
                    f"{cfg_requested} require `guidance`; pass guidance=0.0 for a "
                    "CFG run at w=0 (which reproduces plain generation), or drop "
                    "them for plain generation"
                )
            return self._generate_cfg(
                prompt_seqs, n_tokens=n_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, verbose=verbose, seed=seed,
                negative_prompts=negative_prompts, guidance=guidance,
                cfg_type=cfg_type, dna_only=dna_only,
                baseline_prompt=baseline_prompt,
                on_desync=on_desync,
            )

        if isinstance(prompt_seqs, str):
            prompt_seqs = [prompt_seqs]
        prompt_seqs = list(prompt_seqs)
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive")
        ids = [self.tokenizer.tokenize(s) for s in prompt_seqs]
        for x in ids:
            if not x:
                raise ValueError("empty prompt")
            self._check_len(len(x) + n_tokens, "prompt + n_tokens")
        base = self.sampling_kwargs(temperature, top_k, top_p)
        params = [
            SamplingParams(max_tokens=n_tokens, min_tokens=n_tokens, ignore_eos=True,
                           logprobs=0, flat_logprobs=True, detokenize=False,
                           seed=None if seed is None else seed + i, **base)
            for i in range(len(ids))
        ]

        if prime_prefix_cache:
            counts = Counter(prompt_seqs)
            prime = [self.tokenizer.tokenize(s) for s, c in counts.items()
                     if c > 1 and len(s) >= self.block_size]
            if prime:
                self.llm.generate(self._prompts(prime),
                                  SamplingParams(max_tokens=1, temperature=0.0, detokenize=False),
                                  use_tqdm=False)

        outs = self.llm.generate(self._prompts(ids), params, use_tqdm=verbose > 1)
        seqs, means, toks, cached = [], [], [], []
        for o in outs:
            c = o.outputs[0]
            t = list(c.token_ids)
            if len(t) != n_tokens:
                raise RuntimeError(f"generated {len(t)} tokens, expected {n_tokens} "
                                   f"(finish_reason={c.finish_reason})")
            lp = np.asarray(c.logprobs.logprobs, dtype=np.float32)
            starts = np.asarray(c.logprobs.start_indices, dtype=np.int64)
            seqs.append(self.tokenizer.detokenize(t))
            means.append(float(np.mean(lp[starts])))
            toks.append(t)
            cached.append(int(getattr(o, "num_cached_tokens", 0) or 0))

        uniform = all(len(x) == len(ids[0]) for x in ids)

        def logits_fn():
            caps = self._capture([p + t[:-1] for p, t in zip(ids, toks)], {}, want_logits=True)
            per = [c["__logits__"][len(p) - 1:].float() for p, c in zip(ids, caps)]
            if batched and uniform:
                return [torch.stack(per)]
            return [x[None] for x in per]

        if verbose:
            for seq, score, prompt in zip(seqs, means, prompt_seqs):
                print(f'Prompt: "{prompt}",\tOutput: "{seq}",\tScore: {score}')
        return GenerationOutput(sequences=seqs, logprobs_mean=means, token_ids=toks,
                                num_cached_tokens=cached, _logits_fn=logits_fn)

    # ---- classifier-free guidance -------------------------------------------

    def dna_only_ids(self) -> tuple[int, ...]:
        """Token ids for A, C, G, T, from the live tokenizer.

        Uppercase only, matching the reference. Lowercase bases, `N`, any
        delimiter characters and EOD are all blocked when `dna_only=True`.
        """
        ids = sorted({int(t) for c in "ACGT" for t in self.tokenizer.tokenize(c)})
        if len(ids) != 4:
            raise RuntimeError(
                f"expected one token per base, got {ids} -- the tokenizer is not "
                "the byte-level one this assumes"
            )
        return tuple(ids)

    def _preemptions(self) -> int:
        """Cumulative preemptions, which a CFG run must not have any of.

        `engine_kwargs(cfg=True)` sets `disable_log_stats=False` for this; the
        explicit error is for an engine built with stats off by an override,
        where the guard would otherwise disappear behind an AssertionError.
        """
        if not self.llm.llm_engine.log_stats:
            raise RuntimeError(
                "CFG needs engine stats (disable_log_stats=False) so the run can "
                "prove no CFG pair was preempted; a preempted request's partner "
                "is sampled without guidance"
            )
        return sum(
            int(getattr(m, "value", 0) or 0)
            for m in self.llm.get_metrics()
            if "preempt" in m.name
        )

    def _cfg_pairs_in_kv(self, max_len: int) -> int:
        """How many CFG pairs of up to `max_len` tokens the KV cache holds at once.

        vLLM admits requests one at a time as blocks allow, so a wave bigger than the
        cache admits one track of a pair and leaves the other waiting: `Evo2CfgDesync`
        on a 24 GB L4, where the cache held 9 of a wave's 16 tracks
        (docs/GCP_GPU_MATRIX.md on branch ops/gcp-gpu-matrix). Count blocks as vLLM's
        own concurrency estimate does: an attention group needs one block per
        `block_size` tokens, a Hyena (Mamba) group its fixed state blocks (2 in prefix
        caching's "align" mode). Prefix-cache sharing is not counted, so the bound is
        conservative.
        """
        from vllm.v1.kv_cache_interface import MambaSpec

        engine = self.llm.llm_engine
        kv = engine.engine_core.engine_core.scheduler.kv_cache_config
        need = 0
        for group in kv.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                need += -(-spec.max_memory_usage_bytes(engine.vllm_config)
                          // spec.page_size_bytes)
            else:
                need += -(-max_len // spec.block_size)
        return (kv.num_blocks - 1) // (2 * need)  # one block is vLLM's null block

    def _generate_cfg(self, prompt_seqs, *, n_tokens, temperature, top_k, top_p,
                      verbose, seed, negative_prompts, guidance, cfg_type,
                      dna_only, baseline_prompt, on_desync) -> GenerationOutput:
        """CFG generation: 2N paired requests mixed by the logits processor.

        One pair per (prompt, guidance) combination. The returned sequences are
        the conditional track's; the baseline track exists only to be
        contrasted against, and by construction emits the same tokens.
        """
        from vllm import SamplingParams

        from .cfg import CfgSpec, current_processor

        if not self.cfg_enabled:
            raise RuntimeError(
                "guidance requires the CFG logits processor: construct "
                "Evo2(..., cfg=True)"
            )
        if isinstance(prompt_seqs, str):
            prompt_seqs = [prompt_seqs]
        prompt_seqs = list(prompt_seqs)
        if not prompt_seqs:
            raise ValueError("no prompts")
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive")

        if cfg_type is None:
            cfg_type = "negative" if negative_prompts is not None else "uncond"
        if cfg_type not in ("uncond", "negative"):
            raise ValueError("cfg_type must be 'uncond' or 'negative' "
                             "(cfg_type='none' is plain generate without guidance)")

        if cfg_type == "negative":
            if negative_prompts is None:
                raise ValueError("cfg_type='negative' needs negative_prompts")
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts] * len(prompt_seqs)
            negative_prompts = list(negative_prompts)
            if len(negative_prompts) != len(prompt_seqs):
                raise ValueError(
                    f"{len(negative_prompts)} negative prompts for "
                    f"{len(prompt_seqs)} prompts"
                )
            bases = negative_prompts
        else:
            if negative_prompts is not None:
                raise ValueError("negative_prompts requires cfg_type='negative'")
            # An unconditional baseline needs *some* prompt, and it must
            # tokenise to at least two tokens or there is nothing to prefill.
            # The content should be neutral; what counts as neutral is the
            # caller's call, so it is an argument.
            if len(self.tokenizer.tokenize(baseline_prompt)) < 2:
                raise ValueError(
                    f"the uncond baseline prompt {baseline_prompt!r} is shorter "
                    "than two tokens; pass a longer `baseline_prompt`"
                )
            bases = [baseline_prompt] * len(prompt_seqs)

        if guidance is None:
            guidance = 0.0
        ws = [float(guidance)] if isinstance(guidance, (int, float)) else [
            float(w) for w in guidance
        ]
        if not ws:
            raise ValueError("guidance must not be an empty list")

        allowed = self.dna_only_ids() if dna_only else None
        base_sampling = dict(temperature=float(temperature), top_k=int(top_k),
                             top_p=float(top_p), dna_only_ids=allowed,
                             seed=seed, on_desync=on_desync)

        # One pair per (prompt, w). `pairs[k]` is (prompt index, w).
        pairs = [(i, w) for i in range(len(prompt_seqs)) for w in ws]
        cond_ids = [self.tokenizer.tokenize(prompt_seqs[i]) for i, _ in pairs]
        base_ids = [self.tokenizer.tokenize(bases[i]) for i, _ in pairs]
        for ids, what in ((cond_ids, "prompt"), (base_ids, "baseline prompt")):
            for x in ids:
                if not x:
                    raise ValueError(f"empty {what}")
                self._check_len(len(x) + n_tokens, f"{what} + n_tokens")

        # Pre-prime every distinct prompt into the prefix cache. Without this a
        # long conditional prompt and a short baseline prompt can finish their
        # prefills on different steps, which would leave one track of the pair
        # sampled without guidance -- the processor detects that and raises, so
        # this is what keeps a legitimate run from tripping the guard.
        distinct = {tuple(x) for x in cond_ids} | {tuple(x) for x in base_ids}
        self.llm.generate(
            self._prompts(sorted(distinct, key=len)),
            SamplingParams(max_tokens=1, temperature=0.0, detokenize=False),
            use_tqdm=False,
        )

        processor = current_processor()
        if processor is None:
            raise RuntimeError(
                "the CFG logits processor was not constructed; the engine must "
                "run in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)"
            )
        processor.reset_run()

        # Both tracks of every pair must be resident in the same step, so never
        # submit more pairs than the batch, or the KV cache, can hold at once.
        longest = max(len(x) for x in cond_ids + base_ids) + n_tokens
        per_wave = max(1, min(self.llm.llm_engine.vllm_config
                              .scheduler_config.max_num_seqs // 2,
                              self._cfg_pairs_in_kv(longest)))
        preempt_before = self._preemptions()

        seqs, means, toks, cached, guided = [], [], [], [], []
        for start in range(0, len(pairs), per_wave):
            wave = range(start, min(start + per_wave, len(pairs)))
            prompts, params = [], []
            for k in wave:
                _, w = pairs[k]
                for role, ids in (("cond", cond_ids[k]), ("base", base_ids[k])):
                    spec = CfgSpec(pair_id=k, role=role, guidance=w,
                                   **base_sampling)
                    prompts.append(self._prompts([ids])[0])
                    params.append(SamplingParams(
                        max_tokens=n_tokens, min_tokens=n_tokens, ignore_eos=True,
                        temperature=0.0, logprobs=0, flat_logprobs=True,
                        detokenize=False, extra_args=spec.to_extra_args(),
                    ))
            outs = self.llm.generate(prompts, params, use_tqdm=verbose > 1)

            for j, k in enumerate(wave):
                cond_out = outs[2 * j].outputs[0]
                base_out = outs[2 * j + 1].outputs[0]
                t = list(cond_out.token_ids)
                if len(t) != n_tokens:
                    raise RuntimeError(
                        f"pair {k} generated {len(t)} tokens, expected {n_tokens} "
                        f"(finish_reason={cond_out.finish_reason})"
                    )
                if list(base_out.token_ids) != t:
                    raise RuntimeError(
                        f"pair {k}: the two tracks emitted different tokens even "
                        "though the processor forced a one-hot into both rows"
                    )
                lp = np.asarray(cond_out.logprobs.logprobs, dtype=np.float32)
                starts = np.asarray(cond_out.logprobs.start_indices, dtype=np.int64)
                seqs.append(self.tokenizer.detokenize(t))
                means.append(float(np.mean(lp[starts])))
                toks.append(t)
                cached.append(int(getattr(outs[2 * j], "num_cached_tokens", 0) or 0))

        preempted = self._preemptions() - preempt_before
        if preempted:
            raise RuntimeError(
                f"{preempted} preemption(s) during a CFG run: a preempted "
                "request's partner is sampled without guidance, so the run is "
                "not trustworthy. Lower the number of prompts, raise "
                "gpu_memory_utilization, or shorten the prompts."
            )

        recorded = processor.guidance_logprob_means()
        guided = [recorded.get(k, float("nan")) for k in range(len(pairs))]

        if verbose:
            for (i, w), seq, score in zip(pairs, seqs, means):
                print(f'Prompt: "{prompt_seqs[i]}",\tw={w},\tOutput: "{seq}",'
                      f'\tScore: {score}')
        return GenerationOutput(sequences=seqs, logprobs_mean=means,
                                token_ids=toks, num_cached_tokens=cached,
                                guidance_logprobs_mean=guided)
