"""Phase 1 — convert a vortex Evo 2 `.pt` checkpoint into an HF-style directory.

Emits `model.safetensors` + `config.json` with renamed, pre-transformed weights so
the vLLM model class can load them with a flat name map and no load-time surgery.

Three transformations matter for correctness; all three are silent if wrong:

1. **Wqkv permutation.** The checkpoint stores qkv interleaved per head as
   `[hidden, num_heads, 3, head_dim]`. vortex permutes it to the standard
   `[Wq; Wk; Wv]` row layout at load time (`custom_load_state_dict`, gated on
   `column_split: True`). We apply the same permutation here so the exported
   weight is directly usable by a standard fused-QKV linear.

2. **Grouped filter expansion.** `filter.h` is stored as `(G, 1, K)` with G < H
   (e.g. 128 groups over 1920 channels) and expanded via `repeat_interleave(H//G)`
   at run time. We pre-expand to `(H, 1, K)` so it becomes a plain depthwise conv.

3. **fp32 filter params.** `log_poles` / `residues` stay float32; everything else
   is bfloat16. Casting these to bf16 measurably changes the long-conv response.

4. **Conv kernel convention.** Hyena-SE (K=7) prefills via `F.conv1d`, which is
   cross-correlation, and its decode `step_fir` uses the kernel unflipped.
   Hyena-MR (K=128) prefills via `fftconv_func`, a true convolution, and its
   decode sets `flip_filter=True`. `causal_conv1d` follows the cross-correlation
   convention, so MR kernels are pre-flipped here to put SE and MR on a single
   convention. Flagged as `conv_kernels_pre_flipped` in the emitted config.

For checkpoints whose source config enables ``use_fp8_input_projections``, the
Hyena input projections are exported as selective static E4M3 W8A8 layers.  The
checkpoint's Transformer Engine scales are authoritative: dropping them and
running the stored bf16 weights changes the released 1B model from NLL 0.5026
to 1.388 on the upstream correctness prompts.  The converter therefore fails
if any required projection metadata is missing or malformed.

Use `convert_checkpoint()` from Python (evo2_flash.Evo2 calls it on first use of a
model name), or the CLI wrapper:
  python scripts/convert_evo2_to_hf.py --model evo2_1b_base --out ckpt/evo2_1b_base_hf
"""

import glob
import io
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

# evo2 model name -> config yml, as in evo2/utils.py (evo2 0.6.0). The ymls are
# vendored verbatim from the evo2 package (Apache-2.0) in evo2_configs/, so
# conversion does not need evo2 or vortex installed.
CONFIG_MAP = {
    "evo2_7b": "evo2-7b-1m.yml",
    "evo2_40b": "evo2-40b-1m.yml",
    "evo2_20b": "evo2-20b-1m.yml",
    "evo2_7b_base": "evo2-7b-8k.yml",
    "evo2_40b_base": "evo2-40b-8k.yml",
    "evo2_1b_base": "evo2-1b-8k.yml",
    "evo2_7b_262k": "evo2-7b-262k.yml",
    "evo2_7b_microviridae": "evo2-7b-8k.yml",
}
HF_MODEL_NAME_MAP = {
    "evo2_40b": "arcinstitute/evo2_40b",
    "evo2_7b": "arcinstitute/evo2_7b",
    "evo2_20b": "arcinstitute/evo2_20b",
    "evo2_40b_base": "arcinstitute/evo2_40b_base",
    "evo2_7b_base": "arcinstitute/evo2_7b_base",
    "evo2_1b_base": "arcinstitute/evo2_1b_base",
    "evo2_7b_262k": "arcinstitute/evo2_7b_262k",
    "evo2_7b_microviridae": "evo-design/evo-2-7b-8k-microviridae",
}
MODEL_NAMES = list(HF_MODEL_NAME_MAP)
VENDORED_CONFIGS = Path(__file__).resolve().parent / "evo2_configs"

FP8_FORMAT = "e4m3fn"
FP8_MAX_FWD = 448.0
# vLLM discovers every root-level ``*.safetensors`` file as a model shard.
# Keep audit-only tensors in a subdirectory so they can never enter load_weights.
FP8_META_FILENAME = "metadata/hyena_in_proj_fp8_meta.safetensors"
FP8_RUNTIME = "vllm_fp8_static_w8a8_v1"


def _load_te_extra_state(value, key: str) -> dict:
    """Deserialize and validate one Transformer Engine ``_extra_state`` blob."""
    if isinstance(value, io.BytesIO):
        value.seek(0)
        value = torch.load(value, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(
            f"{key}: expected Transformer Engine BytesIO/dict metadata, "
            f"got {type(value).__name__}"
        )

    required = (
        "scale_fwd",
        "scale_inv_fwd",
        "amax_history_fwd",
        "extra_fp8_variables",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"{key}: missing FP8 metadata fields {missing}")

    scale = value["scale_fwd"]
    scale_inv = value["scale_inv_fwd"]
    amax = value["amax_history_fwd"]
    for name, tensor, shape in (
        ("scale_fwd", scale, (3,)),
        ("scale_inv_fwd", scale_inv, (3,)),
        ("amax_history_fwd", amax, (16, 3)),
    ):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            got = (
                tuple(tensor.shape)
                if isinstance(tensor, torch.Tensor)
                else type(tensor).__name__
            )
            raise ValueError(f"{key}: {name} must have shape {shape}, got {got}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key}: {name} contains non-finite values")

    if not (scale > 0).all() or not (scale_inv > 0).all():
        raise ValueError(f"{key}: forward FP8 scales and inverses must be positive")
    if not torch.allclose(
        scale.float() * scale_inv.float(),
        torch.ones(3),
        rtol=2e-3,
        atol=2e-3,
    ):
        raise ValueError(f"{key}: scale_inv_fwd is not reciprocal to scale_fwd")

    extra = value["extra_fp8_variables"]
    expected = {"fp8_checkpoint": True, "num_gemms": 1, "fp8_max_fwd": 448}
    if not isinstance(extra, dict):
        raise ValueError(f"{key}: extra_fp8_variables must be a dict")
    bad = {
        name: extra.get(name)
        for name, want in expected.items()
        if extra.get(name) != want
    }
    if bad:
        raise ValueError(
            f"{key}: unsupported Transformer Engine FP8 descriptor {bad}; "
            f"expected {expected}"
        )
    return value


def _quantize_te_weight(
    weight: torch.Tensor, scale_fwd: torch.Tensor
) -> torch.Tensor:
    """Apply TE's forward E4M3 saturation using its multiplicative weight scale."""
    scaled = (
        weight.detach()
        .float()
        .mul_(scale_fwd.float())
        .clamp_(-FP8_MAX_FWD, FP8_MAX_FWD)
    )
    return scaled.to(torch.float8_e4m3fn).contiguous()


def export_hyena_fp8(
    sd: dict,
    new: dict[str, torch.Tensor],
    hyena_layers: list[int],
) -> dict[str, torch.Tensor]:
    """Export native vLLM static-W8A8 tensors plus an audit metadata sidecar."""
    audit: dict[str, torch.Tensor] = {}
    for i in hyena_layers:
        src = f"blocks.{i}.projections"
        dst = f"model.layers.{i}.mixer.in_proj"
        key = src + "._extra_state"
        if key not in sd:
            raise ValueError(
                "source config requires FP8 Hyena input projections, "
                f"but {key} is missing"
            )
        meta = _load_te_extra_state(sd[key], key)
        scale = meta["scale_fwd"].float().cpu()
        scale_inv = meta["scale_inv_fwd"].float().cpu()

        # TE scale_fwd is multiplicative. vLLM's serialized FP8 convention
        # stores the reciprocal/dequantization scale.
        new[dst + ".weight"] = _quantize_te_weight(
            sd[src + ".weight"], scale[1]
        )
        new[dst + ".weight_scale"] = scale_inv[1].reshape(1).contiguous()
        new[dst + ".input_scale"] = scale_inv[0].reshape(1).contiguous()

        prefix = f"layers.{i}"
        audit[prefix + ".scale_fwd"] = scale.contiguous()
        audit[prefix + ".scale_inv_fwd"] = scale_inv.contiguous()
        audit[prefix + ".amax_history_fwd"] = (
            meta["amax_history_fwd"].float().cpu().contiguous()
        )
    return audit


def permute_qkv(w: torch.Tensor, hidden: int, num_heads: int) -> torch.Tensor:
    """Checkpoint per-head-interleaved [h, nH, 3, hd] -> standard [Wq;Wk;Wv] rows.

    Mirrors vortex StripedHyena.custom_load_state_dict under `column_split: True`.
    """
    head_dim = hidden // num_heads
    w = w.permute(1, 0)  # (3H, H) -> (H, 3H)
    w = w.reshape(hidden, num_heads, 3, head_dim)
    wq, wk, wv = w.unbind(dim=-2)
    wq = wq.reshape(hidden, -1)
    wk = wk.reshape(hidden, -1)
    wv = wv.reshape(hidden, -1)
    return torch.cat([wq, wk, wv], dim=-1).permute(1, 0).contiguous()




# ---------------------------------------------------------------------------
# Locating sources
# ---------------------------------------------------------------------------

def hf_hub_cache() -> str:
    """The HuggingFace hub cache, honouring HF_HUB_CACHE / HF_HOME."""
    try:
        from huggingface_hub import constants

        return constants.HF_HUB_CACHE
    except ImportError:
        home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        return os.environ.get("HF_HUB_CACHE", os.path.join(home, "hub"))


def find_config(model: str) -> str:
    """Path of the vortex config yml for an evo2 model name."""
    if model in CONFIG_MAP:
        return str(VENDORED_CONFIGS / CONFIG_MAP[model])
    # Variants outside evo2's CONFIG_MAP ship a config.yml in their HF snapshot.
    pat = os.path.join(hf_hub_cache(), f"models--arcinstitute--{model}",
                       "snapshots", "*", "config.yml")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(
            f"{model} is not in CONFIG_MAP and no config.yml found at {pat}; "
            f"pass the config explicitly"
        )
    return hits[-1]


def _shards(stem: str) -> list[str]:
    parts, i = [], 0
    while os.path.exists(f"{stem}.part{i}"):
        parts.append(f"{stem}.part{i}")
        i += 1
    return parts


def find_checkpoint(model: str, download: bool = False) -> str | list[str]:
    """The vortex `.pt` for an evo2 model name: a path, or its ordered shards.

    Looks where `evo2.Evo2` looks: the merged file evo2 leaves next to the hub
    cache, then the snapshot (whole file or `.partN` shards). With
    `download=True` a missing snapshot is fetched like evo2 does.
    """
    fname = f"{model}.pt"
    merged = os.path.join(os.path.dirname(hf_hub_cache()), fname)
    if os.path.exists(merged):
        return merged
    repo = HF_MODEL_NAME_MAP.get(model, f"arcinstitute/{model}")
    snap_glob = os.path.join(hf_hub_cache(), "models--" + repo.replace("/", "--"),
                             "snapshots", "*")
    for snap in sorted(glob.glob(snap_glob), reverse=True):
        whole = os.path.join(snap, fname)
        if os.path.exists(whole):
            return whole
        parts = _shards(whole)
        if parts:
            return parts
    if download:
        from huggingface_hub import snapshot_download

        snap = snapshot_download(repo_id=repo, allow_patterns=[f"{fname}*", "*.yml", "*.json"])
        whole = os.path.join(snap, fname)
        if os.path.exists(whole):
            return whole
        parts = _shards(whole)
        if parts:
            return parts
    raise FileNotFoundError(
        f"no checkpoint for {model}: looked for {merged} and {fname}[.partN] under "
        f"{snap_glob}" + ("" if download else " (download=False)")
    )


class _ConcatReader(io.RawIOBase):
    """Read-only, seekable view of several files laid end to end.

    evo2 ships 40B as `.pt.part0` / `.pt.part1` and merges them into one ~80 GB
    file before loading. torch.load only needs a seekable file object, so the
    shards are read in place instead.
    """

    def __init__(self, paths):
        super().__init__()
        self._files = [open(p, "rb") for p in paths]
        self._sizes = [os.path.getsize(p) for p in paths]
        self._starts, acc = [], 0
        for s in self._sizes:
            self._starts.append(acc)
            acc += s
        self._total, self._pos = acc, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._total}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buf):
        view, done = memoryview(buf).cast("B"), 0
        while done < len(view) and self._pos < self._total:
            i = max(k for k, s in enumerate(self._starts) if s <= self._pos)
            f = self._files[i]
            f.seek(self._pos - self._starts[i])
            want = min(len(view) - done, self._starts[i] + self._sizes[i] - self._pos)
            n = f.readinto(view[done:done + want])
            if not n:
                break
            done += n
            self._pos += n
        return done

    def close(self):
        for f in self._files:
            f.close()
        super().close()


def _load_state_dict(ckpt):
    if isinstance(ckpt, (list, tuple)):
        with io.BufferedReader(_ConcatReader(ckpt), buffer_size=64 << 20) as fh:
            sd = torch.load(fh, map_location="cpu", weights_only=False)
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    return sd.get("module", sd)


def _source_name(ckpt) -> str:
    if isinstance(ckpt, (list, tuple)):
        name = os.path.basename(ckpt[0])
        return name[: name.rindex(".part")]
    return os.path.basename(ckpt)


def _staging_dir(out: str) -> str:
    """Where to write. A new `out` is staged and renamed into place when done,
    so a crashed conversion never leaves a loadable half-written directory; an
    existing `out` (possibly a symlink) is written in place, as before."""
    if os.path.exists(out):
        return out
    stage = f"{out.rstrip(os.sep)}.partial-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    return stage


def _publish(stage: str, out: str) -> None:
    if stage == out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.replace(stage, out)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_checkpoint(ckpt_path, cfg_path: str, out: str, log=print) -> dict:
    """Convert a vortex Evo 2 checkpoint into an HF-style directory at `out`.

    ckpt_path: a `.pt` file, or the ordered list of its `.pt.partN` shards.
    cfg_path:  the vortex config yml (see `find_config`).
    Writes into a staging directory and renames it into place on success.
    Returns the emitted config.json contents.
    """

    log(f"checkpoint : {ckpt_path}")
    log(f"vortex cfg : {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))

    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_layers = cfg["num_layers"]
    attn_idx = set(cfg["attn_layer_idxs"])
    hcs_idx = set(cfg["hcs_layer_idxs"])
    hcm_idx = set(cfg["hcm_layer_idxs"])
    hcl_idx = set(cfg["hcl_layer_idxs"])
    hyena_idx = sorted(hcs_idx | hcm_idx | hcl_idx)
    requires_fp8 = bool(cfg.get("use_fp8_input_projections", False))

    if not cfg.get("column_split", True):
        raise NotImplementedError(
            "config has column_split=False; the qkv permutation in permute_qkv() "
            "assumes the column-split layout vortex applies at load time."
        )

    sd = _load_state_dict(ckpt_path)
    sd = sd.get("module", sd)

    new: dict[str, torch.Tensor] = {}
    fp8_audit: dict[str, torch.Tensor] = {}

    emb = sd["embedding_layer.weight"]
    new["model.embed_tokens.weight"] = emb.contiguous()
    if "unembed.weight" in sd:
        tied = torch.equal(sd["unembed.weight"], emb)
        log(f"tie_embeddings: config={cfg.get('tie_embeddings')} weights_equal={tied}")
        if not tied and cfg.get("tie_embeddings"):
            raise ValueError(
                "config says tie_embeddings but unembed.weight != embedding_layer.weight"
            )
    new["model.norm.weight"] = sd["norm.scale"].contiguous()

    for i in range(n_layers):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        new[q + "pre_norm.weight"] = sd[p + "pre_norm.scale"].contiguous()
        new[q + "post_norm.weight"] = sd[p + "post_norm.scale"].contiguous()
        for l in ("l1", "l2", "l3"):
            new[f"{q}mlp.{l}.weight"] = sd[f"{p}mlp.{l}.weight"].contiguous()

        if i in attn_idx:
            w = sd[p + "inner_mha_cls.Wqkv.weight"]
            new[q + "mixer.qkv_proj.weight"] = permute_qkv(w, H, n_heads)
            if p + "inner_mha_cls.Wqkv.bias" in sd:
                raise NotImplementedError(
                    "qkv bias present; permute_qkv handles weights only"
                )
            new[q + "mixer.o_proj.weight"] = sd[
                p + "inner_mha_cls.out_proj.weight"
            ].contiguous()
            if p + "inner_mha_cls.out_proj.bias" in sd:
                new[q + "mixer.o_proj.bias"] = sd[
                    p + "inner_mha_cls.out_proj.bias"
                ].contiguous()
            continue

        # --- Hyena mixers (SE / MR / LI) share the outer path ---
        if not requires_fp8:
            new[q + "mixer.in_proj.weight"] = sd[
                p + "projections.weight"
            ].contiguous()
        new[q + "mixer.short_filter.weight"] = sd[
            p + "filter.short_filter_weight"
        ].contiguous()
        new[q + "mixer.out_proj.weight"] = sd[p + "out_filter_dense.weight"].contiguous()
        new[q + "mixer.out_proj.bias"] = sd[p + "out_filter_dense.bias"].contiguous()

        if i in hcl_idx:
            new[q + "mixer.log_poles"] = sd[p + "filter.log_poles"].float().contiguous()
            new[q + "mixer.residues"] = sd[p + "filter.residues"].float().contiguous()
            new[q + "mixer.D"] = sd[p + "filter.D"].contiguous()
        elif i in hcs_idx or i in hcm_idx:
            h = sd[p + "filter.h"]  # (G, 1, K)
            groups = h.shape[0]
            if H % groups:
                raise ValueError(f"layer {i}: hidden {H} not divisible by groups {groups}")
            h = h.repeat_interleave(H // groups, dim=0)  # (H, 1, K)
            # Convention normalisation, see module docstring item 4:
            # SE (K<128) prefills with F.conv1d -> cross-correlation, and
            # step_fir consumes the kernel unflipped. MR (K>=128) prefills with
            # fftconv_func -> true convolution, and step_fir sets
            # flip_filter=True. causal_conv1d (what the vLLM port will use) is
            # cross-correlation, so pre-flip MR here to put both on one
            # convention. Recorded as conv_kernels_pre_flipped in config.json.
            if h.shape[-1] >= 128:
                h = h.flip(-1)
            new[q + "mixer.conv.weight"] = h.contiguous()
            if p + "filter.D" in sd:
                new[q + "mixer.D"] = sd[p + "filter.D"].contiguous()
        else:
            raise ValueError(f"layer {i} is in no known layer-type list")

    if requires_fp8:
        expected = {f"blocks.{i}.projections._extra_state" for i in hyena_idx}
        projection_extra = {
            key
            for key in sd
            if key.endswith(".projections._extra_state")
        }
        if projection_extra != expected:
            raise ValueError(
                "FP8 projection metadata layers do not match the source config: "
                f"missing={sorted(expected - projection_extra)}, "
                f"unexpected={sorted(projection_extra - expected)}"
            )
        fp8_audit = export_hyena_fp8(sd, new, hyena_idx)

    all_extra = {key for key in sd if key.endswith("._extra_state")}
    consumed_extra = {
        f"blocks.{i}.projections._extra_state" for i in hyena_idx
    } if requires_fp8 else set()
    dropped_extra = len(all_extra - consumed_extra)

    # Rotary base check.
    #
    # The checkpoint carries an `inv_freq` buffer, but for the long-context 7B
    # models it is STALE: evo2_7b stores inv_freq for base 1e4 while
    # evo2-7b-1m.yml declares base 1e8 + linear scaling 16. Verified empirically
    # against goldens (see PLAN.md §1.9): reproducing vortex's attention output
    # requires (base=1e8, scaling=16) — the config values — not the stored
    # buffer. A port that loaded inv_freq from the checkpoint, which is the
    # natural thing to do, would silently get base 1e4 and broken long-context
    # behaviour. So: always derive rotary from the config, never the checkpoint.
    # The CHECKPOINT's inv_freq is authoritative, not the config's
    # `rotary_emb_base` (see PLAN.md §1.9, which originally claimed the reverse).
    #
    # For long-context 7B the two disagree: evo2-7b-1m.yml declares base 1e11
    # while the stored inv_freq implies base 1e4. Verified against vortex at
    # L=8192: base 1e4 + scaling 128 reproduces its block-3 activations
    # (8.15e-2 vs 8.11e-2) and its next-token prediction; base 1e11 does not
    # (3.11e-1, and predicts a different base). The effective rope is therefore
    # the original 8k base plus linear position interpolation.
    #
    # This is invisible at short context: with base 1e11 the inv_freqs are so
    # small that rotations are near-identity over a few hundred positions, so an
    # L=512 parity test passes with either value. Discriminating requires a
    # long-context test — hence tests/test_long_context.py.
    cfg_base = cfg.get("rotary_emb_base", 10000)
    hd = H // n_heads
    rope_base = cfg_base
    for i in sorted(attn_idx):
        k = f"blocks.{i}.inner_mha_cls.rotary_emb.inv_freq"
        if k in sd:
            got = sd[k].float()
            ks = torch.arange(0, hd, 2).float()
            # inv_freq[j] = base**(-k_j/hd)  =>  log(base) = -log(inv_freq)/(k_j/hd)
            implied = torch.exp(
                (-got[1:].log() / (ks[1:] / hd)).median()
            ).item()
            want = 1.0 / (cfg_base ** (ks / hd))
            rel = ((got - want).abs() / want.clamp_min(1e-12)).max().item()
            if rel < 5e-2:
                log(f"rope: checkpoint inv_freq agrees with config base {cfg_base:g}")
            else:
                # inv_freq is stored in bf16, so the implied base carries ~0.1%
                # noise (10010 rather than 10000). Snap to 3 significant figures.
                exp10 = int(f"{implied:e}".split("e")[1])
                rope_base = round(implied, -(exp10 - 2))
                log(
                    f"rope: config declares base {cfg_base:g} but checkpoint inv_freq "
                    f"implies {implied:.4g} -> using {rope_base:g} (checkpoint wins)"
                )
            break

    # Guard: the emitted config must agree with the exported tensors. The 7B
    # variants are not architecturally identical (evo2_7b/1M has
    # inner_mlp_size 11264, evo2_7b_262k has 11008), so a config field taken
    # from the wrong yml would load as a silent shape mismatch.
    l1 = new["model.layers.0.mlp.l1.weight"].shape
    if l1 != (cfg["inner_mlp_size"], H):
        raise ValueError(
            f"config inner_mlp_size={cfg['inner_mlp_size']} / hidden={H} disagrees "
            f"with checkpoint mlp.l1 shape {tuple(l1)} — wrong config for this checkpoint?"
        )

    hf_cfg = {
        "architectures": ["StripedHyena2ForCausalLM"],
        "model_type": "striped_hyena2",
        "vocab_size": cfg["vocab_size"],
        "hidden_size": H,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads // cfg.get("proj_groups", 1),
        "intermediate_size": cfg["inner_mlp_size"],
        "attn_layer_idxs": sorted(attn_idx),
        "hcs_layer_idxs": sorted(hcs_idx),
        "hcm_layer_idxs": sorted(hcm_idx),
        "hcl_layer_idxs": sorted(hcl_idx),
        "short_filter_length": cfg["short_filter_length"],
        "hcs_filter_length": cfg["hcs_filter_length"],
        "hcm_filter_length": cfg["hcm_filter_length"],
        "state_size": cfg["state_size"],
        "rms_norm_eps": float(cfg.get("eps", 1e-6)),
        "rope_theta": float(rope_base),
        "rope_scaling": (
            {"rope_type": "linear", "factor": float(cfg["rotary_emb_scaling_factor"])}
            if cfg.get("use_interpolated_rotary_pos_emb", False)
            else None
        ),
        # PRE-scaling length. vortex's `max_seqlen` is the POST-scaling context,
        # and HF/vLLM linear-scaling rope defines usable context as
        # max_position_embeddings * factor -- vLLM literally allocates a cos/sin
        # cache of that many positions, so passing the post-scaling value here
        # asks for 1048576*128 positions and OOMs on a 32 GiB allocation.
        # Every Evo 2 variant is consistent with an 8192 base:
        #   131k = 8192*16,  262k = 8192*32,  1M = 8192*128.
        # Un-interpolated (base) models: the TRAINED length, 8192, not max_seqlen.
        # evo2-7b-8k.yml declares max_seqlen 32768 (vortex's cache size); past
        # ~16k the 7B-base NLL is 15+ nats, so advertising 32k is a trap.
        "max_position_embeddings": (
            cfg["max_seqlen"] // int(cfg["rotary_emb_scaling_factor"])
            if cfg.get("use_interpolated_rotary_pos_emb", False)
            else min(cfg.get("max_seqlen", 8192), 8192)
        ),
        "tie_word_embeddings": bool(cfg.get("tie_embeddings", True)),
        "conv_kernels_pre_flipped": True,
        "interleave": bool(cfg.get("interleave", True)),
        "column_split_hyena": bool(cfg.get("column_split_hyena", False)),
        "hyena_flip_x1x2": bool(cfg.get("hyena_flip_x1x2", False)),
        "mlp_activation": cfg.get("mlp_activation", "gelu"),
        "use_fp8_input_projections": requires_fp8,
        "torch_dtype": "bfloat16",
        "_source_checkpoint": _source_name(ckpt_path),
        "_source_config": os.path.basename(cfg_path),
    }

    if requires_fp8:
        hf_cfg["hyena_in_proj_fp8"] = {
            "required": True,
            "runtime": FP8_RUNTIME,
            "format": FP8_FORMAT,
            "fp8_max": int(FP8_MAX_FWD),
            "scale_scheme": "per_tensor_static",
            "scale_convention": "dequantization",
            "input_scale_source": "scale_inv_fwd[0]",
            "weight_scale_source": "scale_inv_fwd[1]",
            "num_layers": len(hyena_idx),
            "metadata_file": FP8_META_FILENAME,
        }

    final_out, out = out, _staging_dir(out)
    save_file(new, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    if requires_fp8:
        os.makedirs(os.path.join(out, os.path.dirname(FP8_META_FILENAME)), exist_ok=True)
        save_file(
            fp8_audit,
            os.path.join(out, FP8_META_FILENAME),
            metadata={
                "format": "pt",
                "source": "transformer_engine_delayed_scaling",
            },
        )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(hf_cfg, f, indent=2)
    _publish(out, final_out)
    out = final_out

    n_par = sum(v.numel() for v in new.values())
    log(
        f"\nwrote {out}\n  tensors={len(new)}  params={n_par/1e9:.3f}B  "
        f"preserved fp8 _extra_state={len(consumed_extra)}  "
        f"dropped unrelated _extra_state={dropped_extra}"
    )
    if requires_fp8:
        log(
            f"  selective Hyena in_proj={FP8_FORMAT} static W8A8; "
            f"audit metadata={FP8_META_FILENAME}"
        )
    log(f"  rope_scaling={hf_cfg['rope_scaling']}")
    return hf_cfg


# ---------------------------------------------------------------------------
# GPUs without FP8 tensor cores (FP8_SURGERY.md)
# ---------------------------------------------------------------------------

# Configs whose block-0 Hyena output is trained to cancel the token embedding, so the
# model depends on Hopper's FP8 fast-accumulation rounding there. Without FP8 tensor
# cores (sm < 8.9) vLLM serves an FP8 in_proj as weight-only Marlin, which does not
# reproduce that rounding: the stock 1B scores 0.805 on an A100 and an RTX 3090
# (docs/GCP_GPU_MATRIX.md on branch ops/gcp-gpu-matrix). The value is the measured fix
# (FP8_SURGERY.md): "token_table" when the table + dequantised-E4M3 variant passes the
# strict gate, else what it would take. The 7B has no such cancellation and is not
# listed.
FP8_BLOCK0_SENSITIVE = {
    "evo2-1b-8k.yml": "token_table",    # 6.6e-4 (table + dequantised E4M3)
    "evo2-20b-1m.yml": "table + hybrid layer 23 (table alone: 1.29e-3, fails)",
    "evo2-40b-1m.yml": "table + dequantised E4M3 (6.5e-4), ~80 GB of bf16 weights",
}
TOKEN_TABLE_KEY = "model.layers.0.mixer.in_proj_token_table"


# Verbatim from the source repo's evo2_layers/model.py (the reference forward).
def rms_norm(x, scale, eps):
    """vortex RMSNorm: eps added to the norm, not under a sqrt.

    Computed in x's dtype, as vortex does — not promoted to fp32.
    """
    h = x.shape[-1]
    y = x / (x.norm(2, dim=-1, keepdim=True) * h ** (-0.5) + eps)
    return scale.to(x.dtype) * y


def build_token_table_checkpoint(src: str, out: str, device: str | None = None,
                                 log=print) -> str:
    """FP8 surgery, table variant: what `scripts/fp8_surgery.py --src src --out out`
    writes with no other flags, as a library call.

    Layer 0's in_proj becomes an exact (vocab, 3H) lookup of its Hopper
    fast-accumulation FP8 output, computed once with the bit-exact emulator
    (fp8_emul.py). Every other Hyena in_proj becomes a bf16 GEMM over
    the dequantised E4M3 weight. The result needs no FP8 GEMM. For the 1B this
    passes the strict gate: 6.6e-4 (FP8_SURGERY.md).
    """
    from safetensors.torch import load_file

    from .fp8_emul import static_fp8_linear_emulated

    cfg = json.load(open(os.path.join(src, "config.json")))
    if not cfg.get("use_fp8_input_projections"):
        raise ValueError(f"{src} is not an FP8-projection checkpoint")
    if 0 in cfg["attn_layer_idxs"]:
        raise ValueError("layer 0 is attention; the token-table surgery needs Hyena")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    w = load_file(os.path.join(src, "model.safetensors"))

    new = {k: v for k, v in w.items() if ".mixer.in_proj." not in k}
    layers = sorted(int(k.split(".")[2]) for k in w if k.endswith("mixer.in_proj.weight"))
    assert layers and layers[0] == 0, layers
    for i in layers:
        p = f"model.layers.{i}.mixer.in_proj."
        W8 = w[p + "weight"]
        assert W8.dtype == torch.float8_e4m3fn, (p, W8.dtype)
        sx = w[p + "input_scale"].float().reshape(1).to(dev)
        sw = w[p + "weight_scale"].float().reshape(1).to(dev)
        if i == 0:
            E = w["model.embed_tokens.weight"].to(device=dev, dtype=torch.bfloat16)
            x = rms_norm(E, w["model.layers.0.pre_norm.weight"].to(dev), cfg["rms_norm_eps"])
            table = static_fp8_linear_emulated(x, W8.to(dev), sx, sw)
            new[TOKEN_TABLE_KEY] = table.to(torch.bfloat16).cpu().contiguous()
        else:
            new[p + "weight"] = (W8.float() * sw.cpu()).to(torch.bfloat16).contiguous()

    out_cfg = dict(cfg)
    out_cfg["use_fp8_input_projections"] = False
    fp8_meta = out_cfg.pop("hyena_in_proj_fp8")
    out_cfg["hyena_token_table_layers"] = [0]
    out_cfg["hyena_fp8_emulated_layers"] = []
    out_cfg["hyena_fp8_emulated_channels"] = {}
    out_cfg["fp8_surgery"] = {
        "version": 1,
        "source": os.path.abspath(src),
        "source_fp8": fp8_meta,
        "layer0": ("token table = Hopper fast-accumulation E4M3 GEMM, emulated "
                   "(evo2_layers/fp8_emul.py)"),
        "emulated_layers": "[]",
        "hybrid_layers": "none",
        "other_hyena_in_proj": "dequantised E4M3 weights, bf16",
        "built_by": "evo2_flash.convert.build_token_table_checkpoint",
        "doc": "FP8_SURGERY.md",
    }
    stage = _staging_dir(out)
    save_file(new, os.path.join(stage, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(stage, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)
    meta = os.path.join(src, "metadata")
    if os.path.isdir(meta):  # keep the TE scale audit for provenance
        shutil.copytree(meta, os.path.join(stage, "metadata"), dirs_exist_ok=True)
    _publish(stage, out)
    log(f"[evo2_flash] wrote {out}: layer 0 token table {tuple(table.shape)}, "
        f"{len(layers) - 1} Hyena in_proj in bf16")
    return out
