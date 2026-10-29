"""One checkpoint per model that runs on any GPU: FP8 weights plus a `surgery/` sidecar.

The 1B and 20B depend on Hopper's FP8 accumulation rounding in block 0. On a GPU
without FP8 tensor cores (sm < 8.9, e.g. the A100) they need their FP8-surgery
form instead: block 0's projection as an exact per-token table, the other Hyena
`in_proj` as bf16, and, for the 20B, 240 calibrated output channels of layer 23
recomputed by the emulated FP8 GEMM. A published checkpoint carries both:

    model.safetensors              the FP8 weights (used as is on sm_89+)
    config.json                    use_fp8_input_projections: true
    surgery/surgery.json           config overrides, and the sha256 of the weights
                                   and of the tensors file below
    surgery/tensors.safetensors    the token table and any hybrid-channel tensors

Below sm_89, `materialize_cached` derives the surgery checkpoint from the two,
once, into the cache. It needs no GPU and no calibration: everything that
depended on FP8 hardware or calibration data is precomputed in the sidecar, and
the rest is a dequantisation of weights the checkpoint already holds.
"""

import hashlib
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SIDECAR_DIR = "surgery"
SIDECAR_META = "surgery.json"
SCHEMA = "evo2_vllm_surgery_sidecar_v1"
# The only config keys a sidecar may override: it cannot change the architecture.
OVERRIDABLE = frozenset({
    "use_fp8_input_projections",
    "hyena_token_table_layers",
    "hyena_fp8_emulated_layers",
    "hyena_fp8_emulated_channels",
})


def sidecar_path(path: str) -> str:
    return os.path.join(path, SIDECAR_DIR, SIDECAR_META)


def has_sidecar(path: str) -> bool:
    """True if `path` holds FP8 weights and a surgery sidecar for them."""
    if not os.path.exists(sidecar_path(path)):
        return False
    try:
        with open(os.path.join(path, "config.json")) as f:
            return bool(json.load(f).get("use_fp8_input_projections"))
    except (OSError, ValueError):
        return False


def read_sidecar(path: str) -> dict:
    with open(sidecar_path(path)) as f:
        meta = json.load(f)
    where = sidecar_path(path)
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"{where}: unsupported schema {meta.get('schema')!r}")
    ov = meta.get("config_overrides", {})
    bad = sorted(set(ov) - OVERRIDABLE)
    if bad:
        raise ValueError(f"{where}: config_overrides may not set {bad}")
    if ov.get("use_fp8_input_projections") is not False:
        raise ValueError(f"{where}: must set use_fp8_input_projections=false")
    rel = meta.get("tensors")
    if not rel or os.path.isabs(rel) or ".." in rel.split("/"):
        raise ValueError(f"{where}: 'tensors' must be a relative path inside the checkpoint")
    if not os.path.exists(os.path.join(path, rel)):
        raise FileNotFoundError(f"{where}: {rel} is missing")
    if not meta.get("weights_sha256"):
        raise ValueError(f"{where}: no weights_sha256")
    return meta


def sha256_file(path: str, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify(path: str, meta: dict, log=print) -> None:
    """Refuse a sidecar next to weights, or tensors, other than the ones it was built for.

    A table or channel list applied to the wrong weights builds a wrong model with
    no error, so both files are checked before every build (about a minute for
    the 20B; it runs once, since the result is cached).
    """
    weights = os.path.join(path, "model.safetensors")
    if meta.get("weights_size") is not None and os.path.getsize(weights) != meta["weights_size"]:
        raise ValueError(f"{sidecar_path(path)}: model.safetensors has the wrong size")
    log("[evo2_flash] checking the surgery sidecar against its weights (sha256)")
    if sha256_file(weights) != meta["weights_sha256"]:
        raise ValueError(f"{sidecar_path(path)}: model.safetensors does not match the "
                         "weights this sidecar was built for")
    want = meta.get("tensors_sha256")
    if want is not None and sha256_file(os.path.join(path, meta["tensors"])) != want:
        raise ValueError(f"{sidecar_path(path)}: {meta['tensors']} is corrupted")


def _derive(src: str, meta: dict):
    """Yield (name, tensor) of the surgery checkpoint, tensor by tensor.

    - a token-table layer's `in_proj` is dropped (the table replaces it);
    - a fully emulated layer keeps its E4M3 weight and both scales;
    - every other Hyena `in_proj` becomes `weight * weight_scale` in bf16, and its
      scales are dropped.
    """
    if os.path.exists(os.path.join(src, "model.safetensors.index.json")):
        raise ValueError(f"{src} is sharded; a surgery sidecar needs one model.safetensors")
    ov = meta["config_overrides"]
    table = set(ov.get("hyena_token_table_layers") or ())
    emulated = set(ov.get("hyena_fp8_emulated_layers") or ())
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as h:
        keys = list(h.keys())
        scales = {k: h.get_tensor(k) for k in keys
                  if k.endswith(("in_proj.weight_scale", "in_proj.input_scale"))}
        for k in keys:
            if k.endswith("mixer.in_proj.weight"):
                layer, prefix = int(k.split(".")[2]), k[: -len("weight")]
                if layer in table:
                    continue
                if layer in emulated:
                    yield k, h.get_tensor(k)
                    yield prefix + "weight_scale", scales[prefix + "weight_scale"]
                    yield prefix + "input_scale", scales[prefix + "input_scale"]
                    continue
                yield k, (h.get_tensor(k).float() * scales[prefix + "weight_scale"].float()
                          ).to(torch.bfloat16).contiguous()
            elif not k.endswith(("in_proj.weight_scale", "in_proj.input_scale")):
                yield k, h.get_tensor(k)
    with safe_open(os.path.join(src, meta["tensors"]), "pt") as h:
        for k in h.keys():
            yield k, h.get_tensor(k)


def surgery_config(src: str, meta: dict) -> dict:
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg.update(meta["config_overrides"])
    cfg["fp8_surgery"] = meta["fp8_surgery"]
    cfg.pop("hyena_in_proj_fp8", None)
    return cfg


def materialize(src: str, out: str, log=print) -> str:
    """Write the surgery form of `src` (FP8 weights + sidecar) to `out`.

    CPU only. The output is written in one pass, so peak host memory is about its
    size (36 GB for the 20B, 2 GB for the 1B).
    """
    from .config import StripedHyena2Config
    from .convert import _publish, _staging_dir

    meta = read_sidecar(src)
    cfg = surgery_config(src, meta)
    # Refuse a config the model would reject, before anything is written.
    StripedHyena2Config(**{k: v for k, v in cfg.items()
                           if k not in ("architectures", "model_type", "torch_dtype")})
    verify(src, meta, log=log)
    stage = _staging_dir(out)
    save_file(dict(_derive(src, meta)), os.path.join(stage, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(stage, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if os.path.isdir(os.path.join(src, "metadata")):
        shutil.copytree(os.path.join(src, "metadata"), os.path.join(stage, "metadata"),
                        dirs_exist_ok=True)
    _publish(stage, out)
    return out


def cache_dir_for(src: str, cache_root: str) -> str:
    """The cache directory for `src`'s surgery form, keyed to everything it depends on.

    Bump SCHEMA if the derivation rules in `_derive` ever change, so no stale copy
    is reused.
    """
    meta = read_sidecar(src)
    with open(sidecar_path(src), "rb") as f:
        side = hashlib.sha256(f.read()).hexdigest()
    key = hashlib.sha256(f"{meta['weights_sha256']}:{side}:{SCHEMA}"
                         .encode()).hexdigest()[:12]
    try:
        with open(os.path.join(src, "config.json")) as f:
            stem = os.path.splitext(json.load(f).get("_source_checkpoint") or "")[0]
    except (OSError, ValueError):
        stem = ""
    return os.path.join(cache_root, f"{stem or os.path.basename(os.path.normpath(src))}-surgery-{key}")


def _is_materialised(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return False
    return (cfg.get("model_type") == "striped_hyena2"
            and not cfg.get("use_fp8_input_projections")
            and isinstance(cfg.get("fp8_surgery"), dict)
            and os.path.exists(os.path.join(path, "model.safetensors")))


def materialize_cached(src: str, cache_root: str, log=print) -> str:
    """`src`'s surgery form in `cache_root`, built once.

    Safe with concurrent callers (e.g. one process per GPU): one builds under a
    file lock, the others wait and reuse its result.
    """
    from filelock import FileLock

    out = cache_dir_for(src, cache_root)
    if _is_materialised(out):
        return out
    os.makedirs(cache_root, exist_ok=True)
    with FileLock(out + ".lock"):
        if not _is_materialised(out):
            log(f"[evo2_flash] building the FP8-free form of {src} -> {out} (first use only)")
            materialize(src, out, log=log)
    return out
