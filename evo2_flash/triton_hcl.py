"""Fused modal-filter build for Hyena-LI.

    h[d, l] = sum_s residues[d, s] * exp(log_poles[d, s] * t[l])

Vendored from Zymrael/vortex (Apache-2.0), `vortex/ops/hcl_interface.py` +
`vortex/ops/triton_common.py` as of PR #76 / commit 8b00afe, by AlphaKhaw. See
https://alphakhaw.com/blog/triton-kernels-evo2-long-context-inference

Why this is worth vendoring rather than reimplementing: the state-size sum runs in
a register accumulator, so the `(D, S, L)` intermediate never exists. Computing
`h` the obvious way needs `D*S*L*4` bytes — 34 GB at H=4096, S=16, L=131072,
which is exactly the 32 GiB OOM this port hit on its first long-context run. The
workaround here previously was a Python loop over the 16 state channels, which
avoids the allocation but costs 16 sequential launches and 16 `(D, L)` temporaries
per LI layer per sequence.

Layout-independent: `(D, S)` parameters plus a `(L,)` time index in, `(D, L)` out.
No batch axis, no varlen concerns — which is why this piece ports cleanly into the
vLLM plugin while vortex's HCS/HCM conv kernels do not (they assume dense
`(B, D, L)` with a padding mask).

Falls back to a chunked torch implementation when triton is unavailable.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:
    _BDL_TILE_CONFIGS = [
        triton.Config({"BLOCK_D": 32, "BLOCK_L": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_L": 256}, num_warps=8),
        triton.Config({"BLOCK_D": 128, "BLOCK_L": 128}, num_warps=8),
    ]

    @triton.autotune(configs=_BDL_TILE_CONFIGS, key=["D", "L"])
    @triton.jit
    def _hcl_compute_filter_kernel(
        residues_ptr, log_poles_ptr, t_ptr, h_ptr,
        D, L,
        stride_rs_d, stride_rs_s, stride_h_d, stride_h_l,
        S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr,
    ):
        """One program covers a (BLOCK_D, BLOCK_L) tile of h; the S-term sum runs
        in an fp32 register accumulator so the (D, S, L) intermediate never exists."""
        pid_d = tl.program_id(0)
        pid_l = tl.program_id(1)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        mask_d = offs_d < D
        mask_l = offs_l < L

        t_tile = tl.load(t_ptr + offs_l, mask=mask_l, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_D, BLOCK_L), dtype=tl.float32)
        for s in tl.static_range(S):
            rs_offs = offs_d * stride_rs_d + s * stride_rs_s
            r_s = tl.load(residues_ptr + rs_offs, mask=mask_d, other=0.0).to(tl.float32)
            lp_s = tl.load(log_poles_ptr + rs_offs, mask=mask_d, other=0.0).to(tl.float32)
            acc += r_s[:, None] * tl.exp(lp_s[:, None] * t_tile[None, :])

        h_ptrs = h_ptr + offs_d[:, None] * stride_h_d + offs_l[None, :] * stride_h_l
        tl.store(h_ptrs, acc, mask=mask_d[:, None] & mask_l[None, :])


def _compute_filter_torch(residues, log_poles, t):
    """Chunked fallback: accumulate over state channels, never materialising (D,S,L)."""
    D, S = residues.shape
    h = torch.zeros(D, t.shape[0], dtype=torch.float32, device=residues.device)
    for s in range(S):
        h += residues[:, s : s + 1].float() * torch.exp(
            log_poles[:, s : s + 1].float() * t.float()
        )
    return h


def hcl_compute_filter(residues, log_poles, t, force_torch: bool = False):
    """h[d,l] = sum_s residues[d,s] * exp(log_poles[d,s] * t[l]) -> (D, L) fp32.

    residues, log_poles: (D, S). t: (L,).
    """
    if residues.dim() != 2 or log_poles.dim() != 2:
        raise ValueError(
            f"expected 2-D residues and log_poles, got {tuple(residues.shape)} "
            f"and {tuple(log_poles.shape)}"
        )
    if residues.shape != log_poles.shape:
        raise ValueError(
            f"residues {tuple(residues.shape)} and log_poles "
            f"{tuple(log_poles.shape)} must match"
        )
    if t.dim() != 1:
        raise ValueError(f"expected 1-D t, got {tuple(t.shape)}")

    if force_torch or not _HAVE_TRITON or not residues.is_cuda:
        return _compute_filter_torch(residues, log_poles, t)

    D, S = residues.shape
    L = t.shape[0]
    residues = residues.contiguous().float()
    log_poles = log_poles.contiguous().float()
    t = t.contiguous().float()
    h = torch.empty(D, L, dtype=torch.float32, device=residues.device)

    _hcl_compute_filter_kernel[
        lambda meta: (triton.cdiv(D, meta["BLOCK_D"]), triton.cdiv(L, meta["BLOCK_L"]))
    ](
        residues, log_poles, t, h, D, L,
        residues.stride(0), residues.stride(1), h.stride(0), h.stride(1),
        S,
    )
    return h
