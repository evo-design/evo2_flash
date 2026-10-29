"""Evo2RMSNorm, optionally with the residual add, in one bit-identical launch.

The torch path (`model.Evo2RMSNorm`) is five kernels per norm, plus one for
the residual add before `post_norm`:

    s = m + residual                                  # bf16
    n = s.norm(2, dim=-1, keepdim=True)               # fp32 sum of squares, sqrt, bf16
    y = weight * (s / (n * H**-0.5 + eps))            # bf16 after every op

`rms_norm` reproduces each rounding point, and the sum of squares in ATen's
order. That order is fixed by `setReduceConfig` (Reduce.cuh, torch 2.11) from
the row count T, the width H and the GPU (`triton_prefill.aten_sum_config`):
each lane accumulates 4-element vectors t, t + NT, t + 2 NT, ... into four
accumulators in order, combines them ((a0 + a1) + a2) + a3, and the lanes then
meet in a halving tree. With few rows a row gets more lanes (512 at T=1, 32
from T=16 at H=4096), and at H=8192 several warps share a row, which adds a
second tree over warps after the one over lanes. Squares of bf16 values are
exact in fp32, so FMA contraction cannot change them. `tests/test_triton_norm.py`
checks the identity for T = 1..16384 at H 1920/4096/8192.

Rows must be contiguous and 16-byte aligned (ATen then reduces no unaligned
head), which every hidden state in the model is. Anything else, CPU tensors and
torch other than 2.11 return None, and the caller runs the torch path.
"""

import functools

import torch
import triton
import triton.language as tl

from .triton_prefill import QUALIFIED_TORCH, aten_sum_config


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _halve(v, A: tl.constexpr, N: tl.constexpr):
    """(A, N) -> (A, N/2): v[:, j] + v[:, j + N/2], one shuffle/shared-memory step."""
    return tl.sum(tl.reshape(v, (A, 2, N // 2)), 1)


@triton.jit
def _tree(v, A: tl.constexpr, N: tl.constexpr):
    """ATen's halving tree along axis 1 of (A, N), N a power of two <= 512."""
    if N >= 512:
        v = _halve(v, A, 512)
    if N >= 256:
        v = _halve(v, A, 256)
    if N >= 128:
        v = _halve(v, A, 128)
    if N >= 64:
        v = _halve(v, A, 64)
    if N >= 32:
        v = _halve(v, A, 32)
    if N >= 16:
        v = _halve(v, A, 16)
    if N >= 8:
        v = _halve(v, A, 8)
    if N >= 4:
        v = _halve(v, A, 4)
    if N >= 2:
        v = _halve(v, A, 2)
    return v


@triton.jit
def _norm_kernel(X, SX, R, SR, S, SS, Y, SY, W, H, inv_sqrt_h, eps,
                 HAS_ADD: tl.constexpr, NK: tl.constexpr, BW: tl.constexpr,
                 BY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BW * BY)[:, None]
    i = tl.arange(0, 4)[None, :]
    acc = tl.zeros((BW * BY, 4), tl.float32)
    for k in tl.static_range(NK):
        e = (k * BW * BY + t) * 4 + i
        me = e < H
        x = tl.load(X + row * SX + e, me, other=0).to(tl.float32)
        if HAS_ADD:
            x = _bf16(x + tl.load(R + row * SR + e, me, other=0).to(tl.float32))
        acc += x * x
    c = tl.sum(tl.where(i == 0, acc, 0), 1)
    c += tl.sum(tl.where(i == 1, acc, 0), 1)
    c += tl.sum(tl.where(i == 2, acc, 0), 1)
    c += tl.sum(tl.where(i == 3, acc, 0), 1)
    # Lane t = x + BW*y: the tree over x (block_x_reduce), then over y (block_y_reduce).
    c = _tree(tl.reshape(c, (BY, BW)), BY, BW)
    c = _tree(tl.reshape(c, (1, BY)), 1, BY)
    n = _bf16(tl.sqrt_rn(tl.reshape(c, (1,))))
    n = _bf16(n * inv_sqrt_h)
    n = _bf16(n + eps)

    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(X + row * SX + offs, mask, other=0).to(tl.float32)
    if HAS_ADD:
        x = _bf16(x + tl.load(R + row * SR + offs, mask, other=0).to(tl.float32))
        tl.store(S + row * SS + offs, x.to(S.dtype.element_ty), mask)
    y = _bf16(tl.div_rn(x, n))
    w = tl.load(W + offs, mask, other=0).to(tl.float32)
    tl.store(Y + row * SY + offs, (w * y).to(Y.dtype.element_ty), mask)


@functools.lru_cache(maxsize=None)
def _plan(T, H, device_index):
    """(NK, BW, BY) for ATen's norm over rows of H, or None if not restated."""
    if H % 4 or H < 128:
        return None
    c = aten_sum_config(T, H, True, 1, device_index)
    if not c["vec_in"] or c["ctas"] > 1:
        return None
    by = c["bh"] if c["imult"][1] else 1
    return triton.cdiv(H // 4, c["bw"] * by), c["bw"], by


def _rows_ok(t, H):
    return (t.dim() >= 1 and t.shape[-1] == H and t.stride(-1) == 1
            and t.dtype == torch.bfloat16 and t.data_ptr() % 16 == 0
            and (t.dim() == 1 or t.view(-1, H).stride(0) * 2 % 16 == 0))


def rms_norm(x, weight, eps, residual=None):
    """weight * s / (||s||_2 * H^-0.5 + eps) with s = x (+ residual), bit-exact.

    Returns y, or (y, s) when `residual` is given; None when the shape, layout
    or torch version is outside what the kernel restates.
    """
    H = x.shape[-1]
    if not (QUALIFIED_TORCH and x.is_cuda and weight.dtype == torch.bfloat16
            and _rows_ok(x, H) and (residual is None or (
                residual.shape == x.shape and _rows_ok(residual, H)))):
        return None
    try:
        x2 = x.view(-1, H)
        r2 = residual.view(-1, H) if residual is not None else x2
    except RuntimeError:
        return None
    T = x2.shape[0]
    plan = _plan(T, H, x.device.index or 0) if T else (1, 1, 1)
    if plan is None:
        return None
    nk, bw, by = plan
    y = torch.empty_like(x2)
    s = torch.empty_like(x2) if residual is not None else y
    if T:
        _norm_kernel[(T,)](
            x2, x2.stride(0), r2, r2.stride(0), s, s.stride(0), y, y.stride(0), weight, H,
            H ** -0.5, eps, HAS_ADD=residual is not None, NK=nk, BW=bw, BY=by,
            BLOCK=triton.next_power_of_2(H), num_warps=4, enable_fp_fusion=False,
        )
    if residual is None:
        return y.view(x.shape)
    return y.view(x.shape), s.view(x.shape)
