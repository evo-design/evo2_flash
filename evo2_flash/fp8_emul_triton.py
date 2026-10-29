"""Triton kernel for the Hopper fast-accumulation FP8 GEMM model (fp8_emul.py).

Same arithmetic as ``fp8_emul.fast_accum_mm`` -- groups of 32 k in ascending
order, alignment to the max operand exponent (and the accumulator's), 13
fraction bits, truncation toward zero, accumulator renormalised to 13 bits --
on CUDA cores, so it runs on any GPU Triton supports (A100, L40S, ...).

Operands are the *unscaled* E4M3 values held in a 16-bit float tensor (bf16 or
fp16 represent every E4M3 value exactly), K-major so each k step loads two
contiguous vectors:

    a_t : (K, N)   activations, transposed
    b_t : (K, M)   weight, transposed (precompute once)

returns the fp32 accumulator (N, M) before the scale epilogue.
"""

import torch
import triton
import triton.language as tl

GROUP = 32
FRAC_BITS = 13
SMALL_N, SMALL_BM, SMALL_WARPS = 16, 64, 4  # decode-size path (_fast_accum_kernel_3d)


@triton.jit
def _exp_floor(x):
    """floor(log2|x|) as int32 for normal fp32 x; very negative for x == 0."""
    bits = x.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127
    return tl.where(x == 0.0, -100000, e)


@triton.jit
def _pow2(e):
    """2^e as fp32 for -126 <= e <= 127 (clamped)."""
    e = tl.minimum(tl.maximum(e, -126), 127)
    return ((e + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fast_accum_kernel(a_ptr, b_ptr, c_ptr, N, M, K,
                       BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                       G: tl.constexpr, F: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = rn < N
    mask_m = rm < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, G):
        # pass 1: alignment exponent = max_k (e(a_k) + e(b_k)), and e(acc)
        emax = _exp_floor(acc)
        for kk in tl.static_range(G):
            a = tl.load(a_ptr + (k0 + kk) * N + rn, mask=mask_n, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + (k0 + kk) * M + rm, mask=mask_m, other=0.0).to(tl.float32)
            e = _exp_floor(a)[:, None] + _exp_floor(b)[None, :]
            emax = tl.maximum(emax, e)
        ulp = _pow2(emax - F)
        inv = _pow2(F - emax)
        # pass 2: truncate every product (and the accumulator) to a multiple of ulp
        s = (acc * inv).to(tl.int32).to(tl.float32) * ulp
        for kk in tl.static_range(G):
            a = tl.load(a_ptr + (k0 + kk) * N + rn, mask=mask_n, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + (k0 + kk) * M + rm, mask=mask_m, other=0.0).to(tl.float32)
            p = a[:, None] * b[None, :]
            s += (p * inv).to(tl.int32).to(tl.float32) * ulp
        # renormalise: keep F fraction bits of the accumulator's own exponent
        ea = _exp_floor(s)
        u2 = _pow2(ea - F)
        i2 = _pow2(F - ea)
        acc = tl.where(s == 0.0, s, (s * i2).to(tl.int32).to(tl.float32) * u2)
    c = c_ptr + rn[:, None] * M + rm[None, :]
    tl.store(c, acc, mask=mask_n[:, None] & mask_m[None, :])


@triton.jit
def _add_rz(a, b):
    return tl.inline_asm_elementwise("add.rz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _trunc_to(x, big):
    """x truncated toward zero to a multiple of ulp, where big = 2^(log2 ulp + 23):
    (x +rz s) -rz s with s = copysign(big, x). Exact for |x| < big / 2^8."""
    s = tl.where(x < 0.0, -big, big)
    return _add_rz(_add_rz(x, s), -s)


@triton.jit
def _fast_accum_kernel_rz(a_ptr, b_ptr, c_ptr, N, M, K,
                          BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                          G: tl.constexpr, F: tl.constexpr):
    """Same arithmetic as _fast_accum_kernel; truncation by round-toward-zero adds
    instead of float<->int conversions (which are quarter-rate)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = rn < N
    mask_m = rm < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, G):
        emax = _exp_floor(acc)
        for kk in tl.static_range(G):
            a = tl.load(a_ptr + (k0 + kk) * N + rn, mask=mask_n, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + (k0 + kk) * M + rm, mask=mask_m, other=0.0).to(tl.float32)
            emax = tl.maximum(emax, _exp_floor(a)[:, None] + _exp_floor(b)[None, :])
        big = _pow2(emax - F + 23)
        s = _trunc_to(acc, big)
        for kk in tl.static_range(G):
            a = tl.load(a_ptr + (k0 + kk) * N + rn, mask=mask_n, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + (k0 + kk) * M + rm, mask=mask_m, other=0.0).to(tl.float32)
            s += _trunc_to(a[:, None] * b[None, :], big)
        acc = tl.where(s == 0.0, s, _trunc_to(s, _pow2(_exp_floor(s) - F + 23)))
    c = c_ptr + rn[:, None] * M + rm[None, :]
    tl.store(c, acc, mask=mask_n[:, None] & mask_m[None, :])


@triton.jit
def _fast_accum_kernel_3d(a_ptr, b_ptr, c_ptr, N, M, K,
                          BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                          G: tl.constexpr, F: tl.constexpr):
    """Exact model for few rows (decode): a group's 32 products are formed as one
    (BLOCK_N, G, BLOCK_M) register tile, so the serial chain is one step per group
    instead of 2 x 32 per-k steps. Same arithmetic as _fast_accum_kernel."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, G)
    mask_n = rn < N
    mask_m = rm < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, G):
        # a_t is (K, N): element (k, n) at k * N + n
        a = tl.load(a_ptr + (k0 + rk)[None, :] * N + rn[:, None], mask=mask_n[:, None],
                    other=0.0).to(tl.float32)                      # (BN, G)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * M + rm[None, :], mask=mask_m[None, :],
                    other=0.0).to(tl.float32)                      # (G, BM)
        e3 = _exp_floor(a)[:, :, None] + _exp_floor(b)[None, :, :]
        emax = tl.maximum(tl.max(e3, axis=1), _exp_floor(acc))
        ulp = _pow2(emax - F)
        inv = _pow2(F - emax)
        p = a[:, :, None] * b[None, :, :]
        t = (p * inv[:, None, :]).to(tl.int32).to(tl.float32)
        s = tl.sum(t, axis=1) * ulp + (acc * inv).to(tl.int32).to(tl.float32) * ulp
        ea = _exp_floor(s)
        acc = tl.where(s == 0.0, s, (s * _pow2(F - ea)).to(tl.int32).to(tl.float32)
                       * _pow2(ea - F))
    c = c_ptr + rn[:, None] * M + rm[None, :]
    tl.store(c, acc, mask=mask_n[:, None] & mask_m[None, :])


def fast_accum_mm_triton(a_t, b_t, block_n=8, block_m=64, num_warps=2, rz=False):
    """a_t (K, N), b_t (K, M): E4M3 values in a 16-bit float dtype, contiguous."""
    K, N = a_t.shape
    K2, M = b_t.shape
    assert K == K2 and K % GROUP == 0, (a_t.shape, b_t.shape)
    assert a_t.is_contiguous() and b_t.is_contiguous()
    c = torch.empty(N, M, device=a_t.device, dtype=torch.float32)
    if N <= SMALL_N and not rz:
        bn = 1 if N == 1 else (2 if N == 2 else 4)
        grid = (triton.cdiv(N, bn), triton.cdiv(M, SMALL_BM))
        _fast_accum_kernel_3d[grid](a_t, b_t, c, N, M, K, BLOCK_N=bn, BLOCK_M=SMALL_BM,
                                    G=GROUP, F=FRAC_BITS, num_warps=SMALL_WARPS)
        return c
    # Tiles from scripts/fp8_emul_tune.py on H200 (20B in_proj, 1024 rows): BN=8,
    # BM=64, 2 warps 183 ms; the first default (BN=32, BM=64, 4 warps) 2992 ms.
    # Larger row tiles spill. The round-toward-zero variant was not faster.
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m))
    kern = _fast_accum_kernel_rz if rz else _fast_accum_kernel
    kern[grid](a_t, b_t, c, N, M, K, BLOCK_N=block_n, BLOCK_M=block_m,
               G=GROUP, F=FRAC_BITS, num_warps=num_warps)
    return c


def static_fp8_linear_emulated_triton(x, w_t_e4m3_values, input_scale, weight_scale):
    """Drop-in for static_fp8_linear without FP8 hardware.

    w_t_e4m3_values: (K, M) bf16 tensor holding the E4M3 weight values, transposed.
    """
    sx = input_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    sw = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    xq = (x.float() / sx).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    a_t = xq.reshape(-1, x.shape[-1]).t().to(torch.bfloat16).contiguous()
    acc = fast_accum_mm_triton(a_t, w_t_e4m3_values)
    return (acc * sx * sw).to(x.dtype).reshape(*x.shape[:-1], w_t_e4m3_values.shape[1])


# ---------------------------------------------------------------------------
# Tensor-core model (FP8_SURGERY.md §5). One K step = one Hopper accumulation
# group of 32. The group's exact-product sum comes from tl.dot on E4M3-valued
# bf16 operands (fp32 accumulate); the group's alignment exponent
# max_k(e(a_k) + e(b_k)) from a second tl.dot on operands re-encoded as
# 2^(6 e): their sum S satisfies 2^(6 emax) <= S <= 32 * 2^(6 emax) < 2^(6 emax + 6),
# so emax = floor(log2(S) / 6) exactly. The cross-group rule is then applied in
# registers. What it cannot reproduce: the per-product truncation inside a group
# (the group sum is truncated once instead); quantified in FP8_SURGERY.md.
# ---------------------------------------------------------------------------


@triton.jit
def _enc6(x):
    """bf16 2^(6 floor(log2|x|)) for E4M3 values (|e| <= 9), 0 for x == 0."""
    v = _pow2(6 * _exp_floor(x))
    return tl.where(x == 0.0, 0.0, v).to(tl.bfloat16)


@triton.jit
def _fast_accum_tc_kernel(a_ptr, b_ptr, c_ptr, N, M, K,
                          BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                          F: tl.constexpr, PER_GROUP_TRUNC: tl.constexpr,
                          RZ: tl.constexpr):
    # a: (N, K) row-major activations; b: (K, M) row-major (= weight^T); c: (N, M)
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, 32)
    mask_n = rn < N
    mask_m = rm < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(0, K, 32):
        a = tl.load(a_ptr + rn[:, None] * K + (k0 + rk)[None, :], mask=mask_n[:, None],
                    other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * M + rm[None, :], mask=mask_m[None, :],
                    other=0.0)
        s = tl.dot(a, b, out_dtype=tl.float32)
        S = tl.dot(_enc6(a.to(tl.float32)), _enc6(b.to(tl.float32)), out_dtype=tl.float32)
        eg = tl.where(S == 0.0, -100000, (_exp_floor(S) + 600) // 6 - 100)
        e = tl.maximum(eg, _exp_floor(acc))
        if RZ:
            # truncation by round-toward-zero adds: full-rate FADDs instead of the
            # quarter-rate float<->int conversions
            big = _pow2(e - F + 23)
            if PER_GROUP_TRUNC:
                s = _trunc_to(s, big)
            s += _trunc_to(acc, big)
            acc = tl.where(s == 0.0, s, _trunc_to(s, _pow2(_exp_floor(s) - F + 23)))
        else:
            ulp = _pow2(e - F)
            inv = _pow2(F - e)
            if PER_GROUP_TRUNC:
                s = (s * inv).to(tl.int32).to(tl.float32) * ulp
            s += (acc * inv).to(tl.int32).to(tl.float32) * ulp
            ea = _exp_floor(s)
            acc = tl.where(s == 0.0, s, (s * _pow2(F - ea)).to(tl.int32).to(tl.float32)
                           * _pow2(ea - F))
    c = c_ptr + rn[:, None] * M + rm[None, :]
    tl.store(c, acc, mask=mask_n[:, None] & mask_m[None, :])


def tc_config(N):
    """(BLOCK_N, BLOCK_M, num_warps, num_stages) by row count; from
    scripts/fp8_emul_bench.py --tc-sweep (no autotuning: it must not run inside
    vLLM's CUDA-graph capture)."""
    for limit, cfg in TC_CONFIGS:
        if N <= limit:
            return cfg
    return TC_CONFIGS[-1][1]


# H200 sweep (log tc-bench-h-46581454): 16x64/4w for <=16 rows, 64x64/4w to 64,
# 128x128/8w beyond, for both the 20B and the 1B shape
TC_CONFIGS = [(16, (16, 64, 4, 3)), (64, (64, 64, 4, 3)), (1 << 62, (128, 128, 8, 4))]


def fast_accum_mm_tc(a, b_t, per_group_trunc=True, cfg=None, rz=True):
    """a (N, K), b_t (K, M): E4M3 values in bf16, contiguous -> fp32 (N, M)."""
    N, K = a.shape
    K2, M = b_t.shape
    assert K == K2 and K % GROUP == 0, (a.shape, b_t.shape)
    a = a.to(torch.bfloat16).contiguous()
    assert b_t.dtype == torch.bfloat16 and b_t.is_contiguous()
    c = torch.empty(N, M, device=a.device, dtype=torch.float32)
    bn, bm, nw, ns = cfg or tc_config(N)
    grid = (triton.cdiv(N, bn), triton.cdiv(M, bm))
    _fast_accum_tc_kernel[grid](a, b_t, c, N, M, K, BLOCK_N=bn, BLOCK_M=bm, F=FRAC_BITS,
                                PER_GROUP_TRUNC=per_group_trunc, RZ=rz, num_warps=nw,
                                num_stages=ns)
    return c


def static_fp8_linear_emulated_tc(x, w_t_e4m3_values, input_scale, weight_scale):
    """Tensor-core FP8 fast-accumulation emulation (any GPU with bf16 tl.dot)."""
    sx = input_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    sw = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    xq = (x.float() / sx).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    acc = fast_accum_mm_tc(xq.reshape(-1, x.shape[-1]).to(torch.bfloat16), w_t_e4m3_values)
    return (acc * sx * sw).to(x.dtype).reshape(*x.shape[:-1], w_t_e4m3_values.shape[1])
