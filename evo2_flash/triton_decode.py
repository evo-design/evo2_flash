"""One fused kernel for an ordinary (q=1) Hyena decode step, bit-identical.

Between in_proj and out_proj, an ordinary decode step runs vLLM's
`causal_conv1d_update` (K=3 short filter), strided torch products for the
x1*v gate, `triton_verifier.fir_one` / `iir_one` for the inner op, then torch
again for `+ D*u` and `* x2`. `hyena_decode` does all of it in one launch and
reproduces every rounding point, so the output and all three cached state
slices match the unfused path bit for bit (`tests/test_triton_decode_step.py`):

  - short filter: fp32 accumulator over three products rounded to bf16, oldest
    column first, then rounded to bf16 -- what vLLM's update kernel computes;
  - u = x1 * v, D * u, y + D*u and x2 * y: bf16 after every op, like torch;
  - inner FIR: `triton_verifier._fir_sum`'s order (Reduce.cuh's, torch 2.11),
    restated over 16-tap chunks so the window is read and written in place in
    coalesced rows, without register gathers;
  - LI: s = s*p + u in fp32 without FMA, the readout's halving tree, and the
    poles as libdevice `exp` of the log-poles (torch's `expf`).

The whole kernel compiles with `enable_fp_fusion=False`: products that torch
rounds are rounded here too (libdevice `mul_rn` would flush denormals). Every
state element is loaded, then a barrier, then stored: the update is in place,
and Triton may replicate an element across warps.

Ordinary decode only. A speculative engine's verify path keeps its snapshots
in `triton_verifier`. Rows whose block is vLLM's null block (CUDA-graph padding)
write no state and a zero output. Only torch 2.11 is qualified.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .triton_prefill import QUALIFIED_TORCH


@triton.jit
def _bf16(x):
    """Round fp32 to bf16 and back, like a torch op with a bf16 output."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _short_tap(Z, WS, CS, ch, mask, CS1, CS2):
    """vLLM's K=3 update for channels `ch`: (bf16 output as fp32, col1, x)."""
    x = tl.load(Z + ch, mask, other=0).to(tl.float32)
    c0 = tl.load(CS + ch * CS1, mask, other=0).to(tl.float32)
    c1 = tl.load(CS + ch * CS1 + CS2, mask, other=0).to(tl.float32)
    w0 = tl.load(WS + ch * 3, mask, other=0).to(tl.float32)
    w1 = tl.load(WS + ch * 3 + 1, mask, other=0).to(tl.float32)
    w2 = tl.load(WS + ch * 3 + 2, mask, other=0).to(tl.float32)
    acc = ((0.0 + _bf16(c0 * w0)) + _bf16(c1 * w1)) + _bf16(x * w2)
    return _bf16(acc), c1, x


@triton.jit
def _hyena_decode_kernel(
    Z, SZ, WS, CS, CS0, CS1, CS2, SLOTS, SSL,
    WI, WI0, WI1, LP, LP0, LP1, IS, IS0, IS1, IS2, D, OUT, SO, H,
    K: tl.constexpr, IS_IIR: tl.constexpr, HAS_D: tl.constexpr, NULL_BLOCK: tl.constexpr,
    BH: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    mh = h < H
    blk = tl.load(SLOTS + b * SSL).to(tl.int64)
    out = OUT + b.to(tl.int64) * SO + h
    if blk == NULL_BLOCK:
        tl.store(out, tl.zeros((BH,), OUT.dtype.element_ty), mh)
        return

    # ---- short filter on the interleaved triple (3h, 3h+1, 3h+2) = (x2, x1, v)
    zrow = Z + b.to(tl.int64) * SZ
    csb = CS + blk * CS0
    x2, x2_c1, x2_x = _short_tap(zrow, WS, csb, 3 * h, mh, CS1, CS2)
    x1, x1_c1, x1_x = _short_tap(zrow, WS, csb, 3 * h + 1, mh, CS1, CS2)
    v, v_c1, v_x = _short_tap(zrow, WS, csb, 3 * h + 2, mh, CS1, CS2)
    u = _bf16(x1 * v)
    isb = IS + blk * IS0

    if IS_IIR:
        s = tl.arange(0, K)
        m2 = mh[:, None]
        sp = isb + h[:, None] * IS1 + s[None, :] * IS2
        st = tl.load(sp, m2, other=0)
        lp = tl.load(LP + h[:, None] * LP0 + s[None, :] * LP1, m2, other=0)
        res = tl.load(WI + h[:, None] * WI0 + s[None, :] * WI1, m2, other=0)
        st = st * libdevice.exp(lp) + u[:, None]
        # Readout: p[s] += p[s + 8], += p[s + 4], += p[s + 2], += p[s + 1]; take s=0.
        p = tl.sum(tl.reshape(st * res, (BH, 2, K // 2)), 1)
        p = tl.sum(tl.reshape(p, (BH, 2, K // 4)), 1)
        p = tl.sum(tl.reshape(p, (BH, 2, K // 8)), 1)
        p = tl.sum(tl.reshape(p, (BH, 2, K // 16)), 1)
        y = _bf16(tl.reshape(p, (BH,)) + 0.0)
        tl.debug_barrier()
        tl.store(sp, st, m2)
    elif K == 128:
        # _fir_sum over taps k = 16j + 4m + l: accumulator (m, l) sums j = 0..7 in
        # order, then ((m0 + m1) + m2) + m3 per lane, then (l0 + l2) + (l1 + l3).
        m = tl.arange(0, 4)[:, None, None]
        l = tl.arange(0, 4)[None, :, None]
        hh = h[None, None, :]
        acc = tl.zeros((4, 4, BH), tl.float32)
        for j in tl.static_range(8):
            k = 16 * j + 4 * m + l
            mk = (k < K - 1) & (hh < H)
            sp = isb + hh * IS1 + k * IS2
            st = tl.load(sp, mk, other=0)
            w = tl.load(WI + hh * WI0 + k * WI1, mk, other=0).to(tl.float32)
            acc += st * w
            tl.debug_barrier()
            tl.store(sp - IS2, st, mk & (k >= 1))     # shift the window by one
        part = tl.sum(tl.where(m == 0, acc, 0), 0)
        part += tl.sum(tl.where(m == 1, acc, 0), 0)
        part += tl.sum(tl.where(m == 2, acc, 0), 0)
        part += tl.sum(tl.where(m == 3, acc, 0), 0)                     # (4, BH)
        lane = tl.arange(0, 4)[:, None]
        p0 = tl.sum(tl.where(lane == 0, part, 0), 0)
        p1 = tl.sum(tl.where(lane == 1, part, 0), 0)
        p2 = tl.sum(tl.where(lane == 2, part, 0), 0)
        p3 = tl.sum(tl.where(lane == 3, part, 0), 0)
        last = tl.load(WI + h * WI0 + (K - 1) * WI1, mh, other=0).to(tl.float32)
        y = _bf16(((p0 + p2) + (p1 + p3)) + last * u)
        tl.store(isb + h * IS1 + (K - 2) * IS2, u, mh)
    else:
        # K = 7: ((p0 + p4) + (p1 + p5)) + p2) + p3, as _fir_sum's width-7 branch.
        tl.static_assert(K == 7)
        sp = isb + h * IS1
        s0 = tl.load(sp, mh, other=0)
        s1 = tl.load(sp + IS2, mh, other=0)
        s2 = tl.load(sp + 2 * IS2, mh, other=0)
        s3 = tl.load(sp + 3 * IS2, mh, other=0)
        s4 = tl.load(sp + 4 * IS2, mh, other=0)
        s5 = tl.load(sp + 5 * IS2, mh, other=0)
        wp = WI + h * WI0
        a = (s0 * tl.load(wp, mh, other=0).to(tl.float32) + 0.0) \
            + s4 * tl.load(wp + 4 * WI1, mh, other=0).to(tl.float32)
        bb = (s1 * tl.load(wp + WI1, mh, other=0).to(tl.float32) + 0.0) \
            + s5 * tl.load(wp + 5 * WI1, mh, other=0).to(tl.float32)
        c = s2 * tl.load(wp + 2 * WI1, mh, other=0).to(tl.float32) + 0.0
        d = s3 * tl.load(wp + 3 * WI1, mh, other=0).to(tl.float32) + 0.0
        last = tl.load(wp + 6 * WI1, mh, other=0).to(tl.float32)
        y = _bf16((((a + bb) + c) + d) + last * u)
        tl.debug_barrier()
        tl.store(sp, s1, mh)
        tl.store(sp + IS2, s2, mh)
        tl.store(sp + 2 * IS2, s3, mh)
        tl.store(sp + 3 * IS2, s4, mh)
        tl.store(sp + 4 * IS2, s5, mh)
        tl.store(sp + 5 * IS2, u, mh)

    # Short-filter window: (c0, c1) <- (c1, x). Every load above precedes a barrier.
    tl.debug_barrier()
    for_c = csb + 3 * h * CS1
    tl.store(for_c, x2_c1.to(CS.dtype.element_ty), mh)
    tl.store(for_c + CS2, x2_x.to(CS.dtype.element_ty), mh)
    tl.store(for_c + CS1, x1_c1.to(CS.dtype.element_ty), mh)
    tl.store(for_c + CS1 + CS2, x1_x.to(CS.dtype.element_ty), mh)
    tl.store(for_c + 2 * CS1, v_c1.to(CS.dtype.element_ty), mh)
    tl.store(for_c + 2 * CS1 + CS2, v_x.to(CS.dtype.element_ty), mh)

    if HAS_D:
        y = _bf16(y + _bf16(tl.load(D + h, mh, other=0).to(tl.float32) * u))
    tl.store(out, (x2 * y).to(OUT.dtype.element_ty), mh)


def decode_supported(z, short_w, conv_state, inner, inner_state, D=None, log_poles=None):
    """Whether `hyena_decode` reproduces the unfused path for these tensors.

    The width-2 short window is the ordinary layout; a speculative engine's is
    wider, and its verify path never comes here.
    """
    H = inner.shape[0]
    iir = log_poles is not None
    return (QUALIFIED_TORCH and z.is_cuda and z.dtype == torch.bfloat16
            and z.stride(-1) == 1 and short_w.shape == (3 * H, 3)
            and short_w.is_contiguous() and short_w.dtype == torch.bfloat16
            and conv_state.dtype == torch.bfloat16 and conv_state.shape[-1] == 2
            and inner_state.dtype == torch.float32 and H % 128 == 0
            and (D is None or D.dtype == torch.bfloat16)
            and (inner.shape[1] == 16 and inner.dtype == log_poles.dtype == torch.float32
                 if iir else inner.shape[1] in (7, 128) and inner.dtype == torch.bfloat16))


def hyena_decode(z, short_w, conv_state, slots, inner, inner_state, D=None,
                 log_poles=None, null_block=0):
    """x2 * (inner(x1*v) + D*(x1*v)) for one decode token per row; states in place.

    z: (B, 3H) bf16 in_proj rows. short_w: (3H, 3). conv_state: (blocks, 3H, 2)
    bf16, any strides. slots: (B,) block per row (a strided view is fine).
    inner: the FIR taps (H, 7 | 128), or the LI residues (H, 16) when
    `log_poles` (H, 16) is given. inner_state: (blocks, H, K-1 | 16) fp32 view.
    D: (H,) or None. Returns (B, H) bf16.
    """
    B, H = z.shape[0], inner.shape[0]
    out = torch.empty(B, H, dtype=z.dtype, device=z.device)
    if not B:
        return out
    is_iir = log_poles is not None
    K = inner.shape[1]
    lp = log_poles if is_iir else inner
    # Fixed launch shapes (no autotuning inside CUDA graphs), from an H100 sweep.
    bh, warps = (64, 2) if K == 128 else (64, 4) if is_iir else (256, 4)
    _hyena_decode_kernel[(B, triton.cdiv(H, bh))](
        z, z.stride(0), short_w, conv_state, *conv_state.stride(), slots, slots.stride(0),
        inner, *inner.stride(), lp, *lp.stride(), inner_state, *inner_state.stride(),
        D if D is not None else inner, out, out.stride(0), H,
        K=K, IS_IIR=is_iir, HAS_D=D is not None, NULL_BLOCK=null_block, BH=bh,
        num_warps=warps, enable_fp_fusion=False,
    )
    return out
