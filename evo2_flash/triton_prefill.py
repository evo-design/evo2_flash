"""Triton kernels for the Hyena prefill: varlen inner FIR and the LI final state.

Both replace per-sequence or per-channel torch loops in `mixers.py` and are
bit-identical to them, which stay as the reference and as a diagnostic fallback
(`EVO2_TORCH_INNER_FIR=1`, `EVO2_TORCH_IIR_STATE=1`; `EVO2_TORCH_FILTER=1`
restores the whole torch LI prefill). `tests/test_triton_prefill.py` asserts the
bit identity, so a torch or cuDNN version that changes an order fails there.

FIR. The window written into the cache is a copy of input values. The output
accumulates the K taps oldest first, one fp32 multiply-add per tap, which is
what cuDNN's depthwise `conv1d` does under torch 2.11.

LI state. `iir_segment`'s torch path sums `seq * exp(lp * lag)` over t with ATen's
CUDA reduction, whose order is fixed by `setReduceConfig` (Reduce.cuh) from the
shape, the product tensor's layout and the GPU's SM count. `aten_sum_config`
restates that function, and the kernels follow the order it implies. Each
running sum is a strided sequence p[a] + p[a + D] + ..., so all of them run in
parallel and in order, followed by the same combination tree:
  - layout contiguous in t (a chunk of >= 2 state channels): `_rows_kernel`; lanes
    take 4-element vectors (from the 16-byte-aligned point of each row, with a
    head and a tail) or, below 128 elements, single elements; then a shuffle tree.
  - layout contiguous in channels (a chunk of 1, i.e. long segments): `_split_kernel`
    per CTA (4 warps x CTAS blocks), then `_global_kernel` for the cross-CTA sum.
The products are rounded as torch rounds them: libdevice `exp` (torch's `expf`)
and no FMA contraction (`enable_fp_fusion=False`; libdevice `mul_rn` would flush
denormals, which torch does not). Shapes whose configuration is not restated here,
such as warp-split rows (H < 1920 at long L), return None and run the torch path.
ATen's order is version-dependent, so both kernels run only on torch 2.11.

No autotuning: a fixed launch configuration keeps the result deterministic.

Two traps, both silent. Masked lanes still evaluate `exp`, and a negative lag
with real poles overflows to inf (`0 * inf = NaN`), so masked accumulators are
selected around, never added to. vLLM's metadata tensors can be strided views
(`state_indices_tensor_p` is column 0 of a 2-D table under speculation), so every
index tensor's stride is passed into the kernel.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# ATen's reduction order and cuDNN's convolution order are version-dependent.
QUALIFIED_TORCH = torch.__version__.split("+")[0].startswith("2.11.")


# ---------------------------------------------------------------- IIR final state
def _last_pow2(n):
    return 1 << (int(n).bit_length() - 1)


@functools.lru_cache(maxsize=None)
def _device_limits(index):
    p = torch.cuda.get_device_properties(index)
    return p.multi_processor_count, p.max_threads_per_multi_processor


def aten_sum_config(num_outputs, L, fastest, ovs, device_index):
    """`setReduceConfig` (Reduce.cuh, torch 2.11) for a float sum over one dim of L.

    `fastest`: the reduced dim is the input's fastest-moving one. `ovs`: ATen's
    output vector size when it is not. Returns lanes `bw` and warps `bh` per block,
    `vec_in`, the input step per thread, the CTAs per output and ATen's input_mult.
    """
    num_mp, max_t_sm = _device_limits(device_index)
    vec_in, step_in, step_out, imult = False, 1, 1, [0, 0, 0]
    ovs = 1 if fastest else ovs
    if fastest:
        dim0, dim1 = L, num_outputs
        if dim0 >= 128:
            vec_in, dim0 = True, dim0 // 4
    else:
        dim0, dim1 = num_outputs // ovs, L
    mnt = 512 // ovs
    d0 = _last_pow2(dim0) if dim0 < mnt else mnt
    d1 = _last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(d0, 32)
    bh = min(d1, mnt // bw)
    bw = min(d0, mnt // bh)
    if fastest:
        imult[0], step_in = step_in, step_in * bw
    else:
        step_out *= bw
    if triton.cdiv(L, step_in) >= min(bh * 16, 256):
        imult[1], step_in = step_in, step_in * bh
    else:
        step_out *= bh
    ctas = 1
    target = num_mp * (max_t_sm // (bw * bh))
    grid = triton.cdiv(num_outputs // ovs, step_out)
    vpt = triton.cdiv(L, step_in)
    if imult[1] != 0 and vpt >= 256 and grid <= target:
        ctas = max(min(triton.cdiv(target, grid), triton.cdiv(vpt, 16)), triton.cdiv(vpt, 256))
        if ctas > 1:
            imult[2], step_in = step_in, step_in * ctas
    return dict(vec_in=vec_in, bw=bw, bh=bh, step_in=step_in, ctas=ctas, imult=imult)


@triton.jit
def _term(x_ptr, sxh, sxl, h, t, lp, L, mask):
    """seq[h, t] * exp(lp * (L-1-t)), each op rounded as torch rounds it."""
    x = tl.load(x_ptr + h.to(tl.int64) * sxh + t.to(tl.int64) * sxl, mask=mask, other=0.0)
    return x * libdevice.exp(lp * (L - 1 - t).to(tl.float32))


@triton.jit
def _halve(v, N: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    """One warp_shfl_down / block_y_reduce step on axis 1 of (A, N, B): v[k] + v[k + N/2]."""
    return tl.sum(tl.reshape(v, (A, 2, N // 2, B)), axis=1)


@triton.jit
def _tree(v, N: tl.constexpr, A: tl.constexpr, B: tl.constexpr):
    """ATen's halving tree over axis 1 of (A, N, B), N a power of two <= 32."""
    if N >= 32:
        v = _halve(v, 32, A, B)
    if N >= 16:
        v = _halve(v, 16, A, B)
    if N >= 8:
        v = _halve(v, 8, A, B)
    if N >= 4:
        v = _halve(v, 4, A, B)
    if N >= 2:
        v = _halve(v, 2, A, B)
    return v


@triton.jit
def _pick(x0, x1, x2, x3, st1, st2, st3):
    """The candidate at offset `start` (0..3) of each row."""
    return tl.where(st3, x3, tl.where(st2, x2, tl.where(st1, x1, x0)))


@triton.jit
def _rows_kernel(x_ptr, lp_ptr, out_ptr, H, L, sxh, sxl, slp, s_begin, n_s, c,
                 n_full, n_iter,
                 NS: tl.constexpr, BLOCK_H: tl.constexpr, BW: tl.constexpr, VEC: tl.constexpr,
                 ALIGNED: tl.constexpr):
    """Reduction along t (product contiguous in t): one row per (h, s), BW lanes.

    Layout (BLOCK_H, NS, BW). Inputs are loaded once per (h, lane) and shared by
    the NS rows of that channel.
    """
    h = (tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H))[:, None, None]
    sn = tl.arange(0, NS)[None, :, None]
    s = s_begin + sn
    mr = (h < H) & (sn < n_s)
    mh = h < H
    lp = tl.load(lp_ptr + h * slp + s, mask=mr, other=0.0)
    j = tl.arange(0, BW)[None, None, :]
    xp = x_ptr + h.to(tl.int64) * sxh
    a0 = tl.zeros((BLOCK_H, NS, BW), tl.float32)
    a1 = tl.zeros((BLOCK_H, NS, BW), tl.float32)
    a2 = tl.zeros((BLOCK_H, NS, BW), tl.float32)
    a3 = tl.zeros((BLOCK_H, NS, BW), tl.float32)
    if VEC:
        # Row o = h*c + (s - s_begin) % c of the (H, c, L) product starts at element
        # o*L: ATen reduces its first (4 - o*L % 4) % 4 elements as a head, then
        # aligned vectors of 4 (lane j takes vectors j, j+32, ...), then a tail.
        o = h.to(tl.int64) * c + (s - s_begin) % c
        sh = ((o * L) % 4).to(tl.int32)
        start = (4 - sh) % 4
        st1, st2, st3 = start == 1, start == 2, start == 3
        endp = L - start
        nvec = endp // 4
        head = (sh > 0) & (j >= sh) & (j < 4) & mr
        a0 = tl.where(head, 0.0 + _term(x_ptr, sxh, sxl, h, j - sh, lp, L, head), a0)
        for k in range(n_full):           # every row has all 32 lanes' vectors
            b = 4 * (j + 32 * k)
            x0 = tl.load(xp + b.to(tl.int64) * sxl, mask=mh, other=0.0)
            x1 = tl.load(xp + (b + 1).to(tl.int64) * sxl, mask=mh, other=0.0)
            x2 = tl.load(xp + (b + 2).to(tl.int64) * sxl, mask=mh, other=0.0)
            x3 = tl.load(xp + (b + 3).to(tl.int64) * sxl, mask=mh, other=0.0)
            if ALIGNED:
                lag = (L - 1 - b).to(tl.float32)
                a0 = a0 + x0 * libdevice.exp(lp * lag)
                a1 = a1 + x1 * libdevice.exp(lp * (lag - 1.0))
                a2 = a2 + x2 * libdevice.exp(lp * (lag - 2.0))
                a3 = a3 + x3 * libdevice.exp(lp * (lag - 3.0))
            else:
                x4 = tl.load(xp + (b + 4).to(tl.int64) * sxl, mask=mh, other=0.0)
                x5 = tl.load(xp + (b + 5).to(tl.int64) * sxl, mask=mh, other=0.0)
                x6 = tl.load(xp + (b + 6).to(tl.int64) * sxl, mask=mh, other=0.0)
                lag = (L - 1 - start - b).to(tl.float32)
                a0 = a0 + _pick(x0, x1, x2, x3, st1, st2, st3) * libdevice.exp(lp * lag)
                a1 = a1 + _pick(x1, x2, x3, x4, st1, st2, st3) * libdevice.exp(lp * (lag - 1.0))
                a2 = a2 + _pick(x2, x3, x4, x5, st1, st2, st3) * libdevice.exp(lp * (lag - 2.0))
                a3 = a3 + _pick(x3, x4, x5, x6, st1, st2, st3) * libdevice.exp(lp * (lag - 3.0))
        for k in range(n_full, n_iter):
            vi = j + 32 * k
            mv = (vi < nvec) & mr
            t = start + 4 * vi
            a0 = tl.where(mv, a0 + _term(x_ptr, sxh, sxl, h, t, lp, L, mv), a0)
            a1 = tl.where(mv, a1 + _term(x_ptr, sxh, sxl, h, t + 1, lp, L, mv), a1)
            a2 = tl.where(mv, a2 + _term(x_ptr, sxh, sxl, h, t + 2, lp, L, mv), a2)
            a3 = tl.where(mv, a3 + _term(x_ptr, sxh, sxl, h, t + 3, lp, L, mv), a3)
        mt = (j < endp % 4) & mr
        a0 = tl.where(mt, a0 + _term(x_ptr, sxh, sxl, h, start + 4 * nvec + j, lp, L, mt), a0)
    else:
        # Element k of lane j (t = j + k*BW) goes to accumulator k % 4, in order.
        for m in range(n_full):           # all four of every lane's elements are < L
            t = j + (4 * m) * BW
            x0 = tl.load(xp + t.to(tl.int64) * sxl, mask=mh, other=0.0)
            x1 = tl.load(xp + (t + BW).to(tl.int64) * sxl, mask=mh, other=0.0)
            x2 = tl.load(xp + (t + 2 * BW).to(tl.int64) * sxl, mask=mh, other=0.0)
            x3 = tl.load(xp + (t + 3 * BW).to(tl.int64) * sxl, mask=mh, other=0.0)
            a0 = a0 + x0 * libdevice.exp(lp * (L - 1 - t).to(tl.float32))
            a1 = a1 + x1 * libdevice.exp(lp * (L - 1 - t - BW).to(tl.float32))
            a2 = a2 + x2 * libdevice.exp(lp * (L - 1 - t - 2 * BW).to(tl.float32))
            a3 = a3 + x3 * libdevice.exp(lp * (L - 1 - t - 3 * BW).to(tl.float32))
        for m in range(n_full, n_iter):
            t = j + (4 * m) * BW
            m0 = (t < L) & mr
            m1 = (t + BW < L) & mr
            m2 = (t + 2 * BW < L) & mr
            m3 = (t + 3 * BW < L) & mr
            a0 = tl.where(m0, a0 + _term(x_ptr, sxh, sxl, h, t, lp, L, m0), a0)
            a1 = tl.where(m1, a1 + _term(x_ptr, sxh, sxl, h, t + BW, lp, L, m1), a1)
            a2 = tl.where(m2, a2 + _term(x_ptr, sxh, sxl, h, t + 2 * BW, lp, L, m2), a2)
            a3 = tl.where(m3, a3 + _term(x_ptr, sxh, sxl, h, t + 3 * BW, lp, L, m3), a3)
    R: tl.constexpr = BLOCK_H * NS
    v = _tree(tl.reshape(((a0 + a1) + a2) + a3, (R, BW, 1)), BW, R, 1)
    tl.store(out_ptr + h * slp + s, tl.reshape(v, (BLOCK_H, NS, 1)), mask=mr)


@triton.jit
def _split_kernel(x_ptr, lp_ptr, buf_ptr, out_ptr, H, L, sxh, sxl, slp, s_begin, n_s,
                  STEP, CTAS, n_full, n_iter,
                  NS: tl.constexpr, BLOCK_H: tl.constexpr, BH: tl.constexpr,
                  BLOCK_C: tl.constexpr, FINAL: tl.constexpr):
    """Reduction across t (product contiguous in channels): virtual thread
    j = y + BH*cta takes t = j + k*STEP into accumulator k % 4, in order; the BH
    warps of each CTA are then combined by block_y_reduce.

    Layout (J, BLOCK_H, NS) with J = BLOCK_C * BH virtual threads.
    """
    J: tl.constexpr = BLOCK_C * BH
    h = (tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H))[None, :, None]
    sn = tl.arange(0, NS)[None, None, :]
    s = s_begin + sn
    mo = (h < H) & (sn < n_s)
    lp = tl.load(lp_ptr + h * slp + s, mask=mo, other=0.0)
    jj = tl.arange(0, J)
    cta = tl.program_id(1) * BLOCK_C + jj // BH
    j = (jj % BH + BH * cta)[:, None, None]
    mj = (cta < CTAS)[:, None, None]
    mx = (h < H) & mj
    xp = x_ptr + h.to(tl.int64) * sxh
    a0 = tl.zeros((J, BLOCK_H, NS), tl.float32)
    a1 = tl.zeros((J, BLOCK_H, NS), tl.float32)
    a2 = tl.zeros((J, BLOCK_H, NS), tl.float32)
    a3 = tl.zeros((J, BLOCK_H, NS), tl.float32)
    # Full rounds: every t < L. Rows past H or CTAs past CTAS add zeros into
    # accumulators that are never stored.
    for m in range(n_full):
        t = j + (4 * m) * STEP
        x0 = tl.load(xp + t.to(tl.int64) * sxl, mask=mx, other=0.0)
        x1 = tl.load(xp + (t + STEP).to(tl.int64) * sxl, mask=mx, other=0.0)
        x2 = tl.load(xp + (t + 2 * STEP).to(tl.int64) * sxl, mask=mx, other=0.0)
        x3 = tl.load(xp + (t + 3 * STEP).to(tl.int64) * sxl, mask=mx, other=0.0)
        a0 = a0 + x0 * libdevice.exp(lp * (L - 1 - t).to(tl.float32))
        a1 = a1 + x1 * libdevice.exp(lp * (L - 1 - t - STEP).to(tl.float32))
        a2 = a2 + x2 * libdevice.exp(lp * (L - 1 - t - 2 * STEP).to(tl.float32))
        a3 = a3 + x3 * libdevice.exp(lp * (L - 1 - t - 3 * STEP).to(tl.float32))
    for m in range(n_full, n_iter):
        t = j + (4 * m) * STEP
        m0 = (t < L) & mj & mo
        m1 = (t + STEP < L) & mj & mo
        m2 = (t + 2 * STEP < L) & mj & mo
        m3 = (t + 3 * STEP < L) & mj & mo
        a0 = tl.where(m0, a0 + _term(x_ptr, sxh, sxl, h, t, lp, L, m0), a0)
        a1 = tl.where(m1, a1 + _term(x_ptr, sxh, sxl, h, t + STEP, lp, L, m1), a1)
        a2 = tl.where(m2, a2 + _term(x_ptr, sxh, sxl, h, t + 2 * STEP, lp, L, m2), a2)
        a3 = tl.where(m3, a3 + _term(x_ptr, sxh, sxl, h, t + 3 * STEP, lp, L, m3), a3)
    X: tl.constexpr = BLOCK_H * NS
    v = _tree(tl.reshape(((a0 + a1) + a2) + a3, (BLOCK_C, BH, X)), BH, BLOCK_C, X)
    w = tl.reshape(v, (BLOCK_C, BLOCK_H, NS))
    if FINAL:
        tl.store(out_ptr + h * slp + s, w, mask=mo)
    else:
        ctab = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None, None]
        tl.store(buf_ptr + ctab.to(tl.int64) * (H * NS) + h * NS + sn, w,
                 mask=(ctab < CTAS) & mo)


@triton.jit
def _global_kernel(buf_ptr, out_ptr, H, slp, s_begin, n_s, CTAS,
                   NS: tl.constexpr, BLOCK_H: tl.constexpr, BH: tl.constexpr):
    """global_reduce: thread y sums CTAs y, y+BH, ... in order, then block_y over y."""
    X: tl.constexpr = BLOCK_H * NS
    x_ = tl.arange(0, X)
    h = tl.program_id(0) * BLOCK_H + x_ // NS
    s = s_begin + x_ % NS
    mo = (h < H) & (x_ % NS < n_s)
    y = tl.arange(0, BH)[:, None]
    g = tl.zeros((BH, X), tl.float32)
    for c0 in range(0, CTAS, BH):
        m = (c0 + y < CTAS) & mo[None, :]
        w = tl.load(buf_ptr + (c0 + y).to(tl.int64) * (H * NS) + (h * NS + x_ % NS)[None, :],
                    mask=m, other=0.0)
        g = tl.where(m, g + w, g)
    g = _tree(tl.reshape(g, (1, BH, X)), BH, 1, X)
    tl.store(out_ptr + h * slp + s, tl.reshape(g, (X,)), mask=mo)


@functools.lru_cache(maxsize=4096)
def _iir_plan(H, L, S, blk, layout, device_index):
    """Launches reproducing torch's loop over chunks of `blk` state channels, one per
    chunk size; None when a chunk's ATen configuration is not restated here."""
    ovs = 4 if H % 4 == 0 else (2 if H % 2 == 0 else 1)
    chunks = {}
    for s0 in range(0, S, blk):
        chunks.setdefault(min(blk, S - s0), []).append(s0)
    plan = []
    for c, starts in chunks.items():
        s_begin, n_s = starts[0], len(starts) * c
        # The (H, c, L) product torch builds: contiguous in t unless c == 1 and seq
        # is the mixer's token-major view, where it inherits seq's channel-fastest layout.
        fastest = layout == "contiguous" or c > 1
        cfg = aten_sum_config(H * c, L, fastest, ovs, device_index)
        ns = triton.next_power_of_2(n_s)
        if fastest:
            if cfg["imult"][1] != 0 or cfg["ctas"] != 1:
                return None
            bw = cfg["bw"]
            if cfg["vec_in"]:
                n_iter, n_full = triton.cdiv(L // 4, 32), ((L - 3) // 4) // 32
            else:
                n_iter, n_full = triton.cdiv(L, 4 * bw), L // (4 * bw)
            bh = max(1, 32 // ns)
            plan.append(("rows", (triton.cdiv(H, bh),), (s_begin, n_s, c, n_full, n_iter),
                         dict(NS=ns, BLOCK_H=bh, BW=bw, VEC=cfg["vec_in"], ALIGNED=L % 4 == 0)))
        else:
            if cfg["imult"][1] != 1 or (cfg["ctas"] > 1 and cfg["imult"][2] != cfg["bh"]):
                return None
            bh_, ctas, step = cfg["bh"], cfg["ctas"], cfg["step_in"]
            bh = max(1, 512 // ns)
            plan.append(("split", (triton.cdiv(H, bh), ctas),
                         (s_begin, n_s, step, ctas, L // (4 * step), triton.cdiv(L, 4 * step)),
                         dict(NS=ns, BLOCK_H=bh, BH=bh_, BLOCK_C=1, FINAL=ctas == 1)))
            if ctas > 1:
                gh = max(1, 128 // ns)
                plan.append(("global", (triton.cdiv(H, gh),), (s_begin, n_s, ctas),
                             dict(NS=ns, BLOCK_H=gh, BH=bh_)))
    return plan


def iir_final_state(seq, lp, blk):
    """s[h, s] = sum_t seq[h, t] * exp(lp[h, s] * (L-1-t)), bit for bit as torch computes it.

    seq: (H, L) fp32, token-major (the mixer's view) or contiguous.  lp: (H, S)
    log-poles.  blk: the number of state channels `iir_segment`'s torch path
    processes per pass (it fixes the layout torch reduces). Returns (H, S) fp32,
    or None where the kernels do not reproduce torch's order (the caller then
    runs the torch path).
    """
    if not (QUALIFIED_TORCH and seq.is_cuda and seq.dtype == torch.float32):
        return None
    H, L = seq.shape
    S = lp.shape[1]
    st = seq.stride()
    layout = "token-major" if st == (1, H) else ("contiguous" if st == (L, 1) else None)
    if layout is None:
        return None
    dev = seq.device.index if seq.device.index is not None else torch.cuda.current_device()
    plan = _iir_plan(H, L, S, blk, layout, dev)
    if plan is None:
        return None
    lp = lp.contiguous().float()
    out = torch.empty(H, S, device=seq.device, dtype=torch.float32)
    buf = None
    for kind, grid, args, meta in plan:
        if kind == "rows":
            _rows_kernel[grid](seq, lp, out, H, L, st[0], st[1], lp.stride(0), *args, **meta,
                               num_warps=4, enable_fp_fusion=False)
        elif kind == "split":
            if not meta["FINAL"]:
                buf = torch.empty(args[3], H, meta["NS"], device=seq.device, dtype=torch.float32)
            _split_kernel[grid](seq, lp, buf, out, H, L, st[0], st[1], lp.stride(0), *args,
                                **meta, num_warps=4, enable_fp_fusion=False)
        else:
            _global_kernel[grid](buf, out, H, lp.stride(0), *args, **meta, num_warps=4,
                                 enable_fp_fusion=False)
    return out


# ---------------------------------------------------------------- inner FIR prefill
@triton.jit
def _fir_prefill_kernel(
    u_ptr, su, w_ptr, st_ptr, st_b, st_d, st_t, seq_ptr, blk_ptr, sblk, tile_ptr,
    out_ptr, so, H, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pt = tl.program_id(0)
    sid = tl.load(tile_ptr + 2 * pt)
    t0 = tl.load(tile_ptr + 2 * pt + 1)
    a = tl.load(seq_ptr + 3 * sid)
    e = tl.load(seq_ptr + 3 * sid + 1)
    init = tl.load(seq_ptr + 3 * sid + 2)
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mt = offs_t < e
    md = offs_d < H
    m2 = mt[:, None] & md[None, :]
    rows = offs_t.to(tl.int64)[:, None] * su + offs_d[None, :]
    acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
    if t0 - (K - 1) >= a:
        # Interior tile: the whole receptive field lies inside this segment.
        for k in range(K):
            xk = tl.load(u_ptr + rows - (K - 1 - k) * su, mask=m2, other=0.0)
            wk = tl.load(w_ptr + offs_d * K + k, mask=md, other=0.0)
            acc += xk.to(tl.float32) * wk.to(tl.float32)[None, :]
    else:
        # Left context: the cached window for a continuation, zeros otherwise.
        blk = tl.load(blk_ptr + sid * sblk).to(tl.int64)
        for k in range(K):
            src = offs_t - (K - 1) + k
            inside = (src >= a)[:, None]
            xk = tl.load(u_ptr + rows - (K - 1 - k) * su, mask=m2 & inside,
                         other=0.0).to(tl.float32)
            col = (K - 1) + (src - a)
            xs = tl.load(st_ptr + blk * st_b + offs_d[None, :] * st_d
                         + col[:, None] * st_t,
                         mask=m2 & (~inside) & (init != 0), other=0.0)
            wk = tl.load(w_ptr + offs_d * K + k, mask=md, other=0.0)
            acc += tl.where(inside, xk, xs) * wk.to(tl.float32)[None, :]
    tl.store(out_ptr + offs_t.to(tl.int64)[:, None] * so + offs_d[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=m2)


@triton.jit
def _fir_state_kernel(u_ptr, su, st_ptr, st_b, st_d, st_t, seq_ptr, blk_ptr, sblk, H,
                      K: tl.constexpr, KP: tl.constexpr, BLOCK_D: tl.constexpr):
    sid = tl.program_id(0)
    a = tl.load(seq_ptr + 3 * sid)
    e = tl.load(seq_ptr + 3 * sid + 1)
    init = tl.load(seq_ptr + 3 * sid + 2)
    if e <= a:
        return
    blk = tl.load(blk_ptr + sid * sblk).to(tl.int64)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, KP)
    md = offs_d[:, None] < H
    mc = (offs_c < K - 1)[None, :]
    src = e - (K - 1) + offs_c          # packed row feeding window column c
    inside = (src >= a)[None, :]
    x = tl.load(u_ptr + src.to(tl.int64)[None, :] * su + offs_d[:, None],
                mask=md & mc & inside, other=0.0).to(tl.float32)
    base = st_ptr + blk * st_b + offs_d[:, None] * st_d
    col = (K - 1) + (src - a)           # for a short continuation: old columns
    xs = tl.load(base + col[None, :] * st_t, mask=md & mc & (~inside) & (init != 0),
                 other=0.0)
    new = tl.where(inside, x, xs)
    # Small tensors are replicated across warps: every load must land before any
    # warp overwrites the window it reads from.
    tl.debug_barrier()
    tl.store(base + offs_c[None, :] * st_t, new, mask=md & mc)


def fir_prefill(u, w, state, seq_table, blocks, tile_table, block_t):
    """Depthwise causal FIR over packed prefill rows, carrying per-sequence state.

    u: (T, H) packed rows.  w: (H, K) cross-correlation order (w[:, -1] pairs
    with the current sample).  state: (blocks, H, >=K-1) fp32, any strides.
    seq_table: (n, 3) int32 [start, end, has_initial_state] per segment.
    blocks: (n,) state block per segment, any stride -- under speculation vLLM
    passes column 0 of an (n, 1+num_spec) table.  tile_table: (ntiles, 2) int32
    [segment, first row], built for `block_t` (`fir_prefill_block_t`).

    Returns the filter output, like `_depthwise_prefill_torch`, and writes each
    segment's final (K-1)-input window into `state`.
    """
    T, H = u.shape
    K = w.shape[1]
    assert u.stride(1) == 1, "rows are addressed as t * stride(0) + channel"
    out = u.new_empty(T, H)
    ntiles = tile_table.shape[0]
    if not ntiles:
        return out
    w = w.contiguous()
    block_d = 64 if K > 16 else 128
    _fir_prefill_kernel[(ntiles, triton.cdiv(H, block_d))](
        u, u.stride(0), w, state, *state.stride(), seq_table, blocks, blocks.stride(0),
        tile_table, out, out.stride(0), H,
        K=K, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4,
    )
    # A second launch, so that no tile can read a window this one overwrites.
    kp = triton.next_power_of_2(max(K - 1, 2))
    sd = 32 if kp > 16 else 128
    _fir_state_kernel[(seq_table.shape[0], triton.cdiv(H, sd))](
        u, u.stride(0), state, *state.stride(), seq_table, blocks, blocks.stride(0), H,
        K=K, KP=kp, BLOCK_D=sd, num_warps=4,
    )
    return out


def fir_prefill_block_t(k):
    """Rows per FIR prefill tile; the tile table is built for this value."""
    return 32 if k > 16 else 64
