"""Fused paged Hyena verification, preserving all rejection snapshots.

The FIR reduction explicitly reproduces Torch 2.11 Reduce.cuh for native
(K, H) state, H divisible by 128, and widths 7/128. Do not replace it with
an arbitrary associative sum: near-tied greedy logits depend on this order.
FMA is disabled to preserve the eager recurrence's separate multiply/add.
"""
import torch
import triton
import triton.language as tl


# ATen reduction geometry/order is version-dependent; other versions fall back.
_QUALIFIED_TORCH = torch.__version__.split("+")[0].startswith("2.11.")


@triton.jit
def _fir_sum(p, W: tl.constexpr, BH: tl.constexpr):
    if W == 128:
        lane = tl.arange(0, 4)
        a = tl.full((4, BH), 0, tl.float32)
        b = tl.full((4, BH), 0, tl.float32)
        c = tl.full((4, BH), 0, tl.float32)
        d = tl.full((4, BH), 0, tl.float32)
        for j in tl.static_range(8):
            a = a + tl.gather(p, tl.broadcast_to((lane + j * 16)[:, None], (4, BH)), 0)
            b = b + tl.gather(p, tl.broadcast_to((lane + j * 16 + 4)[:, None], (4, BH)), 0)
            c = c + tl.gather(p, tl.broadcast_to((lane + j * 16 + 8)[:, None], (4, BH)), 0)
            d = d + tl.gather(p, tl.broadcast_to((lane + j * 16 + 12)[:, None], (4, BH)), 0)
        partial = ((a + b) + c) + d
        p0 = tl.sum(tl.where(lane[:, None] == 0, partial, 0), 0)
        p1 = tl.sum(tl.where(lane[:, None] == 1, partial, 0), 0)
        p2 = tl.sum(tl.where(lane[:, None] == 2, partial, 0), 0)
        p3 = tl.sum(tl.where(lane[:, None] == 3, partial, 0), 0)
        return (p0 + p2) + (p1 + p3)
    else:
        # Width 7: no inter-warp split; four accumulators, two tail values.
        a = tl.sum(tl.where(tl.arange(0, 8)[:, None] == 0, p, 0), 0)
        b = tl.sum(tl.where(tl.arange(0, 8)[:, None] == 1, p, 0), 0)
        c = tl.sum(tl.where(tl.arange(0, 8)[:, None] == 2, p, 0), 0)
        d = tl.sum(tl.where(tl.arange(0, 8)[:, None] == 3, p, 0), 0)
        a += tl.sum(tl.where(tl.arange(0, 8)[:, None] == 4, p, 0), 0)
        b += tl.sum(tl.where(tl.arange(0, 8)[:, None] == 5, p, 0), 0)
        return ((a + b) + c) + d


@triton.jit
def _fir_kernel(X, Wt, State, InSlots, OutSlots, Starts, Ends, Accepted, Y,
                H: tl.constexpr, W: tl.constexpr, Q: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr,
                WS0: tl.constexpr, WS1: tl.constexpr,
                SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
                IS0: tl.constexpr, IS1: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
                BK: tl.constexpr, BH: tl.constexpr, SINGLE: tl.constexpr = False):
    req = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    k = tl.arange(0, BK)
    if SINGLE:
        first, end = req, req + 1
        source = tl.load(InSlots + req * IS0).to(tl.int64)
    else:
        first = tl.load(Starts + req)
        end = tl.load(Ends + req)
        accepted = tl.load(Accepted + req)
        # vLLM block IDs are int32; padded offsets may exceed 2**31 elements.
        source = tl.load(InSlots + req * IS0 + (accepted - 1) * IS1).to(tl.int64)
    st = tl.load(State + source * SS0 + h[None, :] * SS1 + k[:, None] * SS2,
                 (h[None, :] < H) & (k[:, None] < W - 1), other=0).to(tl.float32)
    w = tl.load(Wt + h[None, :] * WS0 + k[:, None] * WS1,
                (h[None, :] < H) & (k[:, None] < W - 1), other=0).to(tl.float32)
    last = tl.load(Wt + h * WS0 + (W - 1) * WS1, h < H, other=0).to(tl.float32)
    for pos in range(Q):
        active = first + pos < end
        x = tl.load(X + (first + pos) * XS0 + h * XS1, active & (h < H), other=0).to(tl.float32)
        y = _fir_sum(st * w, W, BH) + last * x
        tl.store(Y + (first + pos) * H + h, y, active & (h < H))
        shifted = tl.gather(st, tl.broadcast_to(tl.minimum(k + 1, BK - 1)[:, None], (BK, BH)), 0)
        st = tl.where(k[:, None] == W - 2, x[None, :], shifted)
        st = tl.where(k[:, None] < W - 1, st, 0)
        destination = tl.load(OutSlots + req * OS0 + pos * OS1).to(tl.int64)
        # Null slots during memory profiling may alias across requests.
        tl.store(State + destination * SS0 + h[None, :] * SS1 + k[:, None] * SS2,
                 st, active & (destination != 0) & (h[None, :] < H) & (k[:, None] < W - 1))


@triton.jit
def _iir_kernel(X, Poles, Residues, State, InSlots, OutSlots, Starts, Ends, Accepted, Y,
                H: tl.constexpr, S: tl.constexpr, Q: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr,
                PS0: tl.constexpr, PS1: tl.constexpr, RS0: tl.constexpr, RS1: tl.constexpr,
                SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr,
                IS0: tl.constexpr, IS1: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
                BH: tl.constexpr, SINGLE: tl.constexpr = False):
    req = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    s = tl.arange(0, S)
    if SINGLE:
        first, end = req, req + 1
        source = tl.load(InSlots + req * IS0).to(tl.int64)
    else:
        first, end = tl.load(Starts + req), tl.load(Ends + req)
        # vLLM block IDs are int32; padded offsets may exceed 2**31 elements.
        source = tl.load(InSlots + req * IS0 + (tl.load(Accepted + req) - 1) * IS1).to(tl.int64)
    st = tl.load(State + source * SS0 + h[:, None] * SS1 + s[None, :] * SS2,
                 h[:, None] < H, other=0).to(tl.float32)
    poles = tl.load(Poles + h[:, None] * PS0 + s[None, :] * PS1, h[:, None] < H, other=0).to(tl.float32)
    res = tl.load(Residues + h[:, None] * RS0 + s[None, :] * RS1, h[:, None] < H, other=0).to(tl.float32)
    for pos in range(Q):
        active = first + pos < end
        x = tl.load(X + (first + pos) * XS0 + h * XS1, active & (h < H), other=0).to(tl.float32)
        st = st * poles + x[:, None]
        destination = tl.load(OutSlots + req * OS0 + pos * OS1).to(tl.int64)
        tl.store(State + destination * SS0 + h[:, None] * SS1 + s[None, :] * SS2,
                 st, active & (destination != 0) & (h[:, None] < H))
        partial = st * res
        for offset in tl.static_range(4):
            shifted = tl.gather(partial, tl.broadcast_to(((s + (8 >> offset)) % S)[None, :], (BH, S)), 1)
            partial = partial + shifted
        y = tl.sum(tl.where(s[None, :] == 0, partial, 0), 1)
        tl.store(Y + (first + pos) * H + h, y, active & (h < H))


def fir_supported(x, weight, state):
    return (_QUALIFIED_TORCH and x.dtype in (torch.float32, torch.bfloat16)
            and x.is_cuda and x.shape[1] >= 128 and x.shape[1] % 128 == 0
            and weight.shape[1] in (7, 128) and state.stride(1) == 1
            and state.dtype == torch.float32)


def fir_snapshots(x, weight, state, metadata):
    ins, outs, starts, ends, accepted = metadata
    y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    _fir_kernel[(ins.shape[0], triton.cdiv(x.shape[1], 32))](
        x, weight, state, ins, outs, starts, ends, accepted, y,
        x.shape[1], weight.shape[1], outs.shape[1], *x.stride(), *weight.stride(),
        *state.stride(), *ins.stride(), *outs.stride(),
        triton.next_power_of_2(weight.shape[1] - 1), 32, enable_fp_fusion=False)
    return y


def iir_supported(x, poles, residues, state):
    return (_QUALIFIED_TORCH and x.dtype in (torch.float32, torch.bfloat16)
            and x.is_cuda and poles.shape[1] == 16 and state.stride(-1) == 1
            and state.dtype == poles.dtype == residues.dtype == torch.float32)


def iir_snapshots(x, poles, residues, state, metadata):
    ins, outs, starts, ends, accepted = metadata
    y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    _iir_kernel[(ins.shape[0], triton.cdiv(x.shape[1], 32))](
        x, poles, residues, state, ins, outs, starts, ends, accepted, y,
        x.shape[1], poles.shape[1], outs.shape[1], *x.stride(), *poles.stride(),
        *residues.stride(), *state.stride(), *ins.stride(), *outs.stride(),
        32, enable_fp_fusion=False)
    return y


def fir_one(x, weight, state, slots):
    """One token per request; no snapshot metadata or output-zeroing kernels."""
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _fir_kernel[(x.shape[0], triton.cdiv(x.shape[1], 32))](
        x, weight, state, slots, slots, slots, slots, slots, y,
        x.shape[1], weight.shape[1], 1, *x.stride(), *weight.stride(),
        *state.stride(), slots.stride(0), 0, slots.stride(0), 0,
        triton.next_power_of_2(weight.shape[1] - 1), 32,
        SINGLE=True, enable_fp_fusion=False)
    return y


def iir_one(x, poles, residues, state, slots):
    """One diagonal-IIR update per request, preserving its FP32 state."""
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _iir_kernel[(x.shape[0], triton.cdiv(x.shape[1], 32))](
        x, poles, residues, state, slots, slots, slots, slots, slots, y,
        x.shape[1], poles.shape[1], 1, *x.stride(), *poles.stride(),
        *residues.stride(), *state.stride(), slots.stride(0), 0, slots.stride(0), 0,
        32, SINGLE=True, enable_fp_fusion=False)
    return y
