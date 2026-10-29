"""Software model of Hopper's FP8 "fast accumulation" GEMM (FP8_SURGERY.md).

The released FP8 Evo 2 checkpoints were trained (and are evaluated) with
Transformer Engine's forward FP8 GEMM, which runs on sm_90 with fast
accumulation: the whole K loop stays in the tensor-core accumulator instead of
being promoted to fp32 every few MMAs. That accumulator is not fp32. Measured
with scripts/fp8_accum_probe.py, bit-exact on random E4M3 operands at K=32, 64
and 1920 (and on real 1B projections, scripts/fp8_ablate.py):

  for each group of 32 consecutive k (ascending):
      e     = max( max_k floor(log2|a_k|) + floor(log2|b_k|),  floor(log2|acc|) )
      ulp   = 2^(e - 13)
      acc   = trunc(acc / ulp) * ulp + sum_k trunc(a_k * b_k / ulp) * ulp
      acc   = acc truncated to 13 fraction bits of its own exponent
  out = acc * scale_a * scale_b          (fp32 epilogue, then cast)

`trunc` rounds toward zero. The alignment exponent comes from the *operand*
exponents (the product's mantissa, in [1, 4), is not looked at), and every
product is truncated individually before the sum, so small terms that share a
group with a large one, or arrive after the accumulator has grown, are dropped.

Everything below is exact in fp32: products of two E4M3 values are exact, every
aligned term is a multiple of ulp with < 2^21 ulps, so the group sum is exact.
Runs on any GPU or CPU; no FP8 hardware is needed.
"""

import torch

E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0
GROUP = 32
FRAC_BITS = 13


def _pow2_floor(x):
    """2^floor(log2|x|) exactly (frexp, not log2), and 0 for x == 0."""
    _, e = torch.frexp(x)
    return torch.where(x == 0, torch.zeros_like(x), torch.ldexp(torch.ones_like(x), e - 1))


def _trunc_to(x, ulp):
    return torch.trunc(x / ulp) * ulp


def fast_accum_mm(a, b_t, chunk_rows=256, group=GROUP, frac_bits=FRAC_BITS):
    """Emulated Hopper fast-accumulation FP8 GEMM.

    a:   (N, K) tensor whose values are E4M3-representable (any float dtype)
    b_t: (M, K) same, i.e. the weight in nn.Linear layout
    returns the fp32 accumulator (N, M), *before* the scale epilogue.
    """
    a = a.float()
    b = b_t.float().t().contiguous()  # (K, M)
    N, K = a.shape
    M = b.shape[1]
    pb = _pow2_floor(b.abs())
    out = torch.empty(N, M, dtype=torch.float32, device=a.device)
    two_f = 2.0 ** -frac_bits
    for r0 in range(0, N, chunk_rows):
        ar = a[r0:r0 + chunk_rows]
        pa = _pow2_floor(ar.abs())
        acc = torch.zeros(ar.shape[0], M, dtype=torch.float32, device=a.device)
        for k0 in range(0, K, group):
            ag, bg = ar[:, k0:k0 + group], b[k0:k0 + group]
            mx = (pa[:, k0:k0 + group, None] * pb[None, k0:k0 + group]).amax(1)
            mx = torch.maximum(mx, _pow2_floor(acc.abs()))
            ulp = torch.where(mx == 0, torch.ones_like(mx), mx * two_f)
            s = _trunc_to(ag[:, :, None] * bg[None], ulp[:, None, :]).sum(1)
            acc = _trunc_to(acc, ulp) + s
            u2 = _pow2_floor(acc.abs()) * two_f
            acc = torch.where(u2 == 0, acc, _trunc_to(acc, torch.where(u2 == 0, torch.ones_like(u2), u2)))
        out[r0:r0 + chunk_rows] = acc
    return out


def quantize_input(x, input_scale):
    """TE/vLLM static per-tensor E4M3 activation quantisation, as fp32 values."""
    return (x.float() / input_scale).clamp(-FP8_MAX, FP8_MAX).to(E4M3).float()


def static_fp8_linear_emulated(x, weight_e4m3, input_scale, weight_scale, **kw):
    """Drop-in for evo2_layers.model.static_fp8_linear without FP8 hardware."""
    input_scale = input_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    weight_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1)
    xq = quantize_input(x, input_scale).reshape(-1, x.shape[-1])
    acc = fast_accum_mm(xq, weight_e4m3.to(x.device).float(), **kw)
    out = (acc * input_scale * weight_scale).to(x.dtype)
    return out.reshape(*x.shape[:-1], weight_e4m3.shape[0])
