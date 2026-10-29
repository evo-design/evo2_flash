"""vLLM MambaBase mixers for Evo 2's three Hyena variants.

Semantics are fixed by `evo2_layers/` and its golden tests; this file is the
vLLM-flavoured restatement (varlen packing, paged state, continuous batching).

State layout per layer. Every mixer declares the SAME spec -- vLLM asserts
identical MambaSpecs across mamba layers once prefix caching is on -- and each
uses only the slices relevant to its recurrence:
    kv_cache[0] : outer FIR window over the 3H projection, (3H, K_short-1)
    kv_cache[1] : inner FIR snapshots over x1*v, (H, K_MR-1) **float32**
                    SE uses [..., :6], MR uses [..., :127]
    kv_cache[2] : diagonal LI temporal state, (H, 16) **float32**; with
                  speculative decoding (H, 17), where [:, 0, 16] of every block
                  is that snapshot's position stamp (see below)
The separate slices are intentional: vLLM's prefix-cache copier assigns one
copy contract to each slice. Outer FIR state is sliding; inner FIR and LI state
are temporal.

Rollback is resolved from position stamps, not from vLLM's acceptance counts.
vLLM 0.26 passes ``num_accepted_tokens`` only on steps where some request has
drafts, so on a step without drafts the counts are missing even though the
previous verify step may have accepted several tokens. Every Hyena layer stamps
each snapshot it writes with the number of tokens that snapshot represents; the
live snapshot is the one whose stamp equals the request's computed-token count
(``evo2_flash.attention`` supplies that count on every step). The stamp lives in
slice 2 for every layer kind, so vLLM's temporal copy moves it together with the
inner-FIR and LI snapshots of the same block.

vLLM orders tokens decode-first, then prefill (see ShortConv). Both halves are
handled separately and re-concatenated in that order. Under speculative decode,
outer-FIR state follows vLLM's native convolution contract: block zero contains
a window extended by K draft positions and the preceding accepted-token count
selects an offset within that window. Inner-FIR and diagonal-IIR state instead
follow vLLM's temporal contract, with one complete snapshot per verifier block.
Treating the packed tokens as independent requests is incorrect:
``num_decode_tokens`` is a token count while ``num_decodes`` is a request count.

TP note: projections are Replicated rather than sharded. The Hyena channel
layout is interleaved (x1,x2,v per triple, regrouped by `interleave()`), so
naive column sharding would split a triple across ranks. TP is out of scope
here — the target is a single H100.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .triton_hcl import hcl_compute_filter


# Bound the IIR state-accumulation temporary; long prefills retain O(H*L) memory.
_STATE_ACC_BUDGET = 64 << 20

# Test-only: assert vLLM's acceptance counts, where sent, agree with the stamps.
_CHECK_VLLM_COUNTS = os.environ.get("EVO2_CHECK_VLLM_COUNTS") == "1"


def _prefill_kernel(x, switch) -> bool:
    """True when prefill runs the Triton kernel (``triton_prefill``) `switch` disables.

    The kernels are bit-identical to the torch path below on the torch version
    they were checked against (tests/test_triton_prefill.py); elsewhere torch runs.
    ``EVO2_TORCH_INNER_FIR=1`` and ``EVO2_TORCH_IIR_STATE=1`` select the torch
    path, which is kept as the reference those kernels are checked against -- a
    diagnostic, not a user option. Read per call so tests can flip it.
    """
    if not x.is_cuda or os.environ.get(switch) == "1":
        return False
    from .triton_prefill import QUALIFIED_TORCH
    return QUALIFIED_TORCH


def _prefill_segments(m):
    """Segment boundaries and initial-state flags as Python lists.

    `m.query_start_loc_p` and `m.has_initial_states_p` are **CUDA** tensors
    (`mamba_attn.py:508,512`), so `int(qsl[j])` / `bool(flags[j])` are
    device->host reads, and each one *synchronises*: it blocks until every
    previously queued kernel has finished, so the CPU cannot run ahead and the
    32-layer forward executes in lockstep with this Python loop instead of as a
    pipelined queue. Three such reads per sequence, in each of the 27 Hyena
    layers, is 648 syncs per step at batch 8 — measured at ~87 ms of the 99 ms a
    13-position verify step costs (bench/bench_verify_cost.py).

    vLLM builds a CPU twin (`query_start_loc_p_cpu`, `mamba_attn.py:504`) but
    does not expose it on `BaseMambaAttentionMetadata`, so transfer once and
    memoise on the metadata object. That is safe because `build()` constructs a
    fresh `metadata_cls(...)` every step (`mamba_attn.py:533`) — the cache cannot
    outlive the tensors it describes. Falls back to a per-call transfer if the
    object refuses attributes, which is still 2 syncs per layer instead of 3B.
    """
    cached = getattr(m, "_evo2_prefill_segments", None)
    if cached is None:
        flags = getattr(m, "has_initial_states_p", None)
        cached = (m.query_start_loc_p.tolist(),
                  None if flags is None else flags.tolist())
        try:
            m._evo2_prefill_segments = cached
        except (AttributeError, TypeError):
            pass
    return cached


def _segment_initial_state(m, j):
    """True when prefill segment j *continues* a sequence rather than starting it.

    vLLM splits a prefill longer than `max_num_batched_tokens` across scheduler
    steps (16384 by default on an H100 via the offline `LLM` API — and because
    chunked prefill is enabled, raising `max_model_len` does NOT raise it). The
    continuation segments must resume from the cached state; treating them as
    fresh sequences silently corrupts the tokens just past each boundary.
    """
    if os.environ.get("EVO2_IGNORE_CHUNK_STATE") == "1":
        # Diagnostic: reproduce the pre-fix behaviour (treat every segment as a
        # fresh sequence). Kept so the regression is reproducible on demand --
        # see tests/test_chunked_prefill.py. Never set this in production.
        return False
    flags = _prefill_segments(m)[1]
    return flags is not None and bool(flags[j])


def iir_segment(seq, res, lp, s_prev, force_torch=False):
    """One prefill segment of the diagonal LTI SSM  s_t = p*s_{t-1} + x_t.

    seq: (H, L) fp32 inputs.  res, lp: (H, S) residues and log-poles.
    s_prev: (H, S) state inherited from an earlier segment, or None to start fresh.
    Returns (y, s_last) with y = (H, L) and s_last = (H, S).

    Unrolling the recurrence from the inherited state s_{-1} gives
        s_t = sum_{tau<=t} p^(t-tau) x_tau  +  p^(t+1) s_{-1}
        y_t = (h * x)[t]                    +  sum_s res_s p_s^(t+1) s_{-1,s}
    with h[u] = sum_s res_s p_s^u the modal filter. The two s_{-1} terms are what a
    split prefill needs and what omitting them costs; they vanish when s_{-1} = 0.
    """
    H, L = seq.shape
    S = res.shape[1]
    t = torch.arange(L, device=seq.device, dtype=torch.float32)

    # The filter is one fused Triton launch (triton_hcl.py): 32x
    # faster and half the peak memory of the equivalent accumulation loop.
    h = hcl_compute_filter(res, lp, t, force_torch=force_torch)  # (H, L)
    lag = t.flip(0)  # L-1 ... 0

    # Final state, per channel: s_{L-1} = sum_tau p^(L-1-tau) x_tau. Contracting
    # over tau while keeping S needs an (H, S, L) intermediate -- 34 GB at H=4096,
    # S=16, L=131072, which OOMs -- so the channels are blocked to a byte budget.
    # A long prefill gets blk=1, i.e. one channel at a time and the original
    # O(H*L) peak; a short segment does all S at once. That matters because at
    # small L the 5 launches per channel dwarf the arithmetic: for a 13-position
    # speculative verify block this is 80 launches collapsing to 5, and the
    # channel loop is ~82% of everything iir_segment launches
    # (bench/bench_verify_cost.py; STATUS.md "Verify cost measured").
    #
    # On CUDA, triton_prefill.iir_final_state computes the same sum bit for bit,
    # 2-27x faster (L=13..16384), without the temporary; `blk` tells it which layout torch would
    # reduce. The torch code below stays for EVO2_TORCH_IIR_STATE=1,
    # EVO2_TORCH_FILTER=1 (the all-torch LI prefill), CPU, and the shapes the
    # kernel does not reproduce (it returns None).
    blk = min(S, max(1, int(_STATE_ACC_BUDGET // max(H * L * 4, 1))))
    st = None
    if not force_torch and _prefill_kernel(seq, "EVO2_TORCH_IIR_STATE"):
        from .triton_prefill import iir_final_state
        st = iir_final_state(seq, lp, blk)
    if st is not None:
        pass
    elif blk == S:
        st = (seq.unsqueeze(1) * torch.exp(lp.unsqueeze(-1) * lag)).sum(-1)
    else:
        st = torch.empty(H, S, device=seq.device, dtype=torch.float32)
        for s0 in range(0, S, blk):
            sl = slice(s0, min(s0 + blk, S))
            st[:, sl] = (seq.unsqueeze(1)
                         * torch.exp(lp[:, sl].unsqueeze(-1) * lag)).sum(-1)

    n = 2 * L
    y = torch.fft.irfft(
        torch.fft.rfft(seq, n=n) * (torch.fft.rfft(h, n=n) / n),
        n=n,
        norm="forward",
    )[..., :L]

    if s_prev is not None:
        st = st + torch.exp(lp * L) * s_prev  # p^L * s_{-1}
        # sum_s (res_s * s_prev_s) * exp(lp_s * (t+1)) is the modal filter with
        # rescaled residues, so it reuses the fused kernel rather than a second
        # (H, S, L) accumulation loop.
        y = y + hcl_compute_filter(res * s_prev, lp, t + 1.0, force_torch=force_torch)
    return y, st


# Standalone multi-token recurrence oracles. Runtime uses the paged snapshots below.


def inner_decode_multi(u, w, win, num_accepted):
    """Inner FIR (K=7 / K=128) over q positions per sequence, in ONE grouped conv.

    u:   (B, q, dim)          the anchor token plus the q-1 drafts
    w:   (dim, k)             cross-correlation order: w[:, -1] pairs with the
                              current sample, matching `_inner_decode`
    win: (B, dim, k-1 + q-1)  sliding window, oldest first
    num_accepted: (B,) int    positions accepted in the PREVIOUS round, >= 1

    Returns (y, new_win) with y (B, q, dim) and new_win the same shape as `win`.

    The window layout is vLLM's spec-decode convention, documented in
    `causal_conv1d.py::_causal_conv1d_update_kernel`:

        before forward     [history1, ..., historyM]
        after  forward     [history2, ..., historyM, draft1, ..., draftN]
        after accepting a  [history_{a+1}, ..., historyM, draft1, ..., draft_a]

    so the live (k-1)-window for this round starts at index `a-1` -- the kernel's
    `conv_state_token_offset = num_accepted_tokens - 1`. That index is exact, not
    a guess: with state_len = k+q-2 the last k-1 genuine inputs begin at
    (k+q-2) - q + a - (k-1) = a-1.
    """
    B, q, dim = u.shape
    k = w.shape[1]
    # Per-sequence offset into the window, so this stays one gather rather than a
    # loop over sequences -- the whole point of the uniform-q path.
    ar = torch.arange(k - 1, device=u.device)
    idx = (num_accepted.view(B, 1, 1) - 1 + ar.view(1, 1, -1)).expand(B, dim, k - 1)
    init = win.gather(2, idx)                            # (B, dim, k-1)
    seq_in = torch.cat([init, u.transpose(1, 2)], dim=-1)  # (B, dim, k-1+q)
    # F.conv1d takes (B, C, L) with groups=C, so the whole batch is one launch.
    y = F.conv1d(seq_in, w.unsqueeze(1), padding=0, groups=dim)  # (B, dim, q)
    return y.transpose(1, 2), seq_in[..., 1:]


def iir_decode_multi(u, res, lp, s0):
    """LI/IIR recurrence s_t = p*s_{t-1} + x_t over q positions per sequence.

    u:   (B, q, H) fp32 inputs.  res, lp: (H, S).  s0: (B, H, S) state entering
    the round, i.e. after the last accepted token.

    Returns (y, s_all) with y (B, q, H) and s_all (B, q, H, S).

    **Returning every intermediate state is the point.** The conv state can be
    rolled back by sliding a window of raw inputs; an IIR state cannot, because
    s_t depends on the entire history. Handing back all q states makes rollback
    exact -- the caller selects `s_all[:, a-1]` once verification says a -- and
    costs O(q*H*S) for the duration of one round (54 MB at B=16, q=13, H=4096,
    S=16, fp32, one layer at a time).

    The alternative, inverting the recurrence as s_{t-1} = (s_t - x_t)/p, is
    rejected: it divides by poles that approach zero while these states reach
    ~1e12, so it would lose precision exactly where this model has none to spare.

    q is fixed at capture time, so the Python loop is CUDA-graph safe.
    """
    p = lp.exp().unsqueeze(0)                            # (1, H, S)
    s = s0
    ys, ss = [], []
    for i in range(u.shape[1]):
        s = s * p + u[:, i].unsqueeze(-1)
        ys.append((res.unsqueeze(0) * s).sum(-1))
        ss.append(s)
    return torch.stack(ys, dim=1), torch.stack(ss, dim=1)


def iir_spec_layout(state_size, q):
    """Column layout of the LI slice under speculative decoding.

    Returns (S, n_prev_col, x_lo, x_hi): `[s_0 | n_prev | x_1..x_q]`, needing
    S + 1 + q columns in the standalone replay oracle. The runtime uses separate
    paged snapshots instead of packing this layout into its shared FIR slice.
    """
    return state_size, state_size, state_size + 1, state_size + 1 + q


def iir_spec_round(x_new, res, lp, store, num_accepted, q):
    """One speculative round of the LI recurrence, with exact rollback.

    x_new: (B, q, H) this round's block inputs (x1*v, not token embeddings)
    store: (B, H, S+1+q) fp32, the layout above, updated IN PLACE
    num_accepted: (B,) int -- positions of the PREVIOUS round that were consumed
                  (vLLM's `num_accepted_tokens`, minimum 1)

    Returns y (B, q, H).

    WHY REPLAY. The conv state rolls back by sliding a window of raw inputs; an
    IIR state cannot, because s_t depends on the whole history. Storing every s_t
    would want q*S = 208 columns at q=13 against the ~123 spare. Storing s_0 plus
    the q inputs needs S+1+q = 30 and reconstructs any s_a exactly:

        s_a = p^a s_0 + sum_{i<=a} p^(a-i) x_i

    Inverting the recurrence instead, s_{t-1} = (s_t - x_t)/p, is rejected: it
    divides by poles approaching zero while these states reach ~1e12.

    WHY n_prev. `num_accepted_tokens` has a minimum of 1, so it cannot express
    "there was no previous round" -- which is exactly the prefill->decode
    boundary, where the stored s_0 is the prefill's final state and must be used
    verbatim. Replaying 1 step there would return p*s_0, an extra decay that no
    output would obviously flag. So the layout carries how many stored inputs are
    real, the replay length is min(num_accepted, n_prev), and s_0 is prepended to
    the history so that 0 steps is representable.

    GRAPH SAFETY. The replay always runs the full q steps and *gathers* the
    wanted state per sequence rather than branching on the count, so the work is
    identical across the batch and there is no device->host sync. It does double
    the IIR step count for the round, which is the price of not storing q states.
    """
    B = x_new.shape[0]
    S = res.shape[1]
    _, n_col, x_lo, x_hi = iir_spec_layout(S, q)

    s0 = store[..., :S]                                   # (B, H, S)
    n_prev = store[..., n_col].to(torch.int64)            # (B, H) -- uniform per seq
    x_prev = store[..., x_lo:x_hi]                        # (B, H, q)

    # Replay the previous round, then pick the accepted point. index 0 is s_0
    # itself, so `steps` of 0 means "use it unchanged".
    _, hist = iir_decode_multi(x_prev.transpose(1, 2), res, lp, s0)   # (B,q,H,S)
    hist = torch.cat([s0.unsqueeze(1), hist], dim=1)                  # (B,q+1,H,S)
    steps = torch.minimum(num_accepted.to(torch.int64), n_prev[:, 0]).clamp(0, q)
    s_start = hist.gather(
        1, steps.view(B, 1, 1, 1).expand(B, 1, s0.shape[1], S)
    ).squeeze(1)

    y, _ = iir_decode_multi(x_new, res, lp, s_start)
    store[..., :S] = s_start
    store[..., n_col] = float(q)
    store[..., x_lo:x_hi] = x_new.transpose(1, 2)
    return y


def _depthwise_prefill_torch(x_p, w, m, state, dim, k):
    """Per-sequence depthwise causal conv over the prefill rows, carrying state.

    Torch stand-in for `causal_conv1d_fn`, needed because vLLM's Triton kernel
    enumerates col0..col3 per KERNEL_WIDTH and only supports widths 2-5, while
    Evo 2's inner filters are K=7 and K=128.

    `w` is in cross-correlation order — w[:, -1] pairs with the current sample and
    w[:, 0] with the sample k-1 steps back — matching `_inner_decode`. The left
    context is therefore the previous k-1 inputs, oldest first: zeros for a fresh
    sequence, the cached window for a continuation. Prepending it explicitly and
    convolving with `padding=0` is equivalent to the old `padding=k-1` in the
    fresh case and correct in the chunked case, which the old code got wrong.
    """
    qsl = _prefill_segments(m)[0]
    out = torch.empty_like(x_p)
    wt = w.float().unsqueeze(1)  # (dim, 1, k) depthwise
    for j in range(m.num_prefills):
        a, b = int(qsl[j]), int(qsl[j + 1])
        if b <= a:
            continue
        idx = m.state_indices_tensor_p[j]
        seq = x_p[a:b].transpose(0, 1).float()  # (dim, L)
        if _segment_initial_state(m, j):
            prev = state[idx, :, : k - 1].float()  # (dim, k-1)
        else:
            prev = torch.zeros(dim, k - 1, device=seq.device, dtype=torch.float32)
        seq_in = torch.cat([prev, seq], dim=-1)  # (dim, L + k - 1)
        y = F.conv1d(seq_in.unsqueeze(0), wt, padding=0, groups=dim)
        out[a:b] = y[0].transpose(0, 1).to(out.dtype)
        # seq_in is always at least k-1 long, so the new window is never partial
        # even for a chunk shorter than the filter.
        state[idx, :, : k - 1] = seq_in[..., -(k - 1):].to(state.dtype)
    return out


def _prefill_tables(m, block_t, device):
    """Device tables for `triton_prefill.fir_prefill`, built once per step.

    seq: (n, 3) int32 [start, end, has_initial_state] per prefill segment.
    tiles: (ntiles, 2) int32 [segment, first row], `block_t` rows per tile.
    Built from the host lists `_prefill_segments` already holds, and sent with a
    pinned non-blocking copy, so this adds no device->host synchronisation.
    Memoised on the metadata object for the same reason as `_prefill_segments`.
    """
    cache = getattr(m, "_evo2_prefill_tables", None)
    if cache is None:
        cache = {}
        try:
            m._evo2_prefill_tables = cache
        except (AttributeError, TypeError):
            pass
    if block_t not in cache:
        qsl = _prefill_segments(m)[0]
        seq, tiles = [], []
        for j in range(m.num_prefills):
            a, b = qsl[j], qsl[j + 1]
            seq.append((a, b, int(_segment_initial_state(m, j))))
            tiles.extend((j, t) for t in range(a, b, block_t))

        def dev(rows, width):
            t = torch.tensor(rows, dtype=torch.int32).reshape(-1, width)
            return t.pin_memory().to(device, non_blocking=True)

        cache[block_t] = (dev(seq, 3), dev(tiles, 2))
    return cache[block_t]


def _depthwise_prefill_triton(x_p, w, m, state, dim, k):
    """`_depthwise_prefill_torch` as two kernel launches, with no per-sequence loop.

    Same arguments and result; see `triton_prefill.fir_prefill`.
    """
    from .triton_prefill import fir_prefill, fir_prefill_block_t

    assert x_p.shape[1] == dim and w.shape == (dim, k)
    block_t = fir_prefill_block_t(k)
    seq, tiles = _prefill_tables(m, block_t, x_p.device)
    return fir_prefill(x_p, w, state, seq, m.state_indices_tensor_p, tiles, block_t)


def _accepted_tokens(m, num_reqs: int, device):
    accepted_tokens = getattr(m, "num_accepted_tokens", None)
    if accepted_tokens is None:
        return torch.ones(num_reqs, device=device, dtype=torch.long)
    if accepted_tokens.numel() < num_reqs:
        raise ValueError(
            "num_accepted_tokens is too short: "
            f"shape={tuple(accepted_tokens.shape)}, num_decodes={num_reqs}"
        )
    return accepted_tokens[:num_reqs].to(dtype=torch.long).clamp_min(1)


def _cache_all_decode_slots(m, slots, num_reqs: int, max_positions: int | None):
    """Resolve cache-all input/output windows from block-table pointers."""
    previous = getattr(m, "block_idx_last_scheduled_token_prev_step", None)
    scheduled = getattr(m, "block_idx_last_scheduled_token", None)
    if previous is None or scheduled is None:
        return slots, slots

    width = slots.shape[1] if max_positions is None else min(
        slots.shape[1], max_positions
    )
    offsets = torch.arange(width, device=slots.device, dtype=torch.long)
    input_columns = previous[:num_reqs, None].long() + offsets
    output_columns = scheduled[:num_reqs, None].long() + offsets
    return slots.gather(1, input_columns), slots.gather(1, output_columns)


def _decode_tensors(
    m, num_tokens: int, max_positions: int | None = None, accepted=None
):
    """Return GPU tensors describing each packed decode request.

    vLLM packs all decode tokens contiguously. In ordinary decode there is one
    token per request; in speculative verification there can be ``1 + K`` and
    ``state_indices_tensor_d`` has the matching number of columns.
    The accepted count belongs to the *previous* step and selects the prior
    snapshot that is the input to this step. The mixers pass ``accepted``
    resolved from position stamps; without it, vLLM's ``num_accepted_tokens``
    is used, which is absent (and read as 1) on steps without drafts.

    Keep the values on-device. Converting ``query_start_loc_d`` or
    ``num_accepted_tokens`` to Python scalars would introduce two GPU
    synchronizations in every Hyena layer and erase any speculative speedup.
    Shape errors are still rejected before indexing; value ranges are owned by
    vLLM's metadata builder.
    """
    slots = m.state_indices_tensor_d
    if slots.dim() == 1:
        slots = slots[:, None]
    if slots.dim() != 2:
        raise ValueError(
            "state_indices_tensor_d must be rank 1 or 2, got "
            f"shape={tuple(slots.shape)}"
        )

    num_reqs = int(m.num_decodes)
    if slots.shape[0] < num_reqs:
        raise ValueError(
            f"decode state rows={slots.shape[0]} smaller than num_decodes={num_reqs}"
        )

    starts = getattr(m, "query_start_loc_d", None)
    if starts is None:
        if num_tokens != num_reqs:
            raise ValueError(
                "packed multi-token decode requires query_start_loc_d: "
                f"num_decode_tokens={num_tokens}, num_decodes={num_reqs}"
            )
        starts = torch.arange(
            num_reqs + 1, device=slots.device, dtype=torch.int64
        )
    else:
        if starts.numel() < num_reqs + 1:
            raise ValueError(
                "query_start_loc_d is too short: "
                f"shape={tuple(starts.shape)}, num_decodes={num_reqs}"
            )
        starts = starts[: num_reqs + 1]

    if accepted is None:
        accepted = _accepted_tokens(m, num_reqs, slots.device)
    else:
        accepted = accepted[:num_reqs].to(dtype=torch.long)
    input_slots, output_slots = _cache_all_decode_slots(
        m, slots[:num_reqs], num_reqs, max_positions
    )
    return (
        input_slots,
        output_slots,
        starts[:-1].long(),
        starts[1:].long(),
        accepted,
    )


def _cache_safe_state_indices(indices, state):
    """Use the reserved null block for vLLM's undersized profiling cache.

    ``profile_cudagraph_memory`` deliberately allocates only one state block per
    possible request, while speculative metadata names one block per verifier
    position. Those extra dummy indices cannot be dereferenced. Real scheduling
    must have at least one distinct block per matrix entry; only the temporary
    profiling allocation violates that lower bound, and its state is discarded.
    The shape-only branch avoids a device synchronization and is constant while
    a CUDA graph is captured.
    """
    if indices.dim() == 2 and state.shape[0] < indices.numel():
        return torch.zeros_like(indices)
    return indices


def _fused(x) -> bool:
    """Decode and verify use the Triton kernels in ``triton_verifier``.

    They copy the torch path's reduction order and are bit-identical to it
    (tests/test_engine_consistency.py: identical statistics in every dense and
    speculative configuration on 1B and 7B). ``EVO2_TORCH_DECODE=1`` forces the
    torch path below, which is kept as the reference those kernels are checked
    against -- a diagnostic, not a user option. Read per call so tests can flip it.
    """
    return x.is_cuda and os.environ.get("EVO2_TORCH_DECODE") != "1"


def _fir_decode_one(x, weight, indices, state):
    """Ordinary decode, retaining its original Torch path as the reference."""
    if _fused(x):
        from .triton_verifier import fir_supported, fir_one
        if fir_supported(x, weight, state):
            return fir_one(x, weight, state, indices)
    st = state[indices].float()
    wf = weight.float()
    y = (st * wf[None, :, :-1]).sum(-1) + wf[None, :, -1] * x.float()
    state[indices] = torch.cat([st[..., 1:], x.float().unsqueeze(-1)], dim=-1).to(state.dtype)
    return y.to(x.dtype)


def _iir_decode_one(x, poles, residues, indices, state):
    if _fused(x):
        from .triton_verifier import iir_supported, iir_one
        if iir_supported(x, poles, residues, state):
            return iir_one(x, poles, residues, state, indices)
    st = state[indices] * poles[None] + x[..., None]
    state[indices] = st
    return (residues[None] * st).sum(-1).to(x.dtype)


def _fir_decode_snapshots(
    x, weight, m, state, max_positions: int | None = None, accepted=None
):
    """Exact packed FIR recurrence with one complete state per verifier slot."""
    if _fused(x):
        from .triton_verifier import fir_supported, fir_snapshots
        if fir_supported(x, weight, state):
            ins, outs, starts, ends, accepted = _decode_tensors(
                m, x.shape[0], max_positions, accepted
            )
            return fir_snapshots(x, weight, state, (
                _cache_safe_state_indices(ins, state), _cache_safe_state_indices(outs, state),
                starts, ends, accepted))
    out = torch.zeros_like(x)
    wf = weight.float()
    input_slots, output_slots, starts, ends, accepted = _decode_tensors(
        m, x.shape[0], max_positions, accepted
    )
    input_slots = _cache_safe_state_indices(input_slots, state)
    output_slots = _cache_safe_state_indices(output_slots, state)
    source = input_slots.gather(1, (accepted - 1)[:, None]).squeeze(1)
    # Preserve the normalised cache's physical reduction orientation. On the
    # current backend it is a (H, K) view over native contiguous (K, H), while
    # dim-first backends may provide contiguous (H, K) directly.
    native_kh = state.stride(-2) == 1
    if native_kh:
        st = state.transpose(-1, -2)[source].float().contiguous()
    else:
        st = state[source].float().contiguous()
    lengths = ends - starts
    for position in range(output_slots.shape[1]):
        active = position < lengths
        token_indices = (starts + position).clamp_max(x.shape[0] - 1)
        xt = x[token_indices].float()
        reduction_state = st.transpose(-1, -2) if native_kh else st
        yt = (
            (reduction_state * wf[None, :, :-1]).sum(-1)
            + wf[None, :, -1] * xt
        )
        if native_kh:
            next_st = torch.cat((st[:, 1:, :], xt[:, None, :]), dim=1)
        else:
            next_st = torch.cat((st[..., 1:], xt[..., None]), dim=-1)
        st = torch.where(active[:, None, None], next_st, st)
        destinations = torch.where(active, output_slots[:, position], 0).long()
        write_state = st.transpose(-1, -2) if native_kh else st
        state.index_copy_(
            0, destinations, write_state.to(state.dtype)
        )
        out.index_add_(0, token_indices, yt.to(out.dtype) * active[:, None])
    return out


def _iir_decode_snapshots(
    x, poles, residues, m, state, max_positions: int | None = None, accepted=None
):
    """Exact packed diagonal recurrence with a state snapshot after each token."""
    if _fused(x):
        from .triton_verifier import iir_supported, iir_snapshots
        if iir_supported(x, poles, residues, state):
            ins, outs, starts, ends, accepted = _decode_tensors(
                m, x.shape[0], max_positions, accepted
            )
            return iir_snapshots(x, poles, residues, state, (
                _cache_safe_state_indices(ins, state), _cache_safe_state_indices(outs, state),
                starts, ends, accepted))
    out = torch.zeros_like(x)
    input_slots, output_slots, starts, ends, accepted = _decode_tensors(
        m, x.shape[0], max_positions, accepted
    )
    input_slots = _cache_safe_state_indices(input_slots, state)
    output_slots = _cache_safe_state_indices(output_slots, state)
    source = input_slots.gather(1, (accepted - 1)[:, None]).squeeze(1)
    st = state[source].float()
    lengths = ends - starts
    for position in range(output_slots.shape[1]):
        active = position < lengths
        token_indices = (starts + position).clamp_max(x.shape[0] - 1)
        xt = x[token_indices].float()
        next_st = st * poles[None] + xt[..., None]
        st = torch.where(active[:, None, None], next_st, st)
        destinations = torch.where(active, output_slots[:, position], 0).long()
        state.index_copy_(0, destinations, st)
        yt = (residues[None] * st).sum(-1)
        out.index_add_(0, token_indices, yt.to(out.dtype) * active[:, None])
    return out


def _direct_temporal_state(cache, hidden: int, state_size: int, stamp_columns: int = 0):
    """Validate a temporal cache declared directly as ``(H, S + stamp_columns)``.

    Returns the ``(H, S)`` state view; the optional trailing column holds the
    position stamp under speculative decoding and is not part of the recurrence.
    """
    expected = (hidden, state_size + stamp_columns)
    if tuple(cache.shape[-2:]) != expected:
        raise RuntimeError(
            "temporal state cache has wrong trailing shape: "
            f"expected {expected}, got {tuple(cache.shape[-2:])}"
        )
    return cache[..., :state_size] if stamp_columns else cache


class _ProjectionTap(nn.Module):
    """Parameter-free stand-in for in_proj in FP8-surgery layers whose projection
    is a token table or a fully emulated GEMM. Returns (z, None) like
    ReplicatedLinear, so a forward hook on mixer.in_proj (the embeddings API's
    blocks.N.projections) sees exactly the projection output."""

    def forward(self, z):
        return z, None


class _HybridProjection(ReplicatedLinear):
    """bf16 in_proj whose calibrated worst output channels are replaced by the
    emulated FP8 GEMM *inside* the module, so hooks see the final projection.
    The emulated tensors stay on the mixer (checkpoint names unchanged);
    `emulate` is attached by the mixer after construction."""

    emulate = None

    def forward(self, x):
        z, bias = super().forward(x)
        cols, emul = self.emulate(x)
        return z.index_copy(-1, cols, emul), bias


def _emulated_fp8_linear(x, w_t, scales):
    from .fp8_emul_triton import static_fp8_linear_emulated_triton

    return static_fp8_linear_emulated_triton(x, w_t, scales[0], scales[1])


class HyenaMixerBase(MambaBase, nn.Module):
    """Shared front half: in_proj -> stateful short FIR -> interleave -> split."""

    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None,
        cache_config: CacheConfig | None,
        prefix: str,
        quant_config=None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.model_config = model_config
        self.cache_config = cache_config

        h = config.hidden_size
        self.hidden = h
        self.k_short = config.short_filter_length
        spec_config = get_current_vllm_config().speculative_config
        self.num_spec = (
            spec_config.num_speculative_tokens if spec_config is not None else 0
        )

        # Only the projections are quantisable; the depthwise filters, poles and
        # residues stay full precision (they are tiny and numerically delicate).
        in_proj_quant_config = quant_config
        if config.use_fp8_input_projections:
            # This is deliberately local rather than the model's global
            # quant_config.  The released Evo checkpoints use Transformer Engine
            # E4M3 only for Hyena input projections; applying global vLLM FP8
            # would also quantize attention, MLPs, output projections and KV
            # cache, changing both numerics and the model being evaluated.
            marker = config.hyena_in_proj_fp8
            if not isinstance(marker, dict) or marker.get("runtime") != (
                "vllm_fp8_static_w8a8_v1"
            ):
                raise ValueError(
                    f"{prefix}: required selective Hyena FP8 metadata is missing"
                )
            in_proj_quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="static",
            )
        self.in_proj_token_table = None
        self.in_proj_emul_weight_t = None
        self.in_proj_emul_cols = None
        if layer_idx in getattr(config, "hyena_token_table_layers", ()):
            # FP8 surgery: layer 0's projection is a per-token lookup of the
            # released model's FP8 fast-accumulation output (config.py).
            self.in_proj = _ProjectionTap()
            self.in_proj_token_table = nn.Parameter(
                torch.empty(config.vocab_size, 3 * h), requires_grad=False
            )
        elif layer_idx in getattr(config, "hyena_fp8_emulated_layers", ()):
            # FP8 surgery: E4M3 operands (held exactly in bf16, K-major) through a
            # software model of the Hopper accumulator; no FP8 hardware needed.
            self.in_proj = _ProjectionTap()
            self.in_proj_emul_weight_t = nn.Parameter(
                torch.empty(h, 3 * h, dtype=torch.bfloat16), requires_grad=False
            )
            self.in_proj_emul_scales = nn.Parameter(
                torch.empty(2, dtype=torch.float32), requires_grad=False
            )
        else:
            n_emul = getattr(config, "hyena_fp8_emulated_channels", {}).get(layer_idx, 0)
            self.in_proj = (_HybridProjection if n_emul else ReplicatedLinear)(
                h, 3 * h, bias=False, prefix=f"{prefix}.in_proj",
                quant_config=in_proj_quant_config,
            )
            if n_emul:
                # FP8 surgery, hybrid: the calibrated worst channels of this bf16
                # in_proj are recomputed by the emulated FP8 GEMM.
                self.in_proj_emul_weight_t = nn.Parameter(
                    torch.empty(h, n_emul, dtype=torch.bfloat16), requires_grad=False
                )
                self.in_proj_emul_scales = nn.Parameter(
                    torch.empty(2, dtype=torch.float32), requires_grad=False
                )
                self.in_proj_emul_cols = nn.Parameter(
                    torch.empty(n_emul, dtype=torch.int64), requires_grad=False
                )
                self.in_proj.emulate = lambda x: (
                    self.in_proj_emul_cols,
                    _emulated_fp8_linear(x, self.in_proj_emul_weight_t,
                                         self.in_proj_emul_scales),
                )
        self.out_proj = ReplicatedLinear(
            h, h, bias=True, prefix=f"{prefix}.out_proj", quant_config=quant_config,
        )
        # depthwise, stored (3H, 1, K) like the checkpoint
        self.short_filter = nn.Parameter(torch.empty(3 * h, 1, self.k_short))

        comp = get_current_vllm_config().compilation_config
        if prefix in comp.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        comp.static_forward_context[prefix] = self
        # The mixer runs as an opaque custom op that piecewise compilation splits
        # around, like vLLM's own mamba_mixer2 / GDN ops. Without the split,
        # torch.compile traces into the metadata-dependent Python of the mixers,
        # graph-breaks, and every non-uniform step (speculative verify with
        # unequal draft counts, prefill mixed with decode) runs as thousands of
        # eagerly launched kernels: 7B batch-8 verify cost 11x a decode step.
        # splitting_ops is read when the model is compiled, after construction,
        # and is part of the compile-cache hash.
        if comp.splitting_ops is not None and HYENA_MIXER_OP not in comp.splitting_ops:
            comp.splitting_ops.append(HYENA_MIXER_OP)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def forward(self, hidden_states: torch.Tensor, token_ids=None) -> torch.Tensor:
        # token_ids: only for a layer served from an fp8-surgery token table
        # (hyena_token_table_layers); None everywhere else.
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.evo2_hyena_mixer(hidden_states, output, token_ids,
                                        _encode_layer_name(self.prefix))
        return output

    # ---- MambaBase plumbing -------------------------------------------------
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        # Evo only consumes the common metadata fields, but it must advertise an
        # accepted-token-aware builder. In vLLM 0.26 gpu_model_runner only passes
        # num_accepted_tokens to Mamba2/GDN/linear-attention builders; SHORT_CONV
        # consequently classifies every multi-token verification as prefill and
        # makes rollback impossible. MAMBA2 adds the required decode metadata;
        # it does not select Evo's math kernels (forward remains implemented here).
        return MambaAttentionBackendEnum.MAMBA2

    def get_attn_backend(self):
        # With speculation, a Mamba2 builder subclass adds per-request computed-
        # token counts, which position-stamp rollback needs on every step. The
        # ordinary path keeps vLLM's stock Mamba2 backend.
        if self.num_spec:
            from .attention import HyenaMamba2Backend

            return HyenaMamba2Backend
        return super().get_attn_backend()

    def _meta(self):
        raw = get_forward_context().attn_metadata
        if raw is None:
            return None
        return raw[self.prefix] if isinstance(raw, dict) else raw

    @staticmethod
    def conv_shape(dim: int, kernel: int) -> tuple[int, ...]:
        """Conv-state shape in whatever orientation this platform expects.

        Do NOT hardcode (dim, kernel-1): vLLM stores conv state as either
        (dim, state_len) or its transpose depending on get_conv_state_layout(),
        and the declared spec has to agree with the layout the kernels index.
        """
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1, intermediate_size=dim, conv_kernel=kernel
        )[0]

    # ---- the one spec every mixer declares -----------------------------------
    # vLLM requires *identical* MambaSpecs across all mamba layers as soon as
    # prefix caching is on: `v1/worker/mamba_utils.py:441` asserts
    # `all(mamba_specs[0] == spec for spec in mamba_specs)` when it builds the
    # state-copy buffers. Evo 2 has three native layouts (SE 6 taps, MR 127 taps,
    # LI 16 fp32 channels), which used to trip that assert.
    #
    # So all three declare one shape and dtype tuple. FIR layers use the leading
    # columns of the temporal second slice:
    #     SE  -> [..., :k_inner-1]  (6)
    #     MR  -> [..., :k_inner-1]  (127, i.e. all of it)
    # LI uses the third temporal slice for its diagonal SSM state. Both temporal
    # slices are fp32: LI states reach ~1e12 against residues ~1, and one common
    # spec is required across layers.
    #
    # Cost at 7B: 61.6 MiB per sequence (54.8 before LI got its own slice).
    # Still small against 80 KiB/token of KV cache.
    #
    # With speculative decoding the third slice gains one trailing column whose
    # row 0 is the block's position stamp (see the module docstring). It is
    # added only when num_spec > 0, so the ordinary layout, and the KV capacity
    # vLLM derives from it, are unchanged.
    @classmethod
    def unified_state_shape(
        cls, hidden, k_short, k_inner_max, state_size, num_spec=0
    ):
        return (
            cls.conv_shape(3 * hidden, k_short + num_spec),
            cls.conv_shape(hidden, k_inner_max),
            (hidden, state_size + (1 if num_spec else 0)),
        )

    @staticmethod
    def unified_state_dtype(model_dtype):
        return (model_dtype, torch.float32, torch.float32)

    def get_state_shape(self):
        cfg = self.config
        return self.unified_state_shape(
            cfg.hidden_size,
            cfg.short_filter_length,
            cfg.hcm_filter_length,
            cfg.state_size,
            self.num_spec,
        )

    def get_state_dtype(self):
        return self.unified_state_dtype(self.model_config.dtype)

    @staticmethod
    def decode_state_indices(m) -> torch.Tensor:
        """state_indices_tensor_d is (num_decodes, 1 + num_spec_tokens).

        The prefill counterpart is already squeezed to 1-D by the builder, but
        the decode one is not; indexing a state tensor with it directly yields
        an extra dimension. Column 0 is the current state block (no spec decode).
        """
        idx = m.state_indices_tensor_d
        return idx[:, 0] if idx.dim() > 1 else idx

    @staticmethod
    def is_speculative_decode(m) -> bool:
        """Whether decode metadata carries speculative state snapshots."""
        idx = m.state_indices_tensor_d
        return (
            (idx.dim() > 1 and idx.shape[1] > 1)
            or getattr(m, "num_accepted_tokens", None) is not None
            or int(m.num_decode_tokens) != int(m.num_decodes)
        )

    def _conv_state(self, idx: int, width: int | None = None) -> torch.Tensor:
        """Normalised state slice -> (num_blocks, dim, state_len), dim-first.

        `width` takes the leading `width` columns, which is how each mixer claims
        its part of the shared second slice (see `unified_state_shape`). The
        result is a view, so writing through it lands in the real paged buffer.
        """
        s = self.kv_cache[idx]
        s = s if is_conv_state_dim_first() else s.transpose(-1, -2)
        return s if width is None else s[..., :width]

    # ---- speculative rollback from position stamps ---------------------------
    def _position_stamps(self) -> torch.Tensor:
        """(num_blocks,) fp32 view: tokens represented by each block's snapshot.

        fp32 is exact for integer positions below 2**24 (16.7M tokens).
        """
        return self.kv_cache[2][:, 0, -1]

    def _spec_accepted(self, m) -> torch.Tensor:
        """(num_decodes,) accepted count of the previous step, from stamps.

        After a verify step that started at n tokens, block column p holds the
        snapshot for n + p + 1 tokens. The next step's computed-token count n'
        equals the stamp of the live column, so it sits at offset n' minus the
        stamp of column 0. Prefill and single-token steps leave column 0 stamped
        with the current count, giving offset 0. Mismatches are asserted on the
        device (no host sync) instead of being clamped silently; null block 0
        is exempt because padded rows point at it.
        """
        nd = int(m.num_decodes)
        computed = getattr(m, "num_computed_tokens_d", None)
        if computed is None:
            raise RuntimeError(
                "speculative Hyena decode needs num_computed_tokens_d; it is "
                "supplied by evo2_flash.attention.HyenaMamba2Backend"
            )
        slots = m.state_indices_tensor_d[:nd]
        if slots.dim() == 1:
            slots = slots[:, None]
        first = _cache_safe_state_indices(slots, self.kv_cache[2])[:, 0].long()
        offset = (
            computed[:nd].to(torch.int64)
            - self._position_stamps()[first].to(torch.int64)
        )
        torch._assert_async(
            ((first == 0) | ((offset >= 0) & (offset <= self.num_spec))).all(),
            "Evo 2 speculative state stamp does not match the request position",
        )
        if _CHECK_VLLM_COUNTS and getattr(m, "num_accepted_tokens", None) is not None:
            # Test-only cross-check (docs/HYBRID_SPEC_DECODING.md): where vLLM does
            # send counts they must agree with the stamps. Upstream fixes to the
            # draft-free-step bug change when counts are sent; this makes any
            # disagreement loud instead of silently trusting either source.
            vllm_counts = _accepted_tokens(m, nd, offset.device)
            torch._assert_async(
                ((first == 0) | (vllm_counts == offset + 1)).all(),
                "Evo 2 position stamps disagree with vLLM num_accepted_tokens",
            )
        return offset.clamp(0, self.num_spec) + 1

    def _repair_resumed_prefill_state(self, m) -> None:
        """Undo a wrong pre-step state copy for prefill rows (prefix caching).

        vLLM 0.26's align-mode `preprocess_mamba` copies each request's state
        into its current block before the step, taking the snapshot at offset
        `num_accepted_tokens_cpu - 1` of the source block. For a request resumed
        after preemption that count can be stale (left over from the speculative
        step before it was preempted), so the copy takes a block that is not the
        cached state -- measured: offsets 1-2, source stamps 2576 / 2650 / 1
        where 1760 / 1440 / 320 / 480 were due (bench/diag_spec_preempt.py,
        MTP_DRAFTER.md). The resumed prefill then continues from garbage.

        Every snapshot carries its position stamp, so the defect is detectable:
        a prefill row continuing from n computed tokens must start from a block
        stamped n. Where it is not, and the block of the last computed token
        (the copy's intended source) is stamped n, copy that block -- all three
        slices, i.e. the copy vLLM meant to make. Rows that are fine copy onto
        themselves; no host synchronisation. Decode rows are not touched: their
        stamps are already checked (``_spec_accepted`` asserts).
        """
        if not self.num_spec or not int(m.num_prefills):
            return
        computed = getattr(m, "computed_before_p", None)
        src = getattr(m, "state_src_block_p", None)
        has_init = getattr(m, "has_initial_states_p", None)
        if computed is None or src is None or has_init is None:
            return
        npf = int(m.num_prefills)
        dst = m.state_indices_tensor_p[:npf].long()
        src = src[:npf].long()
        computed = computed[:npf].to(torch.float32)
        stamps = self._position_stamps()
        bad = (has_init[:npf].bool() & (stamps[dst] != computed)
               & (stamps[src] == computed) & (src != 0) & (dst != 0))
        take = torch.where(bad, src, dst)
        for state in self.kv_cache:
            state[dst] = state[take]

    def _resolve_accepted(self, m):
        """Stamp-derived accepted counts for a speculative decode step, else None.

        Must run before this layer writes any state in the current step.
        """
        if self.num_spec and int(m.num_decodes) and self.is_speculative_decode(m):
            return self._spec_accepted(m)
        return None

    def _record_stamps(self, m) -> None:
        """Stamp every snapshot written this step (decode rows and prefills)."""
        if not self.num_spec:
            return
        stamps = self._position_stamps()
        nd = int(m.num_decodes)
        if nd and self.is_speculative_decode(m):
            ones = torch.ones(nd, device=stamps.device, dtype=torch.long)
            _, outs, starts, ends, _ = _decode_tensors(
                m, int(m.num_decode_tokens), 1 + self.num_spec, ones
            )
            outs = _cache_safe_state_indices(outs, self.kv_cache[2])
            steps = torch.arange(outs.shape[1], device=outs.device)
            active = steps[None, :] < (ends - starts)[:, None]
            computed = m.num_computed_tokens_d[:nd].to(torch.int64)
            values = computed[:, None] + steps[None, :] + 1
            destinations = torch.where(active, outs.long(), 0).reshape(-1)
            # Inactive rows all target null block 0; its value is irrelevant.
            stamps.index_put_(
                (destinations,),
                torch.where(active, values, 0).reshape(-1).to(stamps.dtype),
            )
        npf = int(m.num_prefills)
        if npf:
            stamps[m.state_indices_tensor_p[:npf].long()] = (
                m.seq_lens[nd : nd + npf].to(stamps.dtype)
            )

    def _project(self, hidden_states: torch.Tensor, token_ids=None) -> torch.Tensor:
        """in_proj; for FP8-surgery layers the token table or the emulated GEMM,
        always returned through self.in_proj so module hooks see the projection."""
        if self.in_proj_token_table is not None:
            if token_ids is None:
                raise ValueError(
                    f"{self.prefix}: token-table projection needs input_ids "
                    "(inputs_embeds are unsupported for FP8-surgery checkpoints)"
                )
            z, _ = self.in_proj(self.in_proj_token_table[token_ids[: hidden_states.shape[0]]])
            return z
        if isinstance(self.in_proj, _ProjectionTap):  # fully emulated layer
            z, _ = self.in_proj(_emulated_fp8_linear(
                hidden_states, self.in_proj_emul_weight_t, self.in_proj_emul_scales))
            return z
        z, _ = self.in_proj(hidden_states)  # bf16 / FP8 / hybrid
        return z

    def _front_half(self, hidden_states: torch.Tensor, m, accepted=None,
                    token_ids=None):
        """-> (x2, x1, v), each (T, H), decode rows first then prefill rows."""
        return self._short_conv(self._project(hidden_states, token_ids), m, accepted)

    def _fused_decode(self, z_d, m, inner, inner_state, D=None, log_poles=None):
        """x2 * y for the ordinary decode rows in one launch, or None.

        `triton_decode.hyena_decode` is bit-identical to the unfused path below
        (vLLM's conv update, `_fir_decode_one` / `_iir_decode_one`, torch
        elementwise). Speculative steps keep that path: their snapshots and
        widened short window are `triton_verifier`'s. `EVO2_UNFUSED_DECODE=1`
        (or `EVO2_TORCH_DECODE=1`) selects the unfused path for diagnosis.
        """
        if (not z_d.shape[0] or not z_d.is_cuda or self.is_speculative_decode(m)
                or os.environ.get("EVO2_UNFUSED_DECODE") == "1"
                or os.environ.get("EVO2_TORCH_DECODE") == "1"):
            return None
        from .triton_decode import decode_supported, hyena_decode

        w = self.short_filter.view(3 * self.hidden, self.k_short)
        conv_state = self._conv_state(0)
        if not decode_supported(z_d, w, conv_state, inner, inner_state, D, log_poles):
            return None
        return hyena_decode(z_d, w, conv_state, self.decode_state_indices(m), inner,
                            inner_state, D, log_poles=log_poles)

    def _short_conv(self, z: torch.Tensor, m, accepted=None, decode=True):
        """in_proj output -> (x2, x1, v) through the K=3 filter.

        Decode rows first, then prefill rows; `decode=False` returns the prefill
        rows only (the decode rows went through `_fused_decode`).
        """
        w = self.short_filter.view(3 * self.hidden, self.k_short)
        conv_state = self._conv_state(0)

        nd, npt = m.num_decode_tokens, m.num_prefill_tokens
        z_d, z_p = torch.split(z[: nd + npt], [nd, npt], dim=0)

        parts = []
        if npt:
            if os.environ.get("EVO2_TORCH_SHORT_FIR") == "1":
                # Diagnostic path: same math in torch, bypassing the vLLM kernel.
                parts.append(self._short_fir_torch(z_p, w, m, conv_state))
            else:
                parts.append(
                    causal_conv1d_fn(
                        z_p.transpose(0, 1),
                        w,
                        None,
                        activation=None,
                        conv_states=conv_state,
                        has_initial_state=m.has_initial_states_p,
                        cache_indices=m.state_indices_tensor_p,
                        metadata=m,
                        query_start_loc=m.query_start_loc_p,
                    ).transpose(0, 1)[:npt]
                )
        if nd and decode:
            if self.is_speculative_decode(m):
                state_indices = _cache_safe_state_indices(
                    m.state_indices_tensor_d, conv_state
                )
                num_decodes = int(m.num_decodes)
                scheduled_pointer = getattr(
                    m, "block_idx_last_scheduled_token", None
                )
                computed_pointer = getattr(m, "block_idx_last_computed_token", None)
                if scheduled_pointer is not None:
                    scheduled_pointer = scheduled_pointer[:num_decodes]
                if computed_pointer is not None:
                    computed_pointer = computed_pointer[:num_decodes]
                # A speculative-capable runner executes an ordinary one-token
                # step when no request has drafts. Mamba metadata then keeps the
                # speculative slot matrix but carries no accepted-count tensor,
                # and 1 is NOT neutral: the previous verify step may have
                # accepted several tokens. forward() passes the count resolved
                # from position stamps; vLLM's counts are only a fallback for
                # direct callers of this helper.
                accepted_tokens = (
                    accepted
                    if accepted is not None
                    else _accepted_tokens(m, num_decodes, state_indices.device)
                )
                parts.insert(
                    0,
                    causal_conv1d_update(
                        z_d.contiguous(),
                        conv_state,
                        w,
                        None,
                        activation=None,
                        conv_state_indices=state_indices,
                        block_idx_last_scheduled_token=scheduled_pointer,
                        initial_state_idx=computed_pointer,
                        num_accepted_tokens=accepted_tokens,
                        query_start_loc=m.query_start_loc_d,
                        max_query_len=m.state_indices_tensor_d.shape[1],
                    ),
                )
            else:
                parts.insert(
                    0,
                    causal_conv1d_update(
                        z_d.contiguous(),
                        conv_state,
                        w,
                        None,
                        activation=None,
                        conv_state_indices=m.state_indices_tensor_d,
                    ),
                )
        zf = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        # interleave() + split -> (x2, x1, v); see evo2_layers/ops.py on naming
        return zf[:, 0::3], zf[:, 1::3], zf[:, 2::3]

    def _short_fir_torch(self, z_p, w, m, state):
        """Torch equivalent of the short-filter causal_conv1d_fn prefill."""
        return _depthwise_prefill_torch(z_p, w, m, state, w.shape[0], self.k_short)

    @staticmethod
    def _unpad(out, hidden_states, n_actual):
        """Return a tensor matching hidden_states' row count.

        With CUDA graphs, vLLM pads the token count up to a captured bucket
        (e.g. 317 -> 320), but the mixers only compute `num_decode_tokens +
        num_prefill_tokens` real rows. Returning the short tensor breaks the
        residual add in the decoder layer. vLLM's own ShortConv sidesteps this by
        writing into a preallocated full-size `output`; same idea here.
        """
        if out.shape[0] == hidden_states.shape[0]:
            return out
        full = torch.zeros_like(hidden_states)
        full[:n_actual] = out
        return full

    def _profile_run(self, hidden_states, token_ids=None):
        """attn_metadata is None during vLLM's memory-profiling pass."""
        z = self._project(hidden_states, token_ids)
        y = z[:, 0::3] * z[:, 1::3] * z[:, 2::3]
        out, _ = self.out_proj(y)
        return out


class HyenaFIRMixer(HyenaMixerBase):
    """Hyena-SE (K=7) and Hyena-MR (K=128): explicit depthwise causal conv."""

    def __init__(self, config, layer_idx, model_config, cache_config, prefix, k_inner,
                 quant_config=None):
        super().__init__(config, layer_idx, model_config, cache_config, prefix,
                         quant_config)
        h = config.hidden_size
        self.k_inner = k_inner
        self.conv = nn.Parameter(torch.empty(h, 1, k_inner))
        # MR carries a gated bias D; SE does not.
        self.has_D = k_inner >= 128
        self.D = nn.Parameter(torch.empty(h)) if self.has_D else None

    # vLLM's causal_conv1d Triton kernels enumerate col0..col3 per KERNEL_WIDTH
    # and only support widths 2-5; Evo 2's inner filters are K=7 (SE) and K=128
    # (MR), which fail to compile with NameError('col0 is not defined'). The
    # short filter (K=3) still uses the fast kernel; the inner filter uses the
    # width-generic varlen kernel in triton_prefill.fir_prefill, with the
    # per-sequence torch loop kept for EVO2_TORCH_INNER_FIR=1 and CPU.
    def _inner_prefill(self, u_p, w, m, state):
        prefill = (_depthwise_prefill_triton
                   if _prefill_kernel(u_p, "EVO2_TORCH_INNER_FIR")
                   else _depthwise_prefill_torch)
        return prefill(u_p, w, m, state, self.hidden, int(w.shape[-1]))

    def _inner_decode(self, u_d, w, m, state, accepted=None):
        if self.is_speculative_decode(m):
            return _fir_decode_snapshots(
                u_d,
                w,
                m,
                state,
                max_positions=1 + self.num_spec,
                accepted=accepted,
            )
        idx = self.decode_state_indices(m)
        return _fir_decode_one(u_d, w, idx, state)

    def _forward_impl(self, hidden_states: torch.Tensor, token_ids=None) -> torch.Tensor:
        m = self._meta()
        if m is None:
            return self._profile_run(hidden_states, token_ids)

        self._repair_resumed_prefill_state(m)
        accepted = self._resolve_accepted(m)  # before any state is written
        w = self.conv.view(self.hidden, self.k_inner)
        inner_state = self._conv_state(1, self.k_inner - 1)
        nd, npt = m.num_decode_tokens, m.num_prefill_tokens
        z = self._project(hidden_states, token_ids)
        g = self._fused_decode(z[:nd], m, w, inner_state, self.D)

        if g is None or npt:
            # Rows the fused step did not take: all of them, or the prefill rows.
            x2, x1, v = self._short_conv(z, m, accepted, decode=g is None)
            u = x1 * v
            parts = []
            if npt:
                parts.append(self._inner_prefill(u[u.shape[0] - npt:], w, m, inner_state))
            if nd and g is None:
                parts.insert(0, self._inner_decode(u[:nd], w, m, inner_state, accepted))
            y = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            if self.has_D:
                y = y + self.D.to(y.dtype) * u
            g = x2 * y if g is None else torch.cat([g, x2 * y], dim=0)
        self._record_stamps(m)
        out, _ = self.out_proj(g)
        return self._unpad(out, hidden_states, nd + npt)


class HyenaIIRMixer(HyenaMixerBase):
    """Hyena-LI: diagonal LTI SSM, s_t = p*s_{t-1} + x_t.

    Decode is a batched 3-line recurrence. Prefill materialises the modal filter
    h[d,t] = sum_s residues[d,s]*exp(log_poles[d,s]*t) and FFT-convolves per
    sequence, then seeds the cached state with the closed form
    s = sum_t p^(L-1-t) x_t (a single einsum, not an O(L) scan).
    """

    def __init__(self, config, layer_idx, model_config, cache_config, prefix,
                 quant_config=None):
        super().__init__(config, layer_idx, model_config, cache_config, prefix,
                         quant_config)
        h, s = config.hidden_size, config.state_size
        self.state_size = s
        self.log_poles = nn.Parameter(torch.empty(h, s, 1, dtype=torch.float32))
        self.residues = nn.Parameter(torch.empty(h, s, dtype=torch.float32))
        self.D = nn.Parameter(torch.empty(h))

    def _forward_impl(self, hidden_states: torch.Tensor, token_ids=None) -> torch.Tensor:
        m = self._meta()
        if m is None:
            return self._profile_run(hidden_states, token_ids)

        self._repair_resumed_prefill_state(m)
        accepted = self._resolve_accepted(m)  # before any state is written
        # Unlike the first two conv-shaped slices, this temporal slice is
        # declared directly as (H, S) and must not be transposed under the
        # default SD convolution layout. This distinction also matters during
        # CUDA-graph profiling, where the dummy cache has exactly the declared
        # shape. Under speculation it carries one trailing stamp column.
        ssm = _direct_temporal_state(
            self.kv_cache[2], self.hidden, self.state_size,
            1 if self.num_spec else 0,
        )  # (num_blocks, H, S) fp32 snapshots
        lp = self.log_poles[..., 0].float()  # (H, S)
        res = self.residues.float()

        nd, npt = m.num_decode_tokens, m.num_prefill_tokens
        z = self._project(hidden_states, token_ids)
        g = self._fused_decode(z[:nd], m, res, ssm, self.D, log_poles=lp)
        if g is not None and not npt:
            self._record_stamps(m)
            out, _ = self.out_proj(g)
            return self._unpad(out, hidden_states, nd)

        # Rows the fused step did not take: all of them, or the prefill rows.
        x2, x1, v = self._short_conv(z, m, accepted, decode=g is None)
        x1v = x1 * v
        y = torch.empty_like(x1v)
        first = nd if g is None else 0   # row of the first prefill token

        if nd and g is None:
            poles = lp.exp()
            xd = x1v[:nd].float()
            if self.is_speculative_decode(m):
                y[:nd] = _iir_decode_snapshots(
                    xd,
                    poles,
                    res,
                    m,
                    ssm,
                    max_positions=1 + self.num_spec,
                    accepted=accepted,
                )
            else:
                idx = self.decode_state_indices(m)
                y[:nd] = _iir_decode_one(xd, poles, res, idx, ssm).to(y.dtype)


        if npt:
            qsl = _prefill_segments(m)[0]
            pidx = m.state_indices_tensor_p
            force_torch = os.environ.get("EVO2_TORCH_FILTER") == "1"
            for j in range(m.num_prefills):
                a, b = int(qsl[j]), int(qsl[j + 1])
                if b <= a:
                    continue
                seq = x1v[first + a : first + b].float().transpose(0, 1)  # (H, L)
                # None for a fresh sequence; the cached state when chunked prefill
                # splits this prompt and this segment continues an earlier one.
                s_prev = (
                    ssm[pidx[j]].float() if _segment_initial_state(m, j) else None
                )
                conv, st = iir_segment(seq, res, lp, s_prev, force_torch=force_torch)
                y[first + a : first + b] = conv.transpose(0, 1).to(y.dtype)
                ssm[pidx[j]] = st

        self._record_stamps(m)
        y = (y + x1v * self.D.to(y.dtype)) * x2
        if g is not None:
            y = torch.cat([g, y], dim=0)
        out, _ = self.out_proj(y)
        return self._unpad(out, hidden_states, nd + npt)


HYENA_MIXER_OP = "vllm::evo2_hyena_mixer"


def evo2_hyena_mixer(hidden_states: torch.Tensor, output: torch.Tensor,
                     token_ids: torch.Tensor | None, layer_name: LayerNameType) -> None:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    output.copy_(layer._forward_impl(hidden_states, token_ids))


def evo2_hyena_mixer_fake(hidden_states: torch.Tensor, output: torch.Tensor,
                          token_ids: torch.Tensor | None, layer_name: LayerNameType) -> None:
    return


direct_register_custom_op(
    op_name="evo2_hyena_mixer",
    op_func=evo2_hyena_mixer,
    mutates_args=["output"],
    fake_impl=evo2_hyena_mixer_fake,
)
