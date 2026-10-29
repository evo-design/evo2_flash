# SPDX-License-Identifier: Apache-2.0
"""A host-driven CFG decoder: obviously correct, deliberately slow.

This is the *slow* side of gate G2 (`tests/test_cfg_consistency.py`). It
reproduces `docs/CFG_REFERENCE.md` §1-§4 with no engine machinery in the loop
at all -- no logits processor, no paired requests, no one-hot, no persistent
batch, no row bookkeeping. One token at a time:

    for each step:
        forward the full cond history and the full base history
        take position -1 of each, cast fp32
        guided = l_cond + w * (l_cond - l_base)
        mask to ACGT, if asked
        token = evo2_flash.cfg_sample.sample(guided, ...)
        append that one token to BOTH histories

which is the reference's architecture (two tracks advanced in lockstep by one
shared sampled token), expressed as repeated prefills instead of two KV caches.
It is the same arithmetic in the same order, obtained a completely different
way, which is the only reason it is worth running.

The forward is `Evo2._capture(..., want_logits=True)`: each history is
submitted as a prefill-only request and the `(L, 512)` logits tensor comes back
in position order. Cost is therefore O(n_tokens x prompt_len) rather than
O(n_tokens), so keep G2's prompts short. `_capture` passes
`skip_reading_prefix_cache=True` (it has to: every position must be computed),
so there is no prefix-cache shortcut here either.

### Why the RNG stream is handled the way it is

The engine-side processor holds **one** `torch.Generator` and samples every
resident pair in a single vectorised `sample()` call, with rows ordered by
`pair_id`. A multinomial draw over a `(P, V)` batch consumes the generator
differently from `P` draws over `(1, V)`, so a host-driven reference that
sampled pair by pair could not match a sampled (non-greedy) engine run even
with the same seed. `cfg_generate_reference` therefore stacks all pairs into
one call, in pair order, exactly as the processor does. That still only matches
when every pair is sampled on every step; with `top_k=1` the question does not
arise, which is why G2's many-pair grid is greedy and its sampled case uses a
single pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from evo2_flash.cfg_sample import MASKED_LOGIT, sample


@dataclass
class ReferenceTrack:
    """One CFG pair driven by the host."""

    guidance: float
    cond_ids: list[int]
    base_ids: list[int]
    tokens: list[int] = field(default_factory=list)
    # log_softmax(guided)[token] per step -- the quantity
    # `Evo2CfgLogitsProcessor.guidance_logprob_means()` averages.
    guided_logprobs: list[float] = field(default_factory=list)

    @property
    def guided_logprob_mean(self) -> float:
        return sum(self.guided_logprobs) / len(self.guided_logprobs)


def _blocked_mask(vocab: int, dna_only_ids, device) -> torch.Tensor:
    blocked = torch.ones(vocab, dtype=torch.bool, device=device)
    blocked[list(dna_only_ids)] = False
    return blocked


def cfg_generate_reference(
    model,
    pairs,
    *,
    n_tokens: int,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.9,
    dna_only_ids=None,
    seed: int | None = None,
    verbose: int = 0,
) -> list[ReferenceTrack]:
    """Decode `pairs` with CFG, one token per step, through repeated prefills.

    `model` is a `evo2_flash.Evo2` (any engine: the logits processor is never
    used, so `cfg=True` is not required). `pairs` is a sequence of
    `(guidance, cond_ids, base_ids)` with token ids, ordered as the engine
    orders its `pair_id`s.

    Returns one `ReferenceTrack` per pair, in the same order.
    """
    tracks = [
        ReferenceTrack(guidance=float(w), cond_ids=list(c), base_ids=list(b))
        for w, c, b in pairs
    ]
    if not tracks:
        raise ValueError("no pairs")
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")

    generator = None
    blocked = None

    for step in range(n_tokens):
        histories = []
        for t in tracks:
            histories.append(t.cond_ids + t.tokens)
            histories.append(t.base_ids + t.tokens)

        caps = model._capture(histories, {}, want_logits=True)

        # (P, V) fp32, rows in pair order -- the processor's own layout.
        guided = torch.stack(
            [
                caps[2 * i]["__logits__"][-1].float()
                + t.guidance
                * (
                    caps[2 * i]["__logits__"][-1].float()
                    - caps[2 * i + 1]["__logits__"][-1].float()
                )
                for i, t in enumerate(tracks)
            ]
        )

        if dna_only_ids is not None:
            if blocked is None:
                blocked = _blocked_mask(
                    guided.shape[-1], dna_only_ids, guided.device
                )
            guided = guided.masked_fill(blocked, MASKED_LOGIT)

        if generator is None and top_k != 1:
            generator = torch.Generator(device=guided.device)
            if seed is not None:
                generator.manual_seed(int(seed))

        token = sample(
            guided,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            generator=generator,
        )
        lp = torch.log_softmax(guided, dim=-1).gather(1, token.unsqueeze(1)).squeeze(1)

        for i, t in enumerate(tracks):
            t.tokens.append(int(token[i].item()))
            t.guided_logprobs.append(float(lp[i].item()))

        if verbose:
            print(f"  [reference] step {step + 1}/{n_tokens}: "
                  f"{[t.tokens[-1] for t in tracks]}")

    return tracks
