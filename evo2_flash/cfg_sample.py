# SPDX-License-Identifier: Apache-2.0
"""A line-by-line port of vortex's sampler, for use inside the CFG processor.

Why this exists is worth stating precisely, because the obvious reason is
wrong. vLLM's order (divide the whole vocabulary by temperature, then top-k,
then top-p) and vortex's order (top-k slice first, divide the slice, top-p on
the slice) are **mathematically equivalent**: they keep the same tokens and
renormalise to the same probabilities. `Evo2.sampling_kwargs` already relies on
that, and `tests/test_cfg_processor.py` pins it -- measured max difference
1.2e-7 over 300 random logit sets x 7 (top_k, top_p, temperature) settings,
including DNA-masked inputs.

It exists because the CFG processor has to sample **itself**: it writes a
one-hot into both rows of a pair, which is what makes lockstep structural
(see `evo2_flash/cfg.py`). Given that a sampler is needed anyway, reusing the
reference's exact code is the cheapest way to be certain nothing subtle
diverges. Two details of it are genuinely not vLLM's behaviour and do matter:
the `nan_to_num` / +/-inf prologue, and `top_k == 1` short-circuiting to argmax
*before* temperature is read.

Reference: `vortex/model/sample.py` (vortex 0.6.0, sha256 5ec1f0e509a53062...),
recorded in `docs/CFG_REFERENCE.md` §4. The only intentional deviation is that
`sample()` here accepts an explicit `generator` so a run is reproducible
without touching global RNG state; vortex uses the global generator.
"""

from __future__ import annotations

import torch

# The DNA-only mask and the one-hot both use a large *finite* negative, the
# value the reference uses. Not because `sample()`'s prologue below would
# un-mask an -inf -- it would not, since `nan_to_num` runs first and has
# already mapped -inf to the fp32 minimum, leaving the two `torch.where` lines
# unreachable -- but because a finite sentinel survives arithmetic:
# (-inf) - (-inf) is NaN, and `nan_to_num` then turns that into 0.0, i.e. the
# *most* likely token. The CFG mix is exactly such a difference.
# See docs/CFG_REFERENCE.md §3.
MASKED_LOGIT = -1e30


def modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> None:
    """Set the logits for non-top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return

    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def sample(
    logits: torch.Tensor,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample from top-k logits. `logits` is (batch_size, vocab_size).

    Note `top_k == 1` short-circuits to argmax *before* temperature is read,
    and `top_p == 0.0` (the default) means no top-p filtering at all.
    """
    logits = torch.nan_to_num(logits)
    logits = torch.where(logits == float("-inf"), 0, logits)
    logits = torch.where(logits == float("inf"), 0, logits)

    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)

    if top_p > 0.0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        logits_top, indices = torch.topk(logits, top_k, dim=-1)
        if temperature != 1.0:
            logits_top /= temperature
        modify_logits_for_top_p_filtering(logits_top, top_p)
        choice = torch.multinomial(
            torch.softmax(logits_top, dim=-1), num_samples=1, generator=generator
        ).squeeze(dim=-1)
        return indices[
            torch.arange(indices.shape[0], device=indices.device), choice
        ]

    # Clone so that when we modify for top_p we don't change the original logits
    logits_top = logits / temperature if temperature != 1.0 else logits.clone()
    modify_logits_for_top_p_filtering(logits_top, top_p)
    return torch.multinomial(
        torch.softmax(logits_top, dim=-1), num_samples=1, generator=generator
    ).squeeze(dim=-1)
