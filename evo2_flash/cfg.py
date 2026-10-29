# SPDX-License-Identifier: Apache-2.0
"""Classifier-free guidance for Evo 2 inside vLLM.

A general steering mechanism, deliberately. A CFG run is fully specified by a
conditional prompt, a baseline prompt, a guidance scale, sampling parameters
and an optional token mask -- nothing in this module knows what any of those
mean. Task vocabulary does not belong here: callers compose prompts however
they like, and every application differs only in that composition.

The reference implementation (`docs/CFG_REFERENCE.md`) runs on vortex: two KV
caches advanced in lockstep, two single-token forwards per step, logits mixed,
**one** token sampled and fed back to both caches. vLLM cannot be driven that
way without forking it, and does not need to be: a CFG pair is just two
requests in the same batch.

So a CFG run submits `2N` paired requests -- interleaved
`[cond_0, base_0, cond_1, base_1, ...]` -- and mixes them inside one
batch-level `LogitsProcessor`. This is the only no-fork hook that sees the
whole batch, and because the processor declares itself **not** argmax-invariant
vLLM calls it on raw fp32 logits before penalties and before temperature
(`v1/sample/sampler.py`) -- exactly where the reference applies CFG.

Lockstep is made structural rather than scheduling-dependent. The processor
samples the shared token itself, with vortex's operator order
(`evo2_flash/cfg_sample.py`), then writes a one-hot into **both** rows of the
pair. CFG requests are required to carry `temperature=0`, so vLLM's own
sampler reduces to `argmax` and provably emits that same token for both rows.
Matched seeds could not substitute for this: vLLM keys sampling noise by batch
index, the two rows have different indices and independently advancing
generator state, and under CFG the two rows see different distributions by
construction.

Consequently the real sampling parameters (`temperature`, `top_k`, `top_p`)
travel in the `CfgSpec`, not in `SamplingParams`; the ones in `SamplingParams`
are pinned to neutral values and enforced at admission by `validate_params`.
The processor samples with `evo2_flash.cfg_sample`, the reference's own sampler
-- not because vLLM's operator order differs (it is equivalent; see that
module) but because a processor that writes a one-hot must choose the token
itself, and reusing the reference's code leaves nothing to argue about.

Limitation inherited from vLLM 0.26: custom logits processors are rejected when
speculative decoding is enabled (`STR_SPEC_DEC_REJECTS_LOGITSPROCS`), so CFG
and the trained draft heads cannot be combined.
"""

from __future__ import annotations

import weakref
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from evo2_flash import attention
from evo2_flash.cfg_sample import MASKED_LOGIT, sample


if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Key under which a CfgSpec rides on SamplingParams.extra_args. Request ids are
# not visible to logits processors, so pairing has to travel this way.
EVO2_CFG_KEY = "evo2_cfg"

ROLE_COND = "cond"
ROLE_BASE = "base"
_ROLES = (ROLE_COND, ROLE_BASE)

# How many trailing output tokens the desync check compares. One would do (the
# check runs every step), but a few make a diagnosis legible.
_DESYNC_TAIL = 8


class Evo2CfgDesync(RuntimeError):
    """A CFG pair's two tracks stopped agreeing.

    There is nothing to salvage: a diverged pair no longer evaluates two
    distributions over the same history, so its output is scientifically
    meaningless. Raised rather than repaired.
    """


@dataclass(frozen=True)
class CfgSpec:
    """Everything the processor needs to know about one request.

    `guidance` is the reference's `guidance_scale` (`w`), and `w = 0` recovers
    plain conditional generation exactly -- this is *not* the
    `l_uncond + w*(l_cond - l_uncond)` convention. See CFG_REFERENCE.md §1.
    """

    pair_id: int
    role: str
    guidance: float
    temperature: float = 1.0
    top_k: int = 4
    top_p: float = 0.9
    dna_only_ids: tuple[int, ...] | None = None
    seed: int | None = None
    on_desync: str = "raise"

    def to_extra_args(self) -> dict[str, Any]:
        d = asdict(self)
        if d["dna_only_ids"] is not None:
            d["dna_only_ids"] = list(d["dna_only_ids"])
        return {EVO2_CFG_KEY: d}

    @classmethod
    def from_extra_args(cls, extra_args: dict[str, Any] | None) -> CfgSpec | None:
        if not extra_args or EVO2_CFG_KEY not in extra_args:
            return None
        raw = extra_args[EVO2_CFG_KEY]
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(
                f"{EVO2_CFG_KEY} must be a dict or CfgSpec, got {type(raw).__name__}"
            )
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown {EVO2_CFG_KEY} keys: {sorted(unknown)}")
        raw = dict(raw)
        ids = raw.get("dna_only_ids")
        if ids is not None:
            raw["dna_only_ids"] = tuple(int(i) for i in ids)
        return cls(**raw)

    # The fields that must agree across every pair in one batch, because the
    # processor mixes and samples them in a single vectorised call.
    def shared_key(self) -> tuple:
        return (self.temperature, self.top_k, self.top_p, self.dna_only_ids)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"CFG request: {msg}")


@dataclass
class _Slot:
    spec: CfgSpec
    # A *live* reference to the request's running output token list, per the
    # BatchUpdate contract. Never copy it.
    out: list[int]
    # Prompt length, used to tell a real next-token logits row from the tail of
    # a mid-prompt chunk. None if vLLM did not supply prompt token ids.
    n_prompt: int | None = None


class Evo2CfgLogitsProcessor(LogitsProcessor):
    """Mixes each CFG pair's logits and forces both rows to the same token."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = torch.device(device)
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.vocab_size = vllm_config.model_config.get_vocab_size()

        self._slots: list[_Slot | None] = [None] * self.max_num_reqs
        self._dirty = True
        self._batch_size = 0

        # Rebuilt by _rebuild(), ordered by pair_id so the sampling stream does
        # not depend on how vLLM happens to order rows this step.
        self._pair_ids: list[int] = []
        self._cond_idx: torch.Tensor | None = None
        self._base_idx: torch.Tensor | None = None
        self._all_idx: torch.Tensor | None = None
        self._w: torch.Tensor | None = None
        self._blocked: torch.Tensor | None = None
        self._sampling: tuple[float, int, float] = (1.0, 1, 0.0)
        self._on_desync = "raise"
        self._cond_slots: list[_Slot] = []
        self._base_slots: list[_Slot] = []
        self._row_index: dict[int, int] = {}
        self._widowed: list[tuple[int, str, int]] = []

        self._generator: torch.Generator | None = None
        self._seed: int | None = None

        # Guided-logprob accumulators, keyed by pair_id and kept across
        # removal so a finished pair's record survives. Device scalars; one
        # host sync at read time.
        self._lp_sum: dict[int, torch.Tensor] = {}
        self._lp_n: dict[int, int] = {}
        # Per-epoch accumulator, aligned with self._pair_ids and valid only
        # between rebuilds.
        self._lp_running: torch.Tensor | None = None
        # Steps accumulated per pair, aligned with self._pair_ids.
        self._lp_steps: list[int] = []

        _register(self)

    # ---------------------------------------------------------------- vLLM API

    def is_argmax_invariant(self) -> bool:
        # False, so vLLM runs us on raw fp32 logits before penalties and
        # temperature -- and so the one-hot actually decides the token.
        return False

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        """Enforce the CFG contract at admission.

        Called by `SamplingParams.verify` for *every* request once this
        processor is installed, so it must be a no-op for non-CFG requests.
        """
        spec = CfgSpec.from_extra_args(sampling_params.extra_args)
        if spec is None:
            return

        _require(spec.role in _ROLES, f"role must be one of {_ROLES}, got {spec.role!r}")
        _require(isinstance(spec.pair_id, int), "pair_id must be an int")
        _require(
            spec.guidance == spec.guidance and abs(spec.guidance) != float("inf"),
            "guidance must be finite",
        )
        _require(spec.temperature > 0.0, "CfgSpec.temperature must be > 0")
        _require(spec.top_k >= 1, "CfgSpec.top_k must be >= 1")
        _require(0.0 <= spec.top_p <= 1.0, "CfgSpec.top_p must be in [0, 1]")
        _require(
            spec.on_desync in ("raise", "repair"),
            "on_desync must be 'raise' or 'repair'",
        )
        if spec.dna_only_ids is not None:
            _require(len(spec.dna_only_ids) > 0, "dna_only_ids must not be empty")
            _require(
                len(set(spec.dna_only_ids)) == len(spec.dna_only_ids),
                "dna_only_ids must not repeat",
            )

        # The processor samples; vLLM must not sample again. temperature == 0
        # makes vLLM's sampler argmax, which on our one-hot is the shared token.
        _require(
            sampling_params.temperature == 0.0,
            "SamplingParams.temperature must be 0 (the real temperature goes in "
            "CfgSpec.temperature); vLLM's sampler must reduce to argmax so both "
            "rows of a pair emit the token the processor chose",
        )
        _require(
            sampling_params.top_k in (0, -1, None),
            "SamplingParams.top_k must be unset (use CfgSpec.top_k)",
        )
        _require(
            sampling_params.top_p == 1.0,
            "SamplingParams.top_p must be 1.0 (use CfgSpec.top_p)",
        )
        _require(sampling_params.min_p == 0.0, "SamplingParams.min_p must be 0")
        _require(sampling_params.n == 1, "n must be 1")
        for name in ("presence_penalty", "frequency_penalty"):
            _require(getattr(sampling_params, name) == 0.0, f"{name} must be 0")
        _require(
            sampling_params.repetition_penalty == 1.0, "repetition_penalty must be 1"
        )
        _require(sampling_params.logit_bias is None, "logit_bias is not supported")
        _require(
            not sampling_params.bad_words, "bad_words is not supported (use dna_only)"
        )
        _require(
            sampling_params.allowed_token_ids is None,
            "allowed_token_ids is not supported (use CfgSpec.dna_only_ids)",
        )

        # Both tracks must stop on the same step, or one row is widowed while
        # still being sampled.
        _require(sampling_params.max_tokens is not None, "max_tokens must be set")
        _require(
            sampling_params.min_tokens == sampling_params.max_tokens,
            "min_tokens must equal max_tokens so both tracks of a pair finish "
            "on the same step",
        )
        _require(sampling_params.ignore_eos, "ignore_eos must be True")

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        # Order is fixed by the interface contract: removed, added, moved.
        for index in batch_update.removed:
            if self._slots[index] is not None:
                self._slots[index] = None
                self._dirty = True

        for index, params, prompt_tok_ids, out_tok_ids in batch_update.added:
            spec = CfgSpec.from_extra_args(params.extra_args)
            had = self._slots[index] is not None
            self._slots[index] = (
                _Slot(
                    spec,
                    out_tok_ids,
                    None if prompt_tok_ids is None else len(prompt_tok_ids),
                )
                if spec is not None
                else None
            )
            if had or spec is not None:
                self._dirty = True

        for adx, bdx, direct in batch_update.moved:
            a, b = self._slots[adx], self._slots[bdx]
            self._slots[bdx] = a
            self._slots[adx] = b if direct == MoveDirectionality.SWAP else None
            if a is not None or b is not None:
                self._dirty = True

        self._batch_size = batch_update.batch_size

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._dirty:
            self._rebuild()
            self._dirty = False

        if self._widowed:
            self._handle_widowed()
        if self._cond_idx is None:
            return logits

        # Pairs still inside their prompt are left alone: vLLM takes one logits
        # row per *scheduled* request, so such a row holds the distribution
        # after a mid-prompt chunk rather than a next-token distribution, and
        # the runner discards whatever is sampled from it.
        pairs = self._ready_pairs(logits.shape[0])
        if not pairs:
            return logits
        cond_idx, base_idx, all_idx, w = self._index_tensors(pairs)

        self._check_desync(pairs)

        lc = logits.index_select(0, cond_idx)
        lb = logits.index_select(0, base_idx)

        # The CFG combination, on raw logits. CFG_REFERENCE.md section 1.
        guided = lc + w.unsqueeze(1) * (lc - lb)

        if self._blocked is not None:
            # A large finite negative, never -inf: see MASKED_LOGIT in
            # cfg_sample.py and CFG_REFERENCE.md section 3.
            guided = guided.masked_fill(self._blocked, MASKED_LOGIT)

        temperature, top_k, top_p = self._sampling
        token = sample(
            guided,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            generator=self._generator,
        )

        # Accumulated vectorised into a per-epoch buffer aligned with
        # self._pair_ids; _flush_records() folds it into the per-pair totals
        # whenever the pair set changes. Doing this per pair per step would
        # mean one tiny kernel per pair per step.
        lp = (
            torch.log_softmax(guided, dim=-1)
            .gather(1, token.unsqueeze(1))
            .squeeze(1)
        )
        if self._lp_running is None:
            self._lp_running = torch.zeros(
                len(self._pair_ids), dtype=lp.dtype, device=lp.device
            )
        if len(pairs) == len(self._pair_ids):
            self._lp_running += lp
        else:
            self._lp_running.index_add_(
                0, torch.tensor(pairs, dtype=torch.long, device=lp.device), lp
            )
        for row in pairs:
            self._lp_steps[row] += 1

        # Force both rows of every mixed pair onto the chosen token. With
        # SamplingParams.temperature == 0 vLLM takes the argmax of each row,
        # so the pair advances in lockstep by construction.
        logits.index_fill_(0, all_idx, MASKED_LOGIT)
        logits[all_idx, token.repeat(2)] = 0.0
        return logits

    # ------------------------------------------------------------- internals

    def _flush_records(self) -> None:
        """Fold the current epoch's accumulator into the per-pair totals."""
        if self._lp_running is None:
            return
        for row, pair_id in enumerate(self._pair_ids):
            if not self._lp_steps[row]:
                continue
            prev = self._lp_sum.get(pair_id)
            value = self._lp_running[row]
            self._lp_sum[pair_id] = value if prev is None else prev + value
            self._lp_n[pair_id] = self._lp_n.get(pair_id, 0) + self._lp_steps[row]
        self._lp_running = None
        self._lp_steps = []

    def _rebuild(self) -> None:
        self._flush_records()
        by_pair: dict[int, dict[str, int]] = {}
        # Collected over *every* CFG slot, not only complete pairs: the widowed
        # handler needs it, and a batch may consist of nothing but a widow.
        on_desync: set[str] = set()
        for index, slot in enumerate(self._slots):
            if slot is None:
                continue
            on_desync.add(slot.spec.on_desync)
            if self._batch_size and index >= self._batch_size:
                raise AssertionError(
                    f"CFG slot at index {index} outside batch of {self._batch_size}"
                )
            roles = by_pair.setdefault(slot.spec.pair_id, {})
            if slot.spec.role in roles:
                raise ValueError(
                    f"two {slot.spec.role!r} requests for CFG pair "
                    f"{slot.spec.pair_id} (rows {roles[slot.spec.role]} and {index})"
                )
            roles[slot.spec.role] = index

        pair_ids: list[int] = []
        cond: list[int] = []
        base: list[int] = []
        widowed: list[tuple[int, str, int]] = []
        shared: set[tuple] = set()
        seeds: set[int | None] = set()

        if len(on_desync) > 1:
            raise ValueError(f"conflicting on_desync settings: {sorted(on_desync)}")
        self._on_desync = next(iter(on_desync)) if on_desync else "raise"

        for pair_id in sorted(by_pair):
            roles = by_pair[pair_id]
            if len(roles) != len(_ROLES):
                role, index = next(iter(roles.items()))
                widowed.append((pair_id, role, index))
                continue
            pair_ids.append(pair_id)
            cond.append(roles[ROLE_COND])
            base.append(roles[ROLE_BASE])
            c = self._slots[roles[ROLE_COND]]
            b = self._slots[roles[ROLE_BASE]]
            if c.spec.guidance != b.spec.guidance:
                raise ValueError(
                    f"CFG pair {pair_id}: cond guidance {c.spec.guidance} != "
                    f"base guidance {b.spec.guidance}"
                )
            shared.add(c.spec.shared_key())
            shared.add(b.spec.shared_key())
            seeds.add(c.spec.seed)

        self._widowed = widowed
        self._pair_ids = pair_ids
        self._cond_slots = [self._slots[i] for i in cond]
        self._base_slots = [self._slots[i] for i in base]
        self._row_index = {id(self._slots[i]): i for i in (*cond, *base)}
        self._lp_steps = [0] * len(pair_ids)

        if not pair_ids:
            self._cond_idx = self._base_idx = self._all_idx = self._w = None
            return

        if len(shared) != 1:
            raise ValueError(
                "all CFG requests in one batch must share temperature, top_k, "
                f"top_p and dna_only_ids; saw {len(shared)} distinct settings: "
                f"{sorted(shared, key=repr)}"
            )
        temperature, top_k, top_p, dna_only_ids = next(iter(shared))
        self._sampling = (temperature, top_k, top_p)

        if dna_only_ids is None:
            self._blocked = None
        else:
            blocked = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            blocked[list(dna_only_ids)] = False
            self._blocked = blocked

        self._cond_idx = torch.tensor(cond, dtype=torch.long, device=self.device)
        self._base_idx = torch.tensor(base, dtype=torch.long, device=self.device)
        self._all_idx = torch.cat((self._cond_idx, self._base_idx))
        self._w = torch.tensor(
            [self._slots[i].spec.guidance for i in cond],
            dtype=torch.float32,
            device=self.device,
        )

        seed = next(iter(seeds)) if len(seeds) == 1 else None
        if self._generator is None or seed != self._seed:
            self._seed = seed
            self._generator = torch.Generator(device=self.device)
            if seed is not None:
                self._generator.manual_seed(int(seed))

    def _ready_pairs(self, num_rows: int) -> list[int]:
        """Indices into `self._pair_ids` of the pairs safe to mix this step.

        A pair is ready once **both** tracks are at or past the end of their
        prompt, i.e. this step's logits row really is their next-token
        distribution. A pair that is not ready is skipped, which is correct:
        vLLM discards the token sampled from a mid-prompt row.

        What must never happen is one track finishing its prompt a step before
        the other -- the finished row would then be sampled *unguided* and the
        pair would diverge from its very first token. That is raised, not
        patched over.
        """
        step = attention.STEP_SEQ_LENS
        if step is None:
            # No per-step counts (a vLLM layout change, or a unit harness with
            # no engine). Mix every resident pair: `Evo2.generate` pre-primes
            # both prompts into the prefix cache so a CFG prompt is not chunked
            # in the first place, and gate G2 checks the first token against a
            # host-driven reference.
            return list(range(len(self._pair_ids)))

        num_reqs, seq_lens = step
        # Padded logits (a CUDA-graph bucket) carry the real rows first.
        limit = min(num_rows, num_reqs)

        ready = []
        for row, (pair_id, c, b) in enumerate(
            zip(self._pair_ids, self._cond_slots, self._base_slots)
        ):
            done_c = self._prompt_done(c, seq_lens, limit)
            done_b = self._prompt_done(b, seq_lens, limit)
            if done_c != done_b:
                raise Evo2CfgDesync(
                    f"CFG pair {pair_id}: one track reached the end of its "
                    f"prompt a step before the other (cond={done_c}, "
                    f"base={done_b}), so the finished row would be sampled "
                    "without guidance. Pre-prime both prompts into the prefix "
                    "cache so neither gets split by chunked prefill."
                )
            if done_c:
                ready.append(row)
        return ready

    def _prompt_done(self, slot: _Slot, seq_lens, limit: int) -> bool:
        if slot.n_prompt is None:
            return True  # nothing to compare against; treat as a decode row
        row = self._row_index[id(slot)]
        if row >= limit:
            raise AssertionError(
                f"CFG row {row} outside the {limit} rows vLLM reported this step"
            )
        return int(seq_lens[row]) >= slot.n_prompt

    def _index_tensors(self, pairs: list[int]):
        """Device index tensors for a subset of pairs (cached for the full set)."""
        if len(pairs) == len(self._pair_ids):
            return self._cond_idx, self._base_idx, self._all_idx, self._w
        sel = torch.tensor(pairs, dtype=torch.long, device=self.device)
        cond_idx = self._cond_idx.index_select(0, sel)
        base_idx = self._base_idx.index_select(0, sel)
        return (
            cond_idx,
            base_idx,
            torch.cat((cond_idx, base_idx)),
            self._w.index_select(0, sel),
        )

    def _handle_widowed(self) -> None:
        detail = ", ".join(
            f"pair {p} has only its {r} track (row {i})" for p, r, i in self._widowed
        )
        if self._on_desync == "raise":
            raise Evo2CfgDesync(
                "CFG pair incomplete in the persistent batch, so a widowed row "
                f"would be sampled without guidance: {detail}. Both tracks of a "
                "pair must be resident every step -- pre-prime both prompts so "
                "they prefill in one step, disable chunked prefill, size the KV "
                "cache to avoid preemption, and submit pairs adjacently."
            )
        # "repair" is a diagnostic-only setting: leave widowed rows unguided.

    def _check_desync(self, pairs: list[int]) -> None:
        bad = []
        for row in pairs:
            pair_id = self._pair_ids[row]
            c, b = self._cond_slots[row], self._base_slots[row]
            if len(c.out) != len(b.out):
                bad.append(
                    f"pair {pair_id}: cond has {len(c.out)} output tokens, "
                    f"base has {len(b.out)}"
                )
            elif c.out[-_DESYNC_TAIL:] != b.out[-_DESYNC_TAIL:]:
                bad.append(
                    f"pair {pair_id}: output tails differ, "
                    f"cond {c.out[-_DESYNC_TAIL:]} vs base {b.out[-_DESYNC_TAIL:]}"
                )
        if not bad:
            return
        message = "CFG pair(s) diverged: " + "; ".join(bad)
        if self._on_desync == "raise":
            raise Evo2CfgDesync(message)

    # ----------------------------------------------------------------- records

    def guidance_logprob_means(self) -> dict[int, float]:
        """Mean guided logprob of the sampled token, per pair_id.

        One host sync. These are the logprobs under the *guided* distribution,
        which is not what vLLM reports: with the one-hot in place vLLM's own
        logprobs are taken before the processor runs, i.e. the unguided
        conditional distribution -- which is the number that keeps its
        vortex-parity meaning.
        """
        self._flush_records()
        return {
            pair_id: float(total.item()) / self._lp_n[pair_id]
            for pair_id, total in self._lp_sum.items()
            if self._lp_n.get(pair_id)
        }

    def reset_run(self) -> None:
        """Start an API generation call, restarting its explicit seed stream.

        Rebuilds within a call (batch moves, completed pairs, admission waves)
        keep the stream. A separate generate(seed=...) call must start again.
        """
        self.reset_records()
        self._generator = None
        self._seed = None
        self._dirty = True

    def reset_records(self) -> None:
        self._lp_sum.clear()
        self._lp_n.clear()
        self._lp_running = None
        self._lp_steps = [0] * len(self._pair_ids)


# The engine constructs the processor itself, so the API needs a way back to the
# live instance to read its records. Safe only because `evo2_flash.Evo2` forces
# an in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); with a separate
# engine process this registry would be empty.
_INSTANCES: list[weakref.ref] = []


def _register(processor: Evo2CfgLogitsProcessor) -> None:
    _INSTANCES.append(weakref.ref(processor))


def current_processor() -> Evo2CfgLogitsProcessor | None:
    """The most recently constructed live processor, or None."""
    while _INSTANCES:
        processor = _INSTANCES[-1]()
        if processor is not None:
            return processor
        _INSTANCES.pop()
    return None
