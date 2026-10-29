"""Evo 2's correctness test (evo2/test/test_evo2.py), run through evo2_flash.

    python -m evo2_flash.test.test_evo2 --model_name evo2_7b

Scores the four sequences of evo2's test data (`data/prompts.csv`, the same file)
and compares the mean next-token loss with the value the Evo 2 authors published
for the model. It passes within their eps, 1e-3, or 5e-3 with the lossy
`--kv_cache_dtype fp8`; the exit code is 0 on a pass and 1 otherwise.

The engine is the one `Evo2(...)` builds, so the test covers checkpoint
conversion on first use, the token-table swap on GPUs without FP8 tensor cores,
the Triton kernels and CUDA graphs. Accuracy is printed, as in evo2, but not
gated: on DNA the model sits near maximum entropy, so argmax agreement is noisy.
"""

import argparse
import csv
import json
import os
from importlib import resources

# (loss, accuracy %) published in evo2's test_evo2.py; evo2_20b from evo2 0.6.0.
EXPECTED = {
    "evo2_1b_base": (0.501953125, 79.556),
    "evo2_7b": (0.3476563, 86.346),
    "evo2_7b_base": (0.3520508, 85.921),
    "evo2_20b": (0.2166748046875, 91.666),
    "evo2_40b": (0.2159424, 91.673),
    "evo2_40b_base": (0.2149658, 91.741),
}
EPS, EPS_LOSSY = 1e-3, 5e-3


def read_prompts() -> list[str]:
    path = resources.files("evo2_flash.test") / "data" / "prompts.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.reader(f)
        next(rows)  # header
        return [row[0] for row in rows]


def score(model, seqs):
    """Per-sequence mean next-token loss and argmax accuracy, from prompt logprobs."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    outs = model.llm.generate(
        [TokensPrompt(prompt_token_ids=list(s.encode())) for s in seqs],
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False),
        use_tqdm=False)
    losses, accs = [], []
    for s, o in zip(seqs, outs):
        targets = list(s.encode())[1:]
        steps = o.prompt_logprobs[1:]  # position 0 predicts nothing
        # Every position must be scored: a test that skipped some would pass on less.
        assert len(steps) == len(targets) and all(d and t in d for d, t in zip(steps, targets))
        losses.append(-sum(d[t].logprob for d, t in zip(steps, targets)) / len(targets))
        top = (min(d.items(), key=lambda kv: kv[1].rank or 99)[0] for d in steps)
        accs.append(sum(p == t for p, t in zip(top, targets)) / len(targets))
    return losses, accs


def main():
    ap = argparse.ArgumentParser(description="Evo 2's correctness test through evo2_flash")
    ap.add_argument("--model_name", choices=sorted(EXPECTED), default="evo2_7b")
    ap.add_argument("--local_path", help="a converted checkpoint directory or a vortex .pt "
                    "(default: the checkpoint cache, converted on first use)")
    ap.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default="auto")
    args = ap.parse_args()

    import torch
    import vllm

    from evo2_flash import Evo2
    from evo2_flash.api import CONFIG_MAP, resolve_checkpoint

    # Published losses differ by less than eps for some pairs (20B and 40B, 40B and
    # 40B base), so a checkpoint converted from another model's config could pass.
    # The HF snapshot of evo2_20b ships its config as config.yml.
    ckpt = resolve_checkpoint(args.model_name, args.local_path)
    with open(os.path.join(ckpt, "config.json")) as f:
        source = json.load(f).get("_source_config")
    allowed = (CONFIG_MAP[args.model_name],) + (("config.yml",) if args.model_name == "evo2_20b" else ())
    if source is not None and source not in allowed:
        raise SystemExit(f"{ckpt} was converted from {source}, not from {allowed[0]} "
                         f"({args.model_name})")

    model = Evo2(args.model_name, local_path=ckpt, kv_cache_dtype=args.kv_cache_dtype)
    sched = model.llm.llm_engine.vllm_config.scheduler_config
    print(f"vLLM {vllm.__version__}, {torch.cuda.get_device_name()}, "
          f"max_num_batched_tokens={sched.max_num_batched_tokens}, checkpoint {model.checkpoint}")

    losses, accs = score(model, read_prompts())
    print("\nSequence Results:")
    for i, (loss, acc) in enumerate(zip(losses, accs)):
        print(f"Sequence {i + 1}: Loss = {loss:.3f}, Accuracy = {acc:.2%}")
    mean_loss = sum(losses) / len(losses)
    print(f"\nMean Loss: {mean_loss:.7f}")
    print(f"Mean Accuracy: {100 * sum(accs) / len(accs):.3f}%")

    expected = EXPECTED[args.model_name][0]
    eps = EPS if args.kv_cache_dtype == "auto" else EPS_LOSSY
    delta = abs(mean_loss - expected)
    ok = delta < eps
    print(f"\nTest {'Passed' if ok else 'Failed'}: loss {mean_loss:.7f}, expected "
          f"{expected:.7f}, delta {delta:.2e} (eps {eps:.0e})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
