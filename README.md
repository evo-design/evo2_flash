# evo2_flash

Evo 2 (StripedHyena2) on vLLM, with a drop-in replacement for `evo2.Evo2`. Evo2_flash supports more accessible genomic ICL across diverse biological systems.

## Requirements and install

- vLLM **0.26.0**, the `+cu129` wheel (the PyPI default is cu130), with torch 2.11 and
  Python 3.12. vLLM is not a declared dependency, so install it first.
- One GPU; the numbers below are from H100/H200. The fp8 input projections that the 1B,
  7B, 20B and 40B checkpoints ship with run as fp8 GEMMs on sm_89+ (L4, L40S, H100,
  H200). Older GPUs (A100, RTX 3090, A40) need the fallback described under
  [GPUs without FP8 tensor cores](#gpus-without-fp8-tensor-cores). 40B needs an H200,
  or fp8 weights on an H100.

```bash
pip install -e .        # registers the plugin via the vllm.general_plugins entry point
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # Evo2 sets this itself; the API needs the engine in-process
```

`dtype="bfloat16"` is mandatory: Evo 2's residual stream reaches ~1e12, and fp16 gives NaN.

## Quick start

```python
# from evo2 import Evo2
from evo2_flash import Evo2

model = Evo2("evo2_7b")                     # any evo2 name; converted on first use
scores = model.score_sequences(seqs)        # mean log-likelihood per sequence
out = model.generate(prompts, n_tokens=500, temperature=1.0, top_k=4)
out.sequences, out.logprobs_mean
logits, emb = model(input_ids, return_embeddings=True, layer_names=["blocks.28.mlp.l3"])
```

Pass a whole prompt list in one call. vLLM batches ragged lengths continuously, and a
repeated prompt's prefill is shared through the prefix cache.

Lower level, with plain `vllm.LLM`. The tokenizer is byte-level, so token ids are UTF-8
bytes:

```python
import evo2_flash; evo2_flash.register()    # not needed once pip-installed
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(model="/path/to/evo2_7b_hf", skip_tokenizer_init=True, dtype="bfloat16",
          enable_prefix_caching=True, max_num_seqs=128)
outs = llm.generate([TokensPrompt(prompt_token_ids=list(b"ACGT..."))],
                    SamplingParams(max_tokens=500, temperature=1.0, top_k=4,
                                   ignore_eos=True, detokenize=False))   # EOD is id 0
bytes(outs[0].outputs[0].token_ids).decode("ascii", "replace")
```

`evo2_flash.api.engine_kwargs(ckpt)` returns the validated engine configuration that
`Evo2` uses. On a GPU without FP8 tensor cores, pass the checkpoint through
`evo2_flash.api.checkpoint_for_this_gpu(path)` first. `Evo2` does this itself, but a
plain `LLM` does not.

## Drop-in replacement for evo2.Evo2

| evo2 | evo2_flash | difference |
|---|---|---|
| `Evo2(name, local_path=None, use_kernels=False)` | same | `local_path` may also be a converted dir; `use_kernels` ignored |
| `model.tokenizer` (`CharLevelTokenizer(512)`) | same semantics | `tokenize` returns `int`s, not `np.uint8` |
| `score_sequences(seqs, batch_size, prepend_bos, reduce_method, average_reverse_complement)` | same | one vLLM call; log-softmax in fp32 (evo2: bf16) |
| `generate(prompt_seqs, n_tokens, temperature, top_k, top_p, batched, cached_generation, verbose, force_prompt_threshold)` | same, + `seed=`, `prime_prefix_cache=`, CFG arguments | one vLLM call; `temperature=0` is greedy; `logits` lazy |
| `forward(input_ids, return_embeddings, layer_names)` / `model(...)` | same | vortex names mapped (table below); outputs on the GPU |
| `model.model` (vortex `StripedHyena`) | `model.model` (`StripedHyena2ForCausalLM`) | plugin module names; use `layer_names` for vortex names |

Semantic differences, all deliberate or unavoidable:

- **Scoring.** All sequences go into one vLLM call, so `batch_size` has no effect. Scores
  match evo2's `batch_size=1`. evo2 with `batch_size>1`, ragged lengths and
  `prepend_bos=False` also scores one spurious pad position for each shorter sequence.
  `score_sequences` requests prompt logprobs, which turns off the prefix-cache read.
- **Sampling.** Matches vortex `sample()`, and `top_k=1` is argmax at any temperature.
  `top_k<=0` means the whole vocabulary, and `top_p<=0` or `>=1` turns top-p off.
  `temperature=0` with `top_k!=1` is greedy here, whereas vortex divides by zero. With
  `seed=s`, request i uses seed `s+i` and reproduces exactly.
- **Length / EOS.** Every prompt gets exactly `n_tokens`, and EOD does not stop
  generation, as in evo2. `logprobs_mean` is the mean log-probability of the sampled
  tokens under the raw, untempered distribution. `batched`, `cached_generation` and
  `force_prompt_threshold` are accepted but do nothing.
- **`GenerationOutput.logits`** has vortex's layout. It is computed on first access by
  one extra prefill over prompt + completion, so the values are prefill logits. They
  agree with decode-time logits to bf16 precision. Extra fields: `token_ids` and
  `num_cached_tokens`.
- **forward().** Each row of `input_ids` is an independent causal sequence, and pad ids
  are ordinary tokens. Logits come back bf16 `(B, L, 512)`. CUDA graphs are switched off
  for these steps only, because a replayed graph does not run Python hooks.

`layer_names` takes vortex module names (native names such as `model.layers.3.mixer` are
also accepted):

| vortex name | plugin module (hooked tensor) |
|---|---|
| `embedding_layer` | `model.embed_tokens` |
| `blocks.N` | `model.layers.N` (residual stream after block N) |
| `blocks.N.pre_norm`, `blocks.N.post_norm` | `model.layers.N.{pre,post}_norm` |
| `blocks.N.mlp`, `blocks.N.mlp.l3` | `model.layers.N.mlp` / `.mlp.l3` (MLP output, before the residual add) |
| `blocks.N.mlp.l1`, `blocks.N.mlp.l2` | `model.layers.N.mlp.l1` / `.l2` (before the gate) |
| `blocks.N.projections` (Hyena) | `model.layers.N.mixer.in_proj` (3H, fp8 GEMM output) |
| `blocks.N.filter` (Hyena) | **input** of `model.layers.N.mixer.out_proj` (gated Hyena output) |
| `blocks.N.out_filter_dense` (Hyena) | `model.layers.N.mixer.out_proj` (mixer output, before the residual add) |
| `blocks.N.inner_mha_cls`, `.inner_mha_cls.out_proj` (attention) | `model.layers.N.mixer` / `.mixer.o_proj` |
| `norm` | `model.norm` |
| `unembed` | the returned logits |

The following names raise `ValueError`:
- `inner_mha_cls.Wqkv`, because the converter permutes its rows
- rotary, inner-attention and HyenaCascade internals
- a Hyena name used on an attention block, and the reverse


## Options

| `Evo2(...)` argument | effect |
|---|---|
| `kv_cache_dtype="fp8"` | 2x KV capacity. Lossy: moves the upstream gate by 1.2-1.8e-3 (7B/20B/40B) |
| `quantization="fp8"` (engine override) | fp8 weights: 2.2e-3 (40B) / 2.95e-3 (7B) on the gate; fits 40B on one 80 GB GPU |
| `speculative=<head dir>` | a trained EAGLE/EAGLE-3 draft head, K=3, probabilistic drafting: 1.3-1.7x at 7B, 1.3-2.0x at 20B |
| `cfg=True` | installs the classifier-free-guidance logits processor, enabling `generate(..., guidance=w)` |
| `max_model_len`, `gpu_memory_utilization` (0.85), `cache_dir` | engine sizing and the checkpoint cache |

Classifier-free guidance computes `guided = l_cond + w * (l_cond - l_base)`. The baseline
is `baseline_prompt` (`cfg_type="uncond"`) or `negative_prompts[i]` (`cfg_type="negative"`),
and `dna_only=True` restricts sampling to A/C/G/T. CFG and speculative decoding are mutually
exclusive, because vLLM rejects custom logits processors under speculation.

fp8-surgery checkpoints run without fp8 GEMMs. In these checkpoints block 0 is served
from an exact token table, and optionally some Hyena layers emulate Hopper fp8
accumulation (`hyena_token_table_layers`, `hyena_fp8_emulated_layers`,
`hyena_fp8_emulated_channels` in the config). They load like any other converted
checkpoint. The 1B's table variant is built by
`evo2_flash.convert.build_token_table_checkpoint`; the others come from
`scripts/fp8_surgery.py` in the source repository.

Other `vllm.LLM` arguments pass through with a warning. `dtype`, `tokenizer`,
`skip_tokenizer_init` and `model` cannot be overridden.

## Checkpoints

`Evo2("evo2_7b")` looks for `$EVO2_VLLM_CACHE/evo2_7b_hf`, defaulting to
`$HF_HOME/evo2_vllm`. If that directory is missing, it finds the vortex `.pt` the way evo2
does, downloading the HF snapshot if absent, and converts it once with
`evo2_flash.convert.convert_checkpoint`. The vortex configs ship in
`evo2_flash/evo2_configs/`, so evo2 and vortex need not be installed. 40B's `.pt.part0/.part1`
shards are read in place.

You can also pass a converted directory, `Evo2("/path/to/evo2_7b_hf")`, or a finetuned
vortex `.pt`, `Evo2("evo2_7b", local_path="ft.pt")`. Checkpoints converted by
`vllm_evo2` load unchanged.

Names: `evo2_7b`, `evo2_7b_262k`, `evo2_7b_base`, `evo2_7b_microviridae`, `evo2_1b_base`,
`evo2_20b`, `evo2_40b`, `evo2_40b_base` (`evo2_flash.MODEL_NAMES`).

## GPUs without FP8 tensor cores

On sm < 8.9 (A100, RTX 3090, A40), vLLM serves the fp8 in_proj as weight-only Marlin.
That works for the 7B, which scores 2.5-3.8e-4 on the gate on an A100 and an RTX 3090.
The stock 1B, however, depends on Hopper's FP8 rounding in block 0 and loses all signal
there: gate loss 0.805 against the authors' 0.502. `Evo2(...)` handles this through
`checkpoint_for_this_gpu`:

- **1B:** swapped for its FP8-surgery token-table variant, built once into the
  checkpoint cache (about 20 s). Through vLLM it passes the strict gate at 4.27e-4.
- **20B and 40B:** raise with instructions, because their fix is not a one-step build.
- **every 7B, and any surgery checkpoint:** used unchanged, as on sm_89+.

`EVO2_FP8_FALLBACK=off` keeps the stock checkpoint (diagnostics). On 24 GB cards such
as the L4, CFG sizes its admission waves to what the KV cache holds.

## Limitations

- One GPU per model. Tensor parallelism is not supported, because Hyena channels are
  interleaved per triple, and pipeline parallelism is not supported either.
- Trained draft heads exist for 7B and 20B only.

## Third-party code

- `evo2_flash/triton_hcl.py`: the fused modal-filter kernel, vendored from
  [Zymrael/vortex](https://github.com/Zymrael/vortex) (Apache-2.0).
- `evo2_flash/evo2_configs/*.yml`: the model configs from evo2 0.6.0 (Apache-2.0).
