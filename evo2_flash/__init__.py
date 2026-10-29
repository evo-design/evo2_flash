"""Out-of-tree vLLM plugin for Evo 2 / StripedHyena2.

Drop-in replacement for `evo2.Evo2` (see evo2_flash/api.py and README.md):

    from evo2_flash import Evo2
    model = Evo2("evo2_7b")

Lower level:

    import evo2_flash; evo2_flash.register()
    from vllm import LLM
    llm = LLM(model="ckpt/evo2_7b_hf", skip_tokenizer_init=True, ...)

If installed as a package, the same thing happens automatically via the
`vllm.general_plugins` entry point declared in pyproject.toml.

Ported from github.com/AlexYkHao/evo2-vllm at 5d6396e. Comments that cite
`tests/`, `scripts/`, `evo2_layers/`, `PLAN.md`, `MTP_DRAFTER.md`, `FP8_SURGERY.md`
or `docs/` refer to that repository.
"""

from .config import StripedHyena2Config

__all__ = ["StripedHyena2Config", "register", "Evo2", "CharLevelTokenizer",
           "GenerationOutput", "MODEL_NAMES"]

_API = {"Evo2", "CharLevelTokenizer", "GenerationOutput", "MODEL_NAMES",
        "reverse_complement", "layer_name_table"}


def __getattr__(name):
    # Lazy, so `import evo2_flash` (e.g. by the converter, in an env without
    # vLLM) does not import vLLM.
    if name in _API:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_REGISTERED = False


def register() -> None:
    """Register the config with transformers and the model with vLLM."""
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig

    from .config import Evo2MTPConfig

    for name, cls in (("striped_hyena2", StripedHyena2Config), ("evo2_mtp", Evo2MTPConfig)):
        try:
            AutoConfig.register(name, cls)
        except ValueError:
            pass  # already registered

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "StripedHyena2ForCausalLM",
        "evo2_flash.model:StripedHyena2ForCausalLM",
    )
    # MTP draft head, run by vLLM's EAGLE proposer (speculative method "eagle").
    ModelRegistry.register_model(
        "EagleEvo2MTPModel",
        "evo2_flash.mtp_draft:EagleEvo2MTPModel",
    )
    # ... and its EAGLE-3 form (multi-layer target features; method "eagle3").
    ModelRegistry.register_model(
        "Eagle3Evo2MTPModel",
        "evo2_flash.mtp_draft:Eagle3Evo2MTPModel",
    )
    _REGISTERED = True
