# SPDX-License-Identifier: Apache-2.0
"""Opt-in activation of Orthrus diffusion-mode speculative decoding.

vLLM has no out-of-tree hook for speculative drafters that need the target's
attention metadata and KV cache (``method="custom_class"`` only passes token
ids). Orthrus' drafter is structurally a DFlash-style parallel drafter (one
bonus token plus K mask tokens, non-causal, KV-shared with the target), so
vLLM's existing ``dflash`` slot already does the config validation, aux
hidden state plumbing and KV-group wiring it needs. This module swaps the
class the runner builds for that slot.

Enable with ``ORTHRUS_VLLM_DIFFUSION=1`` and then use::

    speculative_config={"method": "dflash",
                        "model": "chiennv/Orthrus-Qwen3-1.7B",
                        "num_speculative_tokens": 4}

It is pinned to the vLLM versions it was tested on and refuses to patch
anything else.
"""

import os

SUPPORTED_VLLM_PREFIXES = ("0.31.",)
ENV_FLAG = "ORTHRUS_VLLM_DIFFUSION"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def activate() -> None:
    import vllm

    if not vllm.__version__.startswith(SUPPORTED_VLLM_PREFIXES):
        raise RuntimeError(
            f"{ENV_FLAG}=1 is only supported on vLLM {SUPPORTED_VLLM_PREFIXES}, "
            f"found {vllm.__version__}. Unset it to use the autoregressive plugin."
        )
    # The drafter lives in the legacy GPUModelRunner; vLLM 0.31's V2 runner has
    # its own speculator classes that this plugin does not patch.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    from vllm import ModelRegistry
    from vllm.v1.worker import gpu_model_runner

    # SpeculativeConfig rewrites the draft architecture of a dflash draft to
    # "DFlash<arch>"; both spellings resolve to the same Orthrus model, which
    # switches to its diffusion forward via building_diffusion_draft().
    supported = ModelRegistry.get_supported_archs()
    for arch in ("DFlashOrthrusLM", "DFlashOrthrusForCausalLM"):
        if arch not in supported:
            ModelRegistry.register_model(arch, "orthrus_vllm.model:OrthrusForCausalLM")

    from .orthrus_proposer import OrthrusProposer

    gpu_model_runner.DFlashProposer = OrthrusProposer
