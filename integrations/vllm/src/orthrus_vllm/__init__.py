# SPDX-License-Identifier: Apache-2.0
"""vLLM out-of-tree plugin: native Orthrus model support.

Registers Orthrus (https://github.com/chiennv2000/orthrus) as a vLLM model
via the standard ``vllm.general_plugins`` entry point, so
``chiennv/Orthrus-*`` checkpoints load with ``trust_remote_code=False``.

This entry point activates the autoregressive serving path. The diffusion /
self-speculative decoding path is opt-in (``ORTHRUS_VLLM_DIFFUSION=1``); see
``docs/DIFFUSION_MODE.md``.
"""


def register() -> None:
    from transformers import AutoConfig

    from vllm import ModelRegistry

    from .transformers_utils.config import OrthrusConfig

    # Standard HF mechanism: lets `AutoConfig.from_pretrained(...,
    # trust_remote_code=False)` resolve `model_type="orthrus"` checkpoints
    # without needing the checkpoint's own remote code.
    AutoConfig.register("orthrus", OrthrusConfig)

    if "OrthrusForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "OrthrusForCausalLM",
            "orthrus_vllm.model:OrthrusForCausalLM",
        )
    if "OrthrusLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "OrthrusLM",
            "orthrus_vllm.model:OrthrusForCausalLM",
        )

    from .speculative import activate

    if activate.enabled():
        activate.activate()
