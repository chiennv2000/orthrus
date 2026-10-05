# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orthrus model configuration."""

import os

from transformers import Qwen3Config


class OrthrusConfig(Qwen3Config):
    """Configuration for Orthrus checkpoints.

    Orthrus extends Qwen3 with diffusion decoding parameters. The regular
    autoregressive path is Qwen3-compatible; ``block_size`` and
    ``mask_token_id`` are used by Orthrus' diffusion generation mode.
    """

    model_type = "orthrus"

    def __init__(
        self,
        *args,
        block_size: int | None = None,
        mask_token_id: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        if os.environ.get("ORTHRUS_VLLM_DIFFUSION") == "1":
            # Diffusion mode rides vLLM's dflash drafter slot (see
            # speculative/activate.py), which reads the mask token and the
            # target aux-hidden-state layers from ``dflash_config``. Orthrus'
            # proposer ignores the aux hidden states, one layer is requested
            # only to satisfy the generic plumbing.
            self.dflash_config = {
                "mask_token_id": mask_token_id,
                "target_layer_ids": [1],
            }
