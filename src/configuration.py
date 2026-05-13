from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from typing import Optional

class OrthrusConfig(Qwen3Config):
    model_type = "orthrus"

    def __init__(
        self,
        *args,
        block_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id