from typing import Optional, Callable
from typing_extensions import Unpack, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention.flex_attention import BlockMask

from transformers import AutoTokenizer, TextStreamer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding as OrthrusRotaryEmbedding,
    Qwen3Config as OrthrusConfig,
    Qwen3PreTrainedModel as OrthrusPreTrainedModel,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    Qwen3RMSNorm, 
    Qwen3MLP,
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)

_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

def fused_flex_attention(q, k, v, mask=None):
    kernel_options = {"BACKEND": "FLASH"}
    if mask is not None:
        q_sparse_bs, kv_sparse_bs = mask.BLOCK_SIZE
        kernel_options["sparse_block_size"] = (int(q_sparse_bs), int(kv_sparse_bs))
    return _compiled_flex_attention(
        q,
        k,
        v,
        block_mask=mask,
        enable_gqa=True,
        kernel_options=kernel_options,
    )
    
def generate_dual_pass_mask(
    B: int,
    H: int,
    diffusion_length: int,
    ar_len: int,
    block_size: int,
    causal_limit: torch.Tensor,
    sparse_block_size: int = 128,
):
    assert causal_limit is not None, "causal_limit tensor is required for dual-pass diffusion masking"
    if causal_limit.shape[1] != diffusion_length:
        raise ValueError(f"causal_limit shape mismatch: expected second dim={diffusion_length}, got {causal_limit.shape[1]}")

    def dual_pass_mask_fn(b, h, q_idx, kv_idx):
        is_kv_ar = kv_idx < ar_len
        valid_ar = is_kv_ar & (kv_idx <= causal_limit[b, q_idx])

        draft_kv_idx = kv_idx - ar_len
        q_block_id = q_idx // block_size
        kv_block_id = draft_kv_idx // block_size
        valid_diffusion = (~is_kv_ar) & (q_block_id == kv_block_id)
        return valid_ar | valid_diffusion

    return create_block_mask(
        dual_pass_mask_fn,
        B=B,
        H=H,
        Q_LEN=diffusion_length,
        KV_LEN=ar_len + diffusion_length,
        BLOCK_SIZE=sparse_block_size,
    )


class OrthrusAttention(Qwen3Attention):
    def __init__(self, config: OrthrusConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.q_proj_diff = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_diff = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_diff = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.o_proj_diff = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        is_diffusion_pass: bool = False,
        causal_limit: torch.Tensor | None = None,
        ar_seq_len: int | None = None,
        flex_block_mask: BlockMask | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        if not is_diffusion_pass:
            self.is_causal = True
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        
        self.is_causal = False
        if past_key_values is None or ar_seq_len is None:
            raise ValueError(f"Both past_key_values and ar_seq_len are required for diffusion pass, but got past_key_values={past_key_values} and ar_seq_len={ar_seq_len}")
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings
        
        query_states = self.q_norm_diff(self.q_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm_diff(self.k_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj_diff(hidden_states).view(hidden_shape).transpose(1, 2)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        shared_cache = past_key_values.layers[self.layer_idx]
        shared_key_states = shared_cache.keys
        shared_value_states = shared_cache.values
        
        assert shared_key_states.shape[2] == ar_seq_len and shared_value_states.shape[2] == ar_seq_len, f"Shared cache shorter than ar_seq_len at layer {self.layer_idx}: k_len={shared_key_states.shape[2]}, v_len={shared_value_states.shape[2]}, ar_seq_len={ar_seq_len}"
        shared_key_states = shared_key_states[:, :, :ar_seq_len, :]
        shared_value_states = shared_value_states[:, :, :ar_seq_len, :]
        
        key_states = torch.cat([shared_key_states, key_states], dim=2)
        value_states = torch.cat([shared_value_states, value_states], dim=2)
        
        if self.training:
            if flex_block_mask is None:
                raise ValueError(f"flex_block_mask is required for diffusion pass during training, but got flex_block_mask={flex_block_mask}")
            attn_output = fused_flex_attention(query_states, key_states, value_states, mask=flex_block_mask)
            attn_output = attn_output.transpose(1, 2)
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                is_causal=self.is_causal,
                **kwargs,
            )
            
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj_diff(attn_output)
        
        return attn_output, None
    

class OrthrusDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OrthrusConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OrthrusAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        flex_block_mask: BlockMask | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            flex_block_mask=flex_block_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class OrthrusModel(OrthrusPreTrainedModel):
    _no_split_modules = ["OrthrusDecoderLayer"]
    
    def __init__(self, config: OrthrusConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OrthrusDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OrthrusRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_diffusion_pass: bool = False,
        causal_limit: torch.Tensor | None = None,
        ar_seq_len: int | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            sequence_length = inputs_embeds.shape[1]
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + sequence_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        flex_block_mask = None
        if self.training and is_diffusion_pass:
            assert causal_limit is not None and ar_seq_len is not None, f"causal_limit={causal_limit} and ar_seq_len={ar_seq_len} are required for diffusion pass during training"
            batch_size, diffusion_length = inputs_embeds.shape[:2]
            flex_block_mask = generate_dual_pass_mask(
                B=batch_size,
                H=self.config.num_attention_heads,
                diffusion_length=diffusion_length,
                ar_len=ar_seq_len,
                block_size=self.config.block_size,
                causal_limit=causal_limit,
            )
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                is_diffusion_pass=is_diffusion_pass,
                causal_limit=causal_limit,
                ar_seq_len=ar_seq_len,
                flex_block_mask=flex_block_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class OrthrusLM(OrthrusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    
    def __init__(self, config):
        super().__init__(config)
        self.model = OrthrusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        is_diffusion_pass: bool = False,
        causal_limit: torch.Tensor | None = None,
        ar_seq_len: int | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            is_diffusion_pass=is_diffusion_pass,
            causal_limit=causal_limit,
            ar_seq_len=ar_seq_len,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        sliced_hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(sliced_hidden_states)
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,),
            attentions=outputs.attentions,
        )
        
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = None,
        max_length: int = None,
        temperature: float = 0.0,
        top_k: int = 20,
        top_p: float = 0.8,
        eos_token_id: Optional[int] = None,
        streamer: Optional[TextStreamer] = None,
        use_diffusion_mode: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        
        eos_token_id = eos_token_id or getattr(self.config, "eos_token_id", None)
    
        if not use_diffusion_mode:
            return super().generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                streamer=streamer,
                use_cache=True,
                **kwargs,
            )

        device = input_ids.device
        num_input_tokens = input_ids.shape[1]
        max_length = max_length or (num_input_tokens + max_new_tokens)
        block_size = self.config.block_size
        mask_token_id = self.config.mask_token_id
        past_key_values = DynamicCache(config=self.config)

        output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
        output_ids[:, :num_input_tokens] = input_ids

        if streamer:
            streamer.put(input_ids)

        def sample(logits: torch.Tensor):
            if temperature < 1e-5:
                return logits.argmax(dim=-1), None
            
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[..., [-1]]] = -float('Inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                logits[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1]), probs

        # Initial Pass
        position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
        outputs = self(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values)
        
        start_idx = num_input_tokens
        next_token, _ = sample(outputs.logits[:, -1, :])
        output_ids[:, start_idx] = next_token
        
        if streamer: streamer.put(next_token)
        if next_token.item() == eos_token_id:
            if streamer: streamer.end()
            return output_ids[:, :start_idx + 1]

        while start_idx < max_length - 1:
            diff_len = min(block_size, max_length - start_idx)
            diff_block_ids = torch.full((1, diff_len), mask_token_id, dtype=torch.long, device=device)
            diff_block_ids[:, 0] = output_ids[:, start_idx]
            diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)

            # Diffusion Path
            diff_outputs = self(
                input_ids=diff_block_ids,
                position_ids=diff_position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                is_diffusion_pass=True,
                ar_seq_len=start_idx,
            )
            
            if diff_len > 1:
                diff_tokens, diff_probs = sample(diff_outputs.logits[:, :-1, :])
            else:
                diff_tokens, diff_probs = torch.empty((1, 0), dtype=torch.long, device=device), None

            proposed_block = torch.cat([output_ids[:, start_idx:start_idx+1], diff_tokens], dim=1)

            # Autoregressive Path for Intra-model Consistency
            ar_outputs = self(
                input_ids=proposed_block,
                position_ids=diff_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                is_diffusion_pass=False,
            )
            ar_tokens, ar_probs = sample(ar_outputs.logits)

            acceptance_len = 0
            if temperature < 1e-5:
                matches = (diff_tokens == ar_tokens[:, :-1])
                acceptance_len = matches.cumprod(dim=1).sum(dim=1)[0].item()
                next_token = ar_tokens[:, acceptance_len]
            else:
                for i in range(diff_tokens.shape[1]):
                    q_prob = diff_probs[0, i, diff_tokens[0, i]]
                    p_prob = ar_probs[0, i, diff_tokens[0, i]]
                    if torch.rand(1, device=device).item() < min(1.0, (p_prob / max(q_prob, 1e-8)).item()):
                        acceptance_len += 1
                    else:
                        break
                
                p_dist = ar_probs[0, acceptance_len]
                if acceptance_len < diff_tokens.shape[1]:
                    residual = torch.clamp(p_dist - diff_probs[0, acceptance_len], min=0.0)
                    residual_sum = residual.sum()
                    next_token = torch.multinomial(residual / residual_sum if residual_sum > 1e-5 else p_dist, 1)
                else:
                    next_token = torch.multinomial(p_dist, 1)

            end_idx = start_idx + acceptance_len + 1
            accepted_block = proposed_block[:, :acceptance_len + 1]
            
            eos_positions = (accepted_block == eos_token_id).nonzero()
            if len(eos_positions) > 0:
                eos_offset = eos_positions[0, -1].item()
                output_ids[:, start_idx : start_idx + eos_offset + 1] = accepted_block[:, :eos_offset + 1]
                if streamer:
                    streamer.put(accepted_block[:, 1 : eos_offset + 1])
                    streamer.end()
                return output_ids[:, : start_idx + eos_offset + 1]

            output_ids[:, start_idx:end_idx] = accepted_block
            if streamer and acceptance_len > 0:
                streamer.put(accepted_block[:, 1:])
            
            start_idx = end_idx
            past_key_values.crop(start_idx)

            if start_idx < max_length:
                output_ids[:, start_idx] = next_token
                if streamer: streamer.put(next_token)
                
                if next_token.item() == eos_token_id:
                    if streamer: streamer.end()
                    return output_ids[:, :start_idx + 1]

        if streamer: streamer.end()
        return output_ids[:, :max_length]