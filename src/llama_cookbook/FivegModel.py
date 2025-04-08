#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from typing import Optional, Tuple, Union, Tuple
from transformers import LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.generation.utils import GenerationMixin
from transformers import logging

logger = logging.get_logger(__name__)


# -------------------------------------------------------------------------
# 1. Helper Modules: DomainAdapter, CodeAdapter, KnowledgeConditionedAttention,
#    and CrossAttention blocks for logs/config/code streams
# -------------------------------------------------------------------------

class DomainAdapterMLP(nn.Module):
    """
    A lightweight domain adapter for 5G knowledge.
    Uses a simple bottleneck with SILU activation.
    """
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=False)
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # LN -> down_proj -> SiLU -> up_proj -> + residual
        residual = hidden_states
        x = self.layernorm(hidden_states)
        x = self.down_proj(x)
        x = F.silu(x)
        x = self.up_proj(x)
        return residual + x


class CodeAdapterMLP(nn.Module):
    """
    A lightweight adapter specialized for code.
    Similar structure but can remain separate for clarity.
    """
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=False)
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # LN -> down_proj -> SiLU -> up_proj -> + residual
        residual = hidden_states
        x = self.layernorm(hidden_states)
        x = self.down_proj(x)
        x = F.silu(x)
        x = self.up_proj(x)
        return residual + x

class DomainAdapter(nn.Module):
    """
    A multi-head attention-based adapter for 5G knowledge (Modified for Quantization Compatibility).
    Uses manual MHA implementation with F.scaled_dot_product_attention.
    Internal projections (Q, K, V, O within attention) should be kept in float16/float32
    by adding "DomainAdapter" to modules_to_not_convert in BitsAndBytesConfig.
    """
    def __init__(self, hidden_size: int, adapter_size: int = 64, num_heads: int = 4):
        super().__init__()
        if adapter_size % num_heads != 0:
            raise ValueError(
                f"adapter_size {adapter_size} must be divisible by num_heads {num_heads}"
            )

        self.head_dim = adapter_size // num_heads
        self.num_heads = num_heads
        self.adapter_size = adapter_size

        # LN normalizes the input before we do anything
        self.layernorm = nn.LayerNorm(hidden_size) # LayerNorm often kept in higher precision

        # Down-project from hidden_size -> adapter_size
        # This might be quantized by bitsandbytes if not excluded
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=False)

        # --- Manual Multi-Head Attention Projections ---
        # These standard nn.Linear layers will NOT be quantized if "DomainAdapter"
        # is added to modules_to_not_convert during inference loading.
        self.q_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.k_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.v_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.o_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        # --- End Manual MHA Projections ---

        # Up-project back to hidden_size
        # This might be quantized by bitsandbytes if not excluded
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        # Standard initialization similar to nn.Linear and nn.MultiheadAttention
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Splits hidden_size dim into nh * hs.
        Input: (B, L, E=num_heads*head_dim)
        Output: (B, nh, L, hs)
        """
        batch_size, seq_length, _ = tensor.size()
        return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Merges nh * hs dims back to E.
        Input: (B, nh, L, hs)
        Output: (B, L, E=num_heads*head_dim)
        """
        batch_size, _, seq_length, _ = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_length, self.adapter_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure input is in the correct dtype for the LayerNorm
        input_dtype = hidden_states.dtype
        adapter_compute_dtype = self.q_proj.weight.dtype # Dtype of internal non-quantized layers (e.g., fp16)

        # Step 1) LN + residual handle
        residual = hidden_states
        # LayerNorm might operate in FP32 for stability, check its weight dtype
        x = self.layernorm(hidden_states.to(self.layernorm.weight.dtype)).to(input_dtype) # Cast input, cast output back

        # Step 2) Down project to smaller dimension
        # down_proj might be quantized (Linear4bit), output should match compute_dtype (e.g., FP16)
        x = self.down_proj(x)

        # --- Manual MHA ---
        # Ensure x is in the adapter's compute dtype (FP16) before internal projections
        x_adapter_dtype = x.to(adapter_compute_dtype)

        # Project Q, K, V (using non-quantized layers)
        q = self.q_proj(x_adapter_dtype)
        k = self.k_proj(x_adapter_dtype)
        v = self.v_proj(x_adapter_dtype)

        # Split heads
        q = self._split_heads(q) # (B, nh, L, hs)
        k = self._split_heads(k) # (B, nh, L, hs)
        v = self._split_heads(v) # (B, nh, L, hs)

        # Apply Scaled Dot Product Attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        # Combine heads
        attn_output = self._combine_heads(attn_output) # (B, L, adapter_size)

        # Output projection (using non-quantized layer)
        attn_output = self.o_proj(attn_output)
        # --- End Manual MHA ---

        # Step 4) Up project back to hidden_size
        # up_proj might be quantized, input needs to match its expected dtype (likely adapter_compute_dtype)
        # The output will be in the compute_dtype (e.g., FP16)
        x = self.up_proj(attn_output.to(self.up_proj.weight.dtype if hasattr(self.up_proj, 'weight') else adapter_compute_dtype))

        # Step 5) Residual
        # Ensure residual add is done in the original input dtype
        return residual + x.to(input_dtype)


class CodeAdapter(nn.Module):
    """
    A multi-head attention-based adapter for code (Modified for Quantization Compatibility).
    Uses manual MHA implementation with F.scaled_dot_product_attention.
    Internal projections (Q, K, V, O within attention) should be kept in float16/float32
    by adding "CodeAdapter" to modules_to_not_convert in BitsAndBytesConfig.
    """
    def __init__(self, hidden_size: int, adapter_size: int = 64, num_heads: int = 4):
        super().__init__()
        if adapter_size % num_heads != 0:
            raise ValueError(
                f"adapter_size {adapter_size} must be divisible by num_heads {num_heads}"
            )

        self.head_dim = adapter_size // num_heads
        self.num_heads = num_heads
        self.adapter_size = adapter_size

        # LN normalizes the input before we do anything
        self.layernorm = nn.LayerNorm(hidden_size)

        # Down-project from hidden_size -> adapter_size
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=False)

        # --- Manual Multi-Head Attention Projections ---
        self.q_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.k_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.v_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        self.o_proj = nn.Linear(adapter_size, adapter_size, bias=True)
        # --- End Manual MHA Projections ---

        # Up-project back to hidden_size
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = tensor.size()
        return tensor.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_length, _ = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_length, self.adapter_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        adapter_compute_dtype = self.q_proj.weight.dtype

        residual = hidden_states
        x = self.layernorm(hidden_states.to(self.layernorm.weight.dtype)).to(input_dtype)
        x = self.down_proj(x)

        x_adapter_dtype = x.to(adapter_compute_dtype)
        q = self.q_proj(x_adapter_dtype)
        k = self.k_proj(x_adapter_dtype)
        v = self.v_proj(x_adapter_dtype)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        attn_output = self._combine_heads(attn_output)
        attn_output = self.o_proj(attn_output)

        x = self.up_proj(attn_output.to(self.up_proj.weight.dtype if hasattr(self.up_proj, 'weight') else adapter_compute_dtype))

        return residual + x.to(input_dtype)


class KnowledgeConditionedAttention(nn.Module):
    """
    Allows tokens to attend to a small learnable memory matrix,
    storing repeated 5G patterns, code references, etc.
    """
    def __init__(self, hidden_size: int, memory_slots: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.domain_memory = nn.Parameter(torch.randn(memory_slots, hidden_size))
        nn.init.kaiming_normal_(self.domain_memory)

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj   = nn.Linear(hidden_size, hidden_size, bias=False)

        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape
        # Expand memory for the batch
        memory_expanded = self.domain_memory.unsqueeze(0).expand(B, self.memory_slots, D)

        Q = self.query_proj(hidden_states)
        K = self.key_proj(memory_expanded)
        V = self.value_proj(memory_expanded)

        # reshape for multi-head
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # (B, heads, L, head_dim)
        K = K.view(B, self.memory_slots, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, self.memory_slots, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, heads, L, memory_slots)

        context = torch.matmul(attn_weights, V)       # (B, heads, L, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        # LN + residual
        context = self.out_proj(context)
        residual = hidden_states
        out = self.layernorm(residual + context)
        return out


class CrossAttention(nn.Module):
    """
    Standard multi-head cross-attention:
      - Q from hidden_states
      - K, V from extra_hidden_states
    """
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj   = nn.Linear(hidden_size, hidden_size, bias=False)

        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        extra_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # hidden_states (main) => Q
        # extra_hidden_states   => K, V
        B, L_main, D = hidden_states.size()
        L_extra = extra_hidden_states.size(1)

        Q = self.query_proj(hidden_states)
        K = self.key_proj(extra_hidden_states)
        V = self.value_proj(extra_hidden_states)

        Q = Q.view(B, L_main, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, L_main, head_dim)
        K = K.view(B, L_extra, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L_extra, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            # adapt shape or broadcast if needed
            scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, heads, L_main, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, L_main, D)

        out = self.out_proj(context)
        # LN + residual
        out = self.layernorm(hidden_states + out)
        return out


# -------------------------------------------------------------------------
# 2. 5G Decoder Layer
#    We'll incorporate:
#      - Base Self-Attn + MLP (from LLaMA)
#      - DomainAdapter
#      - CodeAdapter
#      - KnowledgeConditionedAttention (KCA)
#      - Separate cross-attn for logs, config
# -------------------------------------------------------------------------
class FivegLlamaDecoderLayer(LlamaDecoderLayer):
    """
    An extension of the LlamaDecoderLayer that includes:
      1) DomainAdapter
      2) CodeAdapter
      3) KnowledgeConditionedAttention
      4) Three separate CrossAttention blocks for logs, config, code
    The forward pass is structured in a sequential manner:
      - standard self-attn + MLP
      - domain adapter
      - code adapter
      - knowledge-conditioned attention
      - cross-attn to logs (if provided)
      - cross-attn to config (if provided)
    """

    def __init__(self, config: LlamaConfig, layer_idx: int,
                 adapter_dim: int = 64,
                 memory_slots: int = 32,
                 kca_heads: int = 4):
        super().__init__(config, layer_idx)

        # Domain + Code Adapters
        self.domain_adapter = DomainAdapter(hidden_size=config.hidden_size, adapter_size=adapter_dim)
        self.code_adapter   = CodeAdapter(hidden_size=config.hidden_size,   adapter_size=adapter_dim)

        # KnowledgeConditionedAttention
        self.kca = KnowledgeConditionedAttention(
            hidden_size=config.hidden_size,
            memory_slots=memory_slots,
            num_heads=kca_heads
        )

        # Cross-Attention for logs / config / code
        self.num_heads = config.num_attention_heads
        self.logs_cross_attn   = CrossAttention(hidden_size=config.hidden_size, num_heads=self.num_heads)
        self.config_cross_attn = CrossAttention(hidden_size=config.hidden_size, num_heads=self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # Additional embeddings for logs, config, code
        logs_hidden_states: Optional[torch.Tensor] = None,
        config_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # 1) LLaMA self-attn + MLP
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 2) Domain Adapter
        hidden_states = self.domain_adapter(hidden_states)

        # 3) Code Adapter
        hidden_states = self.code_adapter(hidden_states)

        # 4) Knowledge Conditioned Attention
        hidden_states = self.kca(hidden_states)

        # 5) Cross-attn: logs
        if logs_hidden_states is not None:
            hidden_states = self.logs_cross_attn(hidden_states, logs_hidden_states, attention_mask=None)

        # 6) Cross-attn: config
        if config_hidden_states is not None:
            hidden_states = self.config_cross_attn(hidden_states, config_hidden_states, attention_mask=None)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# -------------------------------------------------------------------------
# 3. 5G Model that uses FivegLlamaDecoderLayer instead of LlamaDecoderLayer
# -------------------------------------------------------------------------
class FivegLlamaModel(LlamaPreTrainedModel):
    """
    Replaces the original LlamaDecoderLayer with FivegLlamaDecoderLayer.
    This model can handle additional embeddings for logs, config,
    plus has domain+code adapters and knowledge-conditioned attention.
    """
    def __init__(self, config: LlamaConfig,
                 adapter_dim: int = 64,
                 memory_slots: int = 32,
                 kca_heads: int = 4):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Build extended decoder layers
        self.layers = nn.ModuleList([
            FivegLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                adapter_dim=adapter_dim,
                memory_slots=memory_slots,
                kca_heads=kca_heads
            ) for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional["Cache"] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # new streams
        logs_hidden_states: Optional[torch.Tensor] = None,
        config_hidden_states: Optional[torch.Tensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                logs_hidden_states=logs_hidden_states,
                config_hidden_states=config_hidden_states,
                **flash_attn_kwargs
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                if len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
                else:
                    all_self_attns += (None,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return (hidden_states, past_key_values, all_hidden_states, all_self_attns)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# -------------------------------------------------------------------------
# 4. 5G Causal LM: Similar to LlamaForCausalLM but uses FivegLlamaModel
# -------------------------------------------------------------------------
class FivegLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self,
                 config: LlamaConfig,
                 adapter_dim: int = 512,
                 memory_slots: int = 128,
                 kca_heads: int = 16):
        super().__init__(config)
        self.model = FivegLlamaModel(
            config,
            adapter_dim=adapter_dim,
            memory_slots=memory_slots,
            kca_heads=kca_heads
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union["Cache", Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        # Additional streams:
        logs_hidden_states: Optional[torch.Tensor] = None,
        config_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logs_hidden_states=logs_hidden_states,
            config_hidden_states=config_hidden_states,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Apply lm_head to the entire hidden_states output from the base model
        # The generate method handles selecting the correct token(s) for prediction
        logits = self.lm_head(hidden_states)
        if num_logits_to_keep > 0:
            logits = logits[:, -num_logits_to_keep:, :]

        loss = None
        if labels is not None:
            # Standard loss calculation (shift logits and labels)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )