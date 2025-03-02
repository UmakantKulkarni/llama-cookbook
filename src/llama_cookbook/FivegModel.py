#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaForCausalLM as _LlamaForCausalLM
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings, logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import LossKwargs
from typing import Optional, Tuple, Union, List, Dict, Tuple, Any, Mapping
from transformers.models.llama.modeling_llama import LlamaModel as _LlamaModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LLAMA_START_DOCSTRING, _CONFIG_FOR_DOC

from dataclasses import dataclass
from transformers import DefaultDataCollator
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import _torch_collate_batch, _tf_collate_batch, pad_without_fast_tokenizer_warning, _numpy_collate_batch


logger = logging.get_logger(__name__)

# --- 1. Define Modified Llama Model with Two Fiveg Embedding Layers ---
class FivegEmbeddingAttentionLayer(nn.Module):
    def __init__(self, config, num_fiveg_features, kg_embedding_dim): # kg_embedding_dim as argument
        super().__init__()
        self.fiveg_embedding = nn.Embedding(num_fiveg_features + 1, kg_embedding_dim) # Use passed kg_embedding_dim
        self.W_q = nn.Linear(config.hidden_size, kg_embedding_dim, bias=True) # Bias added
        self.W_k = nn.Linear(kg_embedding_dim, kg_embedding_dim, bias=True) # Bias added
        self.W_v = nn.Linear(kg_embedding_dim, kg_embedding_dim, bias=True) # Bias added
        self.W_o = nn.Linear(kg_embedding_dim, config.hidden_size, bias=True) # Bias added
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.scale_factor = kg_embedding_dim**-0.5 # Scale factor for scaled dot product attention


    def forward(self, hidden_states: torch.Tensor, fiveg_feature_indices: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fiveg_embeddings = self.fiveg_embedding(fiveg_feature_indices) # B, L, kg_dim

        query_layer = self.W_q(hidden_states) # B, L, kg_dim
        key_layer = self.W_k(fiveg_embeddings) # B, L, kg_dim
        value_layer = self.W_v(fiveg_embeddings) # B, L, kg_dim

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, L, L
        attention_scores = attention_scores * self.scale_factor

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))

        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # B, L, L
        attention_probs = self.attn_dropout(attention_probs) # B, L, L

        fiveg_aware_output = torch.matmul(attention_probs, value_layer) # B, L, kg_dim
        fiveg_aware_output = self.W_o(fiveg_aware_output) # B, L, hidden_size

        return fiveg_aware_output, None # Attention weights are not returned for now


# Define Modified LlamaDecoderLayer with Two Fiveg Knowledge Injection Layers
class FivegLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx) # Initialize the original LlamaDecoderLayer
        self.spec_fiveg_knowledge_layer = FivegEmbeddingAttentionLayer(config=config, num_fiveg_features=config.num_spec_features, kg_embedding_dim=config.spec_kg_embedding_dim) # Initialize FivegEmbeddingAttentionLayer for spec features
        self.code_fiveg_knowledge_layer = FivegEmbeddingAttentionLayer(config=config,  num_fiveg_features=config.num_code_features, kg_embedding_dim=config.code_kg_embedding_dim) # Initialize FivegEmbeddingAttentionLayer for code features
        self.post_spec_fiveg_knowledge_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # LayerNorm after Spec Fiveg Layer
        self.post_code_fiveg_knowledge_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # LayerNorm after Code Fiveg Layer


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        spec_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add spec_fiveg_feature_indices as input
        code_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add code_fiveg_feature_indices as input
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Spec Fiveg Knowledge Injection - Inserted AFTER MLP
        if spec_fiveg_feature_indices is not None: # Conditionally apply if spec fiveg features are provided
            spec_fiveg_output, _ = self.spec_fiveg_knowledge_layer(
                hidden_states=hidden_states,
                fiveg_feature_indices=spec_fiveg_feature_indices, # Use spec_fiveg_feature_indices
                attention_mask=attention_mask # You can pass the same attention mask if relevant
            )
            hidden_states = hidden_states + spec_fiveg_output # Residual connection
            hidden_states = self.post_spec_fiveg_knowledge_layernorm(hidden_states) # Normalize again

        # Code Fiveg Knowledge Injection - Inserted AFTER Spec Fiveg Layer
        if code_fiveg_feature_indices is not None: # Conditionally apply if code fiveg features are provided
            code_fiveg_output, _ = self.code_fiveg_knowledge_layer(
                hidden_states=hidden_states,
                fiveg_feature_indices=code_fiveg_feature_indices, # Use code_fiveg_feature_indices
                attention_mask=attention_mask # You can pass the same attention mask if relevant
            )
            hidden_states = hidden_states + code_fiveg_output # Residual connection
            hidden_states = self.post_code_fiveg_knowledge_layernorm(hidden_states) # Normalize again


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# Define Modified LlamaModel with ModifiedLlamaDecoderLayer
class FivegLlamaModel(_LlamaModel): # Inherit from the original LlamaModel
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FivegLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)] # Use ModifiedLlamaDecoderLayer here
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    # We are keeping the forward method of the original LlamaModel as it is compatible with ModifiedLlamaDecoderLayer
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        spec_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add spec_fiveg_feature_indices to forward method
        code_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add code_fiveg_feature_indices to forward method
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
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

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    spec_fiveg_feature_indices, # Pass spec_fiveg_feature_indices to decoder layer
                    code_fiveg_feature_indices # Pass code_fiveg_feature_indices to decoder layer
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    spec_fiveg_feature_indices=spec_fiveg_feature_indices, # Pass spec_fiveg_feature_indices to decoder layer
                    code_fiveg_feature_indices=code_fiveg_feature_indices, # Pass code_fiveg_feature_indices to decoder layer
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

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
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
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
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class FivegLlamaForCausalLM(_LlamaForCausalLM): # Inherit from the original LlamaForCausalLM
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = FivegLlamaModel(config) # Use ModifiedLlamaModel here
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Modify the forward method to accept and pass fiveg_feature_indices
    # @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        spec_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add spec_fiveg_feature_indices to forward method
        code_fiveg_feature_indices: Optional[torch.LongTensor] = None, # Add code_fiveg_feature_indices to forward method
        **kwargs: Unpack[KwargsForCausalLM],
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
            spec_fiveg_feature_indices=spec_fiveg_feature_indices, # Pass spec_fiveg_feature_indices to the model
            code_fiveg_feature_indices=code_fiveg_feature_indices, # Pass code_fiveg_feature_indices to the model
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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

@dataclass
class FivegDataCollatorForLanguageModeling(DefaultDataCollator):

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    fiveg_feature_vocab_size: Optional[int] = None # Add fiveg_feature_vocab_size
    code_feature_vocab: Optional[Dict[str, List[str]]] = None # Add code_feature_vocab
    fiveg_feature_embedding_dim: int = 32
    code_feature_embedding_dim: int = 32 # Example dimension for code features
    code_feature_vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
        if self.mlm_probability < 0 or self.mlm_probability > 1:
            raise ValueError("mlm_probability should be between 0 and 1.")
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")
        if self.fiveg_feature_vocab_size <= 0 and self.fiveg_feature_embedding_dim <= 0:
            raise ValueError("code_feature_embedding_dim should be greater than 0 when code_feature_vocab is provided.")
        if self.code_feature_vocab is not None:
            self.code_feature_vocab_size = sum(len(v) for v in self.code_feature_vocab.values())
        if self.code_feature_vocab and (not self.code_feature_vocab_size or self.code_feature_vocab_size <= 0):
            raise ValueError("code_feature_vocab_size must be > 0 when code_feature_vocab is provided.")



        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    @staticmethod
    def tf_bernoulli(shape, probability):
        import tensorflow as tf

        prob_matrix = tf.fill(shape, probability)
        return tf.cast(prob_matrix - tf.random.uniform(shape, 0, 1) >= 0, tf.bool)

    def tf_mask_tokens(
        self, inputs: Any, vocab_size, mask_token_id, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import tensorflow as tf

        mask_token_id = tf.cast(mask_token_id, inputs.dtype)

        input_shape = tf.shape(inputs)
        masked_indices = self.tf_bernoulli(input_shape, self.mlm_probability) & ~special_tokens_mask
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        labels = tf.where(masked_indices, inputs, -100)

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = self.tf_bernoulli(input_shape, self.mask_replace_prob) & masked_indices

        inputs = tf.where(indices_replaced, mask_token_id, inputs)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob
        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            self.tf_bernoulli(input_shape, random_replace_prob_scaled) & masked_indices & ~indices_replaced
        )
        random_words = tf.random.uniform(input_shape, maxval=vocab_size, dtype=inputs.dtype)

        inputs = tf.where(indices_random, random_words, inputs)

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        import tensorflow as tf

        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="tf", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _tf_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # Fiveg Feature Indices Processing (same as before)
        fiveg_feature_indices_batch = [example.get("fiveg_feature_indices", tf.constant([0] * self.fiveg_feature_vocab_size, dtype=tf.int64)) for example in examples] # Ensure default padding for missing features
        max_fiveg_feature_len = max(len(indices) for indices in fiveg_feature_indices_batch) if fiveg_feature_indices_batch else 0
        padded_fiveg_feature_indices_batch = []
        for indices in fiveg_feature_indices_batch:
            padding_length = max_fiveg_feature_len - len(indices)
            padded_indices = tf.concat([indices, tf.zeros(padding_length, dtype=tf.int64)], axis=0)
            padded_fiveg_feature_indices_batch.append(padded_indices)
        batch["fiveg_feature_indices"] = tf.stack(padded_fiveg_feature_indices_batch) if padded_fiveg_feature_indices_batch else tf.constant([], dtype=tf.int64)


        # Code Feature Indices Processing (NEW)
        if self.code_feature_vocab:
            code_feature_indices_batch = [example.get("code_feature_indices", tf.constant([-1] * self.code_feature_vocab_size, dtype=tf.int64)) for example in examples] # Ensure default padding for missing features
            max_code_feature_len = max(len(indices) for indices in code_feature_indices_batch) if code_feature_indices_batch else 0
            padded_code_feature_indices_batch = []
            for indices in code_feature_indices_batch:
                padding_length = max_code_feature_len - len(indices)
                padded_indices = tf.concat([indices, tf.zeros(padding_length, dtype=tf.int64)], axis=0) # Pad with 0
                padded_code_feature_indices_batch.append(padded_indices)
            batch["code_feature_indices"] = tf.stack(padded_code_feature_indices_batch) if padded_code_feature_indices_batch else tf.constant([], dtype=tf.int64)


        # Masking (same as before)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in batch["input_ids"].numpy().tolist()
                ]
                special_tokens_mask = tf.cast(tf.convert_to_tensor(special_tokens_mask, dtype=tf.int64), tf.bool)
            else:
                special_tokens_mask = tf.cast(special_tokens_mask, tf.bool)
            batch["input_ids"], batch["labels"] = self.tf_mask_tokens(
                tf.cast(batch["input_ids"], tf.int64),
                special_tokens_mask=special_tokens_mask,
                mask_token_id=self.tokenizer.mask_token_id,
                vocab_size=len(self.tokenizer),
            )
        else:
            labels = batch["input_ids"]
            if self.tokenizer.pad_token_id is not None:
                labels = tf.where(labels == self.tokenizer.pad_token_id, -100, labels)
            else:
                labels = tf.identity(labels)
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # Fiveg Feature Indices Processing (same as before)
        fiveg_feature_indices_batch = [example.get("fiveg_feature_indices", torch.tensor([0] * self.fiveg_feature_vocab_size, dtype=torch.long)) for example in examples] # Ensure default padding for missing features
        max_fiveg_feature_len = max(len(indices) for indices in fiveg_feature_indices_batch) if fiveg_feature_indices_batch else 0
        padded_fiveg_feature_indices_batch = []
        for indices in fiveg_feature_indices_batch:
            padding_length = max_fiveg_feature_len - len(indices)
            padded_indices = torch.cat([indices, torch.zeros(padding_length, dtype=torch.long)], dim=0)
            padded_fiveg_feature_indices_batch.append(padded_indices)
        batch["fiveg_feature_indices"] = torch.stack(padded_fiveg_feature_indices_batch) if padded_fiveg_feature_indices_batch else torch.tensor([], dtype=torch.long)


        # Code Feature Indices Processing (NEW)
        if self.code_feature_vocab:
            code_feature_indices_batch = [example.get("code_feature_indices", torch.tensor([-1] * self.code_feature_vocab_size, dtype=torch.long)) for example in examples] # Ensure default padding for missing features
            max_code_feature_len = max(len(indices) for indices in code_feature_indices_batch) if code_feature_indices_batch else 0
            padded_code_feature_indices_batch = []
            for indices in code_feature_indices_batch:
                padding_length = max_code_feature_len - len(indices)
                padded_indices = torch.cat([indices, torch.zeros(padding_length, dtype=torch.long)], dim=0) # Pad with 0
                padded_code_feature_indices_batch.append(padded_indices)
            batch["code_feature_indices"] = torch.stack(padded_code_feature_indices_batch) if padded_code_feature_indices_batch else torch.tensor([], dtype=torch.long)


        # Masking (same as before)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # Fiveg Feature Indices Processing (same as before)
        fiveg_feature_indices_batch = [example.get("fiveg_feature_indices", np.array([0] * self.fiveg_feature_vocab_size, dtype=np.int64)) for example in examples] # Ensure default padding for missing features
        max_fiveg_feature_len = max(len(indices) for indices in fiveg_feature_indices_batch) if fiveg_feature_indices_batch else 0
        padded_fiveg_feature_indices_batch = []
        for indices in fiveg_feature_indices_batch:
            padding_length = max_fiveg_feature_len - len(indices)
            padded_indices = np.concatenate([indices, np.zeros(padding_length, dtype=np.int64)], axis=0)
            padded_fiveg_feature_indices_batch.append(padded_indices)
        batch["fiveg_feature_indices"] = np.stack(padded_fiveg_feature_indices_batch) if padded_fiveg_feature_indices_batch else np.array([], dtype=np.int64)


        # Code Feature Indices Processing (NEW)
        if self.code_feature_vocab:
            code_feature_indices_batch = [example.get("code_feature_indices", np.array([-1] * self.code_feature_vocab_size, dtype=np.int64)) for example in examples] # Ensure default padding for missing features
            max_code_feature_len = max(len(indices) for indices in code_feature_indices_batch) if code_feature_indices_batch else 0
            padded_code_feature_indices_batch = []
            for indices in code_feature_indices_batch:
                padding_length = max_code_feature_len - len(indices)
                padded_indices = np.concatenate([indices, np.zeros(padding_length, dtype=np.int64)], axis=0) # Pad with 0
                padded_code_feature_indices_batch.append(padded_indices)
            batch["code_feature_indices"] = np.stack(padded_code_feature_indices_batch) if padded_code_feature_indices_batch else np.array([], dtype=np.int64)


        # Masking (same as before)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.astype(bool)

        probability_matrix[special_tokens_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            np.random.binomial(1, self.mask_replace_prob, size=labels.shape).astype(bool) & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob
        indices_random = (
            np.random.binomial(1, random_replace_prob_scaled, size=labels.shape).astype(bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


