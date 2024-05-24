import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaDecoderLayer, \
                                                     apply_rotary_pos_emb, repeat_kv
from lm_eval.models.huggingface import HFLM

from utils.quantizer import UniformAffineQuantizer


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


class IntactKVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, model: LlamaForCausalLM, tokenizer,
                 intactkv, intactkv_ids, intactkv_logits):
        super(LlamaForCausalLM, self).__init__(model.config)
        self.model = model.model
        self.lm_head = model.lm_head
        self.config = model.config
        if hasattr(model, "seqlen"):
            self.seqlen = model.seqlen

        # intactKV
        self.intactkv = intactkv
        self.intactkv_ids = intactkv_ids
        self.intactkv_logits = intactkv_logits
        if intactkv is not None:
            self.intactkv_len = self.intactkv_ids.shape[-1]
            dtype = next(self.model.parameters()).dtype
            # move intactKV to GPU
            self.intactkv = []
            for i, layer_kv in enumerate(intactkv):
                gpu_layer_kv = []
                device = next(self.model.layers[i].parameters()).device
                for cache in layer_kv:
                    gpu_layer_kv.append(cache.to(device).to(dtype))
                self.intactkv.append(tuple(gpu_layer_kv))
            self.intactkv = tuple(self.intactkv)
            self.intactkv_logits = self.intactkv_logits.to(device)
        else:
            device = next(self.model.parameters()).device
            self.bos_ids = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # TODO. support bs=1 only
        assert input_ids.shape[0] == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_len = input_ids.shape[1]
        activate_intactkv = (past_key_values is None and self.intactkv is not None)
        if activate_intactkv:
            if (input_ids[:, :self.intactkv_len] == self.intactkv_ids.to(input_ids.device)).all():
                input_ids = input_ids[:, self.intactkv_len:]
            past_key_values = self.intactkv
        else:
            # prepend <bos>
            if not (input_ids[:, :1] == self.bos_ids).all():
                input_ids = torch.cat([self.bos_ids, input_ids], dim=1).contiguous()
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if activate_intactkv:
            logits = torch.cat([self.intactkv_logits.to(logits), logits], dim=1)
        logits = logits.float()
        logits = logits[:, -input_len:, :]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
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


def inject_intactkv_inference_model(args, model, tokenizer, intactkv, intactkv_ids, intactkv_logits):
    # locate causal LM
    if isinstance(model, HFLM):
        if args.quant_method == "gptq":
            causal_lm = model.model.model
        else:
            causal_lm = model.model
    else:
        causal_lm = model

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # intactKV model
    intactkv_params = (causal_lm, tokenizer, intactkv, intactkv_ids, intactkv_logits)
    if isinstance(causal_lm, LlamaForCausalLM):
        causal_lm = IntactKVLlamaForCausalLM(*intactkv_params)
    else:
        raise NotImplementedError
    
    # inject model
    if isinstance(model, HFLM):
        if args.quant_method == "gptq":
            model.model.model = causal_lm
        else:
            model._model = causal_lm
    else:
        model = causal_lm
    
    return model


class LlamaAttentionKVQuantized(LlamaAttention):
    def __init__(self, args, m: LlamaAttention):
        super().__init__(m.config)
        self.layer_idx = m.layer_idx
        
        self.q_proj = m.q_proj
        self.k_proj = m.k_proj
        self.v_proj = m.v_proj
        self.o_proj = m.o_proj
        self.rotary_emb = m.rotary_emb

        # KV cache quantizer
        self.quant_params = {
            "n_bits": args.kv_bits,
            "symmetric": False,
        }
        self.k_quantizer = UniformAffineQuantizer(**self.quant_params)
        self.v_quantizer = UniformAffineQuantizer(**self.quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # quantize KV cache
        key_states = self.k_quantizer(key_states)
        value_states = self.v_quantizer(value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def inject_quantized_kv_model(args, model: HFLM):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # TODO. support pseudo-quantized model only
    if args.quant_method == "gptq":
        raise NotImplementedError
    
    causal_lm = model.model

    # intactKV model
    if isinstance(causal_lm, LlamaForCausalLM):
        for name, m in causal_lm.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                m.self_attn = LlamaAttentionKVQuantized(args, m.self_attn)
    else:
        raise NotImplementedError
    
    model._model = causal_lm
    
    return model
