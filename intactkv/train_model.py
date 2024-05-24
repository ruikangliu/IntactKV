import functools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast

from auto_gptq import AutoGPTQForCausalLM
from llm_awq.quantize.pre_quant import apply_awq
from llm_awq.quantize.quantizer import pseudo_quantize_model_weight

from utils.modelutils import auto_dispatch_model


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


class LlamaForPTQLoss(LlamaForCausalLM):
    def __init__(self, args, model: LlamaForCausalLM, tokenizer,
                 acts_list, intactkv, intactkv_ids):
        super(LlamaForCausalLM, self).__init__(model.config)

        self.model = model.model
        self.vocab_size = model.vocab_size
        self.lm_head = model.lm_head
        self.tokenizer = tokenizer

        self.args = args
        self.fp16_acts_list = acts_list

        # log
        if hasattr(args, "writer"):
            self.step = 0
            self.writer = args.writer
            self.gradient_accumulation_steps = args.gradient_accumulation_steps
            self.log_history = {
                "loss/total": [],
                "loss/last_block": [],
            }

        # intactkv
        self.intactkv = intactkv
        self.intactkv_ids = intactkv_ids
        self.intactkv_len = 0
        if intactkv is not None:
            self.intactkv_len = self.intactkv_ids.shape[-1]
            dtype = next(self.model.parameters()).dtype
            # move intactKV to GPU
            self.intactkv = []
            for i, layer_kv in enumerate(intactkv):
                gpu_layer_kv = []
                device = next(self.model.layers[i].parameters()).device
                for cache in layer_kv:
                    gpu_layer_kv.append(cache.unsqueeze(0).to(device).to(dtype))
                gpu_layer_kv = torch.cat(gpu_layer_kv, dim=0)
                self.intactkv.append(gpu_layer_kv)
            self.intactkv = nn.ParameterList(self.intactkv)

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
        data_index: Optional[dict] = None,
        gpt_masks: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # add hook to get activation
        acts = {}

        def stat_input_hook(m, x, y, name):
            if isinstance(m, LlamaDecoderLayer):
                y = y
            acts[name] = y[0]

        target_modules = [LlamaDecoderLayer]
        hooks = []
        for name, m in self.model.named_modules():
            # remove prefix
            name = name[name.find("layers."):]
            if any([isinstance(m, target_module) for target_module in target_modules]):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_input_hook, name=name))
                )

        # intactkv
        activate_intactkv = (past_key_values is None and self.intactkv is not None)
        if activate_intactkv:
            if (input_ids[:, :self.intactkv_len] == self.intactkv_ids.to(input_ids.device)).all():
                input_ids = input_ids[:, self.intactkv_len:]
            past_key_values = self.get_past_key_values()

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
        logits = logits.float()

        fp16_acts = self.fp16_acts_list[data_index] if data_index is not None else None
        if fp16_acts is not None:
            if self.intactkv is not None:
                gpt_masks = gpt_masks - self.intactkv_len
            loss, losses_dict = self.compute_ptq_loss(fp16_acts, acts, gpt_masks, return_losses_dict=True)

            self.log_history["loss/total"].append(loss)
            self.log_history["loss/last_block"].append(losses_dict[f"layers.{self.config.num_hidden_layers - 1}"])
            if self.writer is not None:
                if self.step % self.gradient_accumulation_steps == self.gradient_accumulation_steps - 1:
                    for name, value_history in self.log_history.items():
                        self.writer.add_scalar(name, sum(value_history) / len(value_history), self.step // self.gradient_accumulation_steps)
                        self.log_history[name] = []
            self.step += 1
        else:
            loss = None

        for h in hooks:
            h.remove()

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
    
    def get_past_key_values(self):
        past_key_values = [layer_kv.clone() for layer_kv in self.intactkv]
        past_key_values = tuple(past_key_values)
        return past_key_values

    def compute_ptq_loss(self, fp16_acts, quantized_acts, gpt_masks=None, return_losses_dict=False):
        loss_device = next(self.parameters()).device
        losses = 0
        losses_dict = {}
        for name in fp16_acts.keys():
            device = loss_device if quantized_acts[name].device == torch.device("cpu") else quantized_acts[name].device
            fp16_act = fp16_acts[name].to(device).float()
            quantized_act = quantized_acts[name].to(device).float()
            if gpt_masks is not None:
                gpt_masks = gpt_masks.to(device)
            if name == "lm_head":
                # logit distillation
                pass
            else:
                if self.intactkv is not None:
                    fp16_act = fp16_act[:, self.intactkv_len:, :]
                # gather acts of tokens generated by gpt
                if gpt_masks is not None:
                    fp16_act = fp16_act[:, gpt_masks, :]
                    quantized_act = quantized_act[:, gpt_masks, :]
                # mse loss
                mse = (quantized_act - fp16_act) ** 2
                act_loss = mse.mean() / 2
                loss = act_loss.to(loss_device)
            losses += loss
            losses_dict[name] = loss
        if return_losses_dict:
            return losses, losses_dict
        else:
            return losses


def get_pseudo_quantized_model(args, model_path, tokenizer, acts_list, intactkv, intactkv_ids):
    # simulated pseudo quantization
    print(f"Building pseudo-quantized model from {model_path}...")

    # all hf model
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation_internal = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, device_map="cpu", torch_dtype=torch.float16
    )
    model.eval()

    if args.quant_method == "awq":
        print(f"Loading pre-computed AWQ results from {args.awq_path}...")
        awq_results = torch.load(args.awq_path, map_location="cpu")
        apply_awq(model, awq_results)

        # weight quantization
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": args.group_size,  # whether to use group quantization
        }
        pseudo_quantize_model_weight(model, w_bit=args.bits, q_config=q_config)
    elif args.quant_method == "rtn":
        # weight quantization
        q_config = {
            "zero_point": True,  # by default True
            "q_group_size": args.group_size,  # whether to use group quantization
        }
        pseudo_quantize_model_weight(model, w_bit=args.bits, q_config=q_config)

    # freeze base model's layers
    for name, param in model.named_parameters():
        param.requires_grad = False

    model = auto_dispatch_model(model)

    # inject ptq-loss model
    # TODO. supports LLaMA only
    if isinstance(model, LlamaForCausalLM):
        model = LlamaForPTQLoss(args, model, tokenizer, acts_list, intactkv, intactkv_ids)
    else:
        raise NotImplementedError(f"PTQ loss not implemented for {type(model)}")
    model.eval()

    return model


def get_gptq_model(args, tokenizer, acts_list, intactkv, intactkv_ids):
    # load quantized model
    print(f'Building GPTQ model from {args.gptq_path}...')
    model = AutoGPTQForCausalLM.from_quantized(
        args.gptq_path,
        device_map='auto',
        inject_fused_attention=False,
        inject_fused_mlp=False,
        use_triton=False,
        warmup_triton=False,
        trainable=False,
    )
    device_map = model.hf_device_map
    model.config.torch_dtype = torch.float16
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # inject ptq-loss model
    # TODO. supports LLaMA only
    if isinstance(model.model, LlamaForCausalLM):
        model.model = LlamaForPTQLoss(args, model.model, tokenizer, acts_list, intactkv, intactkv_ids)
    else:
        raise NotImplementedError(f"PTQ loss not implemented for {type(model.model)}")
    model.model.quantize_config = model.quantize_config
    model.model.hf_device_map = device_map
    model.eval()

    return model


def get_quantized_model(args, tokenizer, acts_list, intactkv, intactkv_ids):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model_params = (tokenizer, acts_list, intactkv, intactkv_ids)
    if args.quant_method == "gptq":
        model = get_gptq_model(args, *model_params)
    else:
        model_path = args.fp16_model_path
        model = get_pseudo_quantized_model(args, model_path, *model_params)
    return model
