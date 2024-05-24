import functools
from tqdm import tqdm

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from fastchat.model.model_adapter import get_conversation_template


@torch.no_grad()
def gen_bos_kv(model, tokenizer):
    model.eval()
    device = next(model.parameters()).device

    prompt = tokenizer.bos_token
    input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    outputs = model(input_ids=input_ids)
    intactkv = []
    # offload intactkv to cpu
    for layer_kv in outputs.past_key_values:
        cpu_layer_kv = []
        for cache in layer_kv:
            cpu_layer_kv.append(cache.detach().cpu())
        intactkv.append(tuple(cpu_layer_kv))
    intactkv = tuple(intactkv)

    return intactkv, input_ids


@torch.no_grad()
def gen_vicuna_prompt_kv(model, tokenizer, intactkv_size=None):
    model.eval()
    device = next(model.parameters()).device

    context = get_conversation_template("vicuna")
    context.append_message(context.roles[0], None)
    prompt = context.get_prompt()
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # prepend <bos>
    if not (input_ids[:, 0] == tokenizer.bos_token_id).all():
        bos_ids = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        input_ids = torch.cat([bos_ids, input_ids], dim=1).contiguous()

    if intactkv_size is not None:
        if intactkv_size == 0:
            return None, None
        input_ids = input_ids[:, :intactkv_size]

    outputs = model(input_ids=input_ids)
    intactkv = []
    # offload kv to cpu
    for layer_kv in outputs.past_key_values:
        cpu_layer_kv = []
        for cache in layer_kv:
            cpu_layer_kv.append(cache.detach().cpu())
        intactkv.append(tuple(cpu_layer_kv))
    intactkv = tuple(intactkv)

    return intactkv, input_ids


@torch.no_grad()
def get_acts_list(model, dataset, collate_fn=None):
    model.eval()
    device = next(model.parameters()).device
    acts_list = []
    acts = {}
    nsamples = 0

    # full-precision model
    if isinstance(model, LlamaForCausalLM):
        target_modules = [LlamaDecoderLayer]
    else:
        raise NotImplementedError
    hook_model = model
    
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        # remove prefix
        if name.find("layers.") != -1:
            name = name[name.find("layers."):]
        if "qkv_proj" in name:
            # split qkv_proj into q_proj, k_proj, v_proj
            layer_name = ".".join(name.split(".")[:-1])
            hidden_size = m.linear_module.in_features
            query_states, key_states, value_states = torch.split(y, hidden_size, dim=2)
            acts[layer_name + ".q_proj"] = query_states
            acts[layer_name + ".k_proj"] = key_states
            acts[layer_name + ".v_proj"] = value_states
        else:
            acts[name] = y

    hooks = []
    for name, m in hook_model.named_modules():
        if any([isinstance(m, target_module) for target_module in target_modules]):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for data in tqdm(dataset):
        if collate_fn:
            x = collate_fn([data])
        else:
            x = data
        model(input_ids=x["input_ids"].to(device), attention_mask=x["attention_mask"].to(device))
        # offload activations to CPU
        for name, act in acts.items():
            acts[name] = act.detach().cpu()
        acts_list.append(acts)
        acts = {}
        nsamples += 1

    for h in hooks:
        h.remove()

    return acts_list
