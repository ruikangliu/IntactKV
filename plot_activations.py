import os
import argparse
import functools

import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    OPTForCausalLM,
    MistralForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from fastchat.model.model_adapter import get_conversation_template

from utils.sharegpt import make_sharegpt_data_module
from utils.modelutils import build_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16_model_path', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--output_path', type=str, default='./outputs/visualizations',
                        help='where to save the visualization results')
    parser.add_argument('--dataset_path', type=str,
                        default="./datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json")
    parser.add_argument('--num_tokens', type=int, default=128)
    args = parser.parse_args()
    return args


@torch.no_grad()
def get_act(args, model, dataset, tokenizer):
    model.eval()
    device = next(model.parameters()).device
    abs_acts = {}

    if isinstance(model, LlamaForCausalLM):
        target_layer_type = LlamaDecoderLayer
    elif isinstance(model, OPTForCausalLM):
        target_layer_type = OPTDecoderLayer
    elif isinstance(model, MistralForCausalLM):
        target_layer_type = MistralDecoderLayer
    else:
        raise NotImplementedError

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        if name in abs_acts:
            abs_acts[name] = torch.cat([abs_acts[name], tensor.cpu()], dim=0)
        else:
            abs_acts[name] = tensor.cpu()

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        stat_tensor(name, y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, target_layer_type):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    i = 0
    tokens = 0
    while tokens < args.num_tokens:
        max_length = args.num_tokens - tokens
        input_ids = dataset[i]["input_ids"][:, :max_length].to(device)

        tokens += input_ids.shape[1]
        model(input_ids, output_attentions=True)
        i += 1

    for h in hooks:
        h.remove()

    return abs_acts


@torch.no_grad()
def get_attn(args, model, dataset):
    model.eval()
    device = next(model.parameters()).device

    attns_list = []
    for data in tqdm(dataset):
        input_ids = data["input_ids"][:, :args.num_tokens].to(device)
        output = model(input_ids, output_attentions=True)
        attns_list.append(output.attentions)

    attns_sum_pooling = [None] * model.config.num_hidden_layers
    for layer in range(model.config.num_hidden_layers):
        attns_sum_pooling[layer] = torch.cat([attns[layer].sum(dim=1) for attns in attns_list], dim=0).mean(0).cpu().numpy()

    return attns_sum_pooling


@torch.no_grad()
def get_kv_act(args, model, dataset):
    model.eval()
    device = next(model.parameters()).device
    acts = {}

    target_layer_type = nn.Linear
    target_layer_names = ["k_proj", "v_proj"]

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        if name in acts:
            acts[name] = torch.cat([acts[name], tensor.cpu()], dim=0)
        else:
            acts[name] = tensor.cpu()

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]
        stat_tensor(name, y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, target_layer_type) and \
            any([target_layer_name in name for target_layer_name in target_layer_names]):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    i = 0
    tokens = 0
    while tokens < args.num_tokens:
        max_length = args.num_tokens - tokens
        input_ids = dataset[i]["input_ids"][:, :max_length].to(device)

        tokens += input_ids.shape[1]
        model(input_ids, output_attentions=True)
        i += 1

    for h in hooks:
        h.remove()

    return acts


def plot_activations(args, name, abs_act: torch.Tensor):
    # set up the figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # plot activation
    n_tokens, hidden_dim = abs_act.shape
    x = np.arange(1, hidden_dim + 1)
    y = np.arange(1, n_tokens + 1)
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, abs_act, cmap='coolwarm',
                          rstride=1, cstride=1, linewidth=0.5, 
                          antialiased=True, zorder=1)
    # fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # plot separation line between prompt & query
    if "vicuna" in args.fp16_model_path:
        margin = 500
        x = np.arange(-margin, hidden_dim + margin)
        y = np.zeros_like(x) + args.prompt_len - 1
        z = np.zeros_like(x)
        ax1.plot(x, y, z, "r--", zorder=10)
    
    # ax1.set_title(f'Layer {name.split(".")[-1]}')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    # ax1.set_zlabel('Absolute Value')

    jpg_path = os.path.join(args.output_path, f'{name}.jpg')
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

    # convert jpg to pdf
    import img2pdf
    pdf_path = os.path.join(args.output_path, f'{name}.pdf')
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(jpg_path))
    os.remove(jpg_path)


def plot_attn_map(args, attns):
    for layer in tqdm(range(0, len(attns), 8)):
        # set up the figure and axes
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_facecolor("grey")

        # plot attn map
        attn = attns[layer]
        attn_mask = np.triu(np.ones_like(attn), k=1)
        attn[attn_mask == 1] = np.nan
        im = ax1.imshow(attn, cmap='coolwarm')
        # llama-2-7B
        # ax1.set_xticks([0, 12])
        # ax1.set_xticklabels(["[BOS]", "."])
        # llama-30B
        # ax1.set_xticks([0, 27])
        # ax1.set_xticklabels(["[BOS]", "'"])
        # ax1.set_xticks(range(args.num_tokens))
        # ax1.set_xticklabels(args.input_tokens, rotation=80, fontsize=3.5)
        plt.colorbar(im)

        # plot separation line
        if "vicuna" in args.fp16_model_path:
            ax1.axvline(args.prompt_len, linestyle="--", color="r")
        
        # ax1.set_title(f'Layer {layer}')

        plt.savefig(os.path.join(args.output_path, f'layer.{layer}.pdf'), bbox_inches='tight', pad_inches=0.0)
        plt.close()


def calc_smoothness(x):
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    x_std = x.std(dim=-1, keepdim=True)

    x_std = x_std.squeeze()
    absmax = x.abs().max(-1)[0]

    return absmax.mean(), x_std.mean()


@torch.no_grad()
def main(args):
    # build model
    config = AutoConfig.from_pretrained(args.fp16_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.fp16_model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    num_hidden_layers = model.config.num_hidden_layers

    tokenizer = build_tokenizer(args, model)

    # system prompt
    context = get_conversation_template("vicuna")
    context.append_message(context.roles[0], None)
    prompt = context.get_prompt()
    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    args.prompt_len = prompt_ids.shape[-1]

    # dataset
    args.max_seq_len = 2048
    args.max_train_samples = 1
    dataset = make_sharegpt_data_module(args, tokenizer, min_gpt_len=args.num_tokens, seed=1)

    # get activation
    args.num_tokens = 128
    abs_acts = get_act(args, model, dataset, tokenizer)

    # get attn
    args.num_tokens = 64    # for better visualization
    args.input_tokens = tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"].squeeze()[:args.num_tokens])
    attns = get_attn(args, model, dataset)

    # get kv activation
    args.num_tokens = 1024
    kv_acts = get_kv_act(args, model, dataset)

    del model
    torch.cuda.empty_cache()

    # output path
    model_name = args.fp16_model_path.split("/")[-1]
    output_path = args.output_path

    # plot attn map
    args.output_path = os.path.join(output_path, model_name, "attn")
    os.makedirs(args.output_path, exist_ok=True)
    plot_attn_map(args, attns)

    # plot activations
    args.output_path = os.path.join(output_path, model_name, "act")
    os.makedirs(args.output_path, exist_ok=True)

    for name, abs_act in tqdm(abs_acts.items()):
        if int(name.split(".")[-1]) in range(0, num_hidden_layers, 8):
            plot_activations(args, name, abs_act)

    # smoothness of kv cache
    if "llama-7b" in model_name:
        pivot_token_ids = [0]
    elif "llama-13b" in model_name:
        pivot_token_ids = [0]
    elif "llama-2-7b" in model_name:
        pivot_token_ids = [0, 12]
    elif "llama-2-13b" in model_name:
        pivot_token_ids = [0]
    elif "llama-3-8b" in model_name:
        pivot_token_ids = [0]
    else:
        return
    for name in ["k", "v"]:
        pivot_range_list, pivot_std_list = [], []
        non_pivot_range_list, non_pivot_std_list = [], []
        for layer in range(config.num_hidden_layers):
            acts = kv_acts[f"model.layers.{layer}.self_attn.{name}_proj"]
            pivot_mask = np.array([False] * acts.shape[0])
            pivot_mask[pivot_token_ids] = True
            pivot_acts = acts[pivot_mask, :]
            non_pivot_acts = acts[~pivot_mask, :]
            pivot_absmax, pivot_std = calc_smoothness(pivot_acts)
            non_pivot_absmax, non_pivot_std = calc_smoothness(non_pivot_acts)
            pivot_range_list.append(pivot_absmax)
            pivot_std_list.append(pivot_std)
            non_pivot_range_list.append(non_pivot_absmax)
            non_pivot_std_list.append(non_pivot_std)
        pivot_absmax = sum(pivot_range_list) / len(pivot_range_list)
        pivot_std = sum(pivot_std_list) / len(pivot_std_list)
        non_pivot_absmax = sum(non_pivot_range_list) / len(non_pivot_range_list)
        non_pivot_std = sum(non_pivot_std_list) / len(non_pivot_std_list)
        print(f"pivot tokens {name} cache -- absmax/std: {pivot_absmax:.2f}/{pivot_std:.2f}\n"
                f"non-pivot tokens {name} cache -- absmax/std: {non_pivot_absmax:.2f}/{non_pivot_std:.2f}")
        


if __name__ == '__main__':
    args = parse_args()
    main(args)
