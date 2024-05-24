import os
import gc
import re
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
import transformers
from transformers import set_seed

from utils.datautils import make_data_module
from utils.modelutils import build_fp16_model
from intactkv.utils import gen_vicuna_prompt_kv, get_acts_list
from intactkv.train_model import get_quantized_model
from intactkv_train import ModelArguments, DataArguments, TrainingArguments


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_mse_loss(args):
    if "llama-2-7b" in args.model_name:
        x_ticks = [1, 13]
    elif "llama-2-13b" in args.model_name:
        x_ticks = [1, 28]
    elif "llama-2-70b" in args.model_name:
        x_ticks = [1, 28]
    elif "llama-30b" in args.model_name:
        x_ticks = [1, 28]
    else:
        x_ticks = [1]

    last_block_losses = []
    with open(args.output_path, "r") as f:
        for line in f.readlines():
            losses = re.findall(r"\d+\.\d+", line)
            total_loss = float(losses[0])
            last_block_loss = float(losses[1])
            last_block_losses.append(last_block_loss)
    assert len(last_block_losses) == args.intactkv_size + 1

    x = np.arange(args.intactkv_size + 1)
    y1 = last_block_losses

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid()
    ax1.plot(x, y1, linewidth=2, marker="o", linestyle="-", label="Transformer Layer")
    font_size = 20
    ax1.set_xticks(x_ticks)   # set x ticks
    ax1.tick_params(axis="x", labelsize=font_size - 2)
    ax1.tick_params(axis="y", labelsize=font_size - 2)
    # from matplotlib.ticker import MaxNLocator
    # ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.set_xlabel('Size of IntactKV', fontsize=font_size)         # set labels
    ax1.set_ylabel('Transformer Layer Loss', fontsize=font_size)
    ax1.legend(loc="upper right")

    plt.savefig(os.path.join(args.output_dir, "intactkv_size_mse.pdf"), bbox_inches='tight')
    plt.close()


def main():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # output path
    args.fp16_model_path = args.fp16_model_path[:-1] if args.fp16_model_path.endswith("/") else args.fp16_model_path
    model_name = args.fp16_model_path.split("/")[-1]
    args.model_name = model_name
    args.model_type = "-".join(model_name.split("-")[:-1])
    args.gptq_path = os.path.join(args.gptq_path, args.model_type, f"{model_name}-{args.bits}bit-{args.group_size}g")
    args.awq_path = os.path.join(args.awq_path, args.model_type, f"{model_name}-w{args.bits}-g{args.group_size}.pt")
    assert args.quant_method in ["rtn"]
    args.output_dir = os.path.join(args.output_dir, args.quant_method, model_name, f"{args.bits}-bits", f"g{args.group_size}",
                                   f"quantized_model", "mse")
    os.makedirs(args.output_dir, exist_ok=True)

    # fp16 model and tokenizer
    model, tokenizer = build_fp16_model(args)

    # dataset
    data_module = make_data_module(args, tokenizer)
    train_dataset = data_module["train_dataset"]
    collate_fn = data_module["data_collator"]

    # get full-precision activations
    logging.info("Getting full-precision activations of trainset...")
    acts_list = get_acts_list(model, train_dataset, collate_fn)

    # get fp16 prompt kv cache
    logging.info("Getting intactKV...")
    intactkv_list, intactkv_ids_list = [], []
    for i in range(args.intactkv_size + 1):
        intactkv, intactkv_ids = gen_vicuna_prompt_kv(model, tokenizer, intactkv_size=i)
        intactkv_list.append(intactkv)
        intactkv_ids_list.append(intactkv_ids)

    # clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    args.output_path = os.path.join(args.output_dir, f"intactkv_size_{args.intactkv_size}.txt")
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    for i in range(args.intactkv_size + 1):
        logging.info(f"Evaluating IntactKV of size {i}...")
        intactkv, intactkv_ids = intactkv_list[i], intactkv_ids_list[i]
        # get quantized model
        logging.info("Loading quantized model...")
        model = get_quantized_model(args, tokenizer, acts_list, intactkv, intactkv_ids)
        model.quant_method = args.quant_method

        # fp16 inference
        for param in model.parameters():
            if param.dtype == torch.float32 or param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

        set_seed(args.seed)
        
        # calculate MSE loss
        fp16_acts_list = acts_list
        print("Getting quant activations of evalset...")
        quant_acts_list = get_acts_list(model, train_dataset, collate_fn)
        n_layers = model.model.config.num_hidden_layers
        losses = []
        last_block_losses = []
        model.eval()
        with tqdm(zip(quant_acts_list, fp16_acts_list), total=len(train_dataset)) as tbar:
            for quant_act, fp16_act in tbar:
                gpt_masks = torch.arange(next(iter(fp16_act.values())).shape[1])[args.intactkv_size:] - i
                total_loss, losses_dict = model.compute_ptq_loss(fp16_act, quant_act, gpt_masks, return_losses_dict=True)
                last_block_loss = losses_dict[f"layers.{n_layers - 1}"]
                last_block_losses.append(last_block_loss)
                loss = 0
                for layer in range(n_layers):
                    loss += losses_dict[f"layers.{layer}"]
                losses.append(loss)
                tbar.update()
                tbar.set_postfix(loss=(sum(losses) / len(losses)).item())
        last_block_losses = sum(last_block_losses) / len(last_block_losses)
        losses = sum(losses) / len(losses)
        result = f"Eval set MSE loss (intactkv_size {i}) - total: [{losses.item()}], last_block: [{last_block_losses.item()}]"
        print(result)
        with open(args.output_path, "a") as f:
            f.write(f"{result}\n")

    plot_mse_loss(args)


if __name__ == "__main__":
    main()
