import os
import gc
import argparse
import json
import logging

import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from utils.datautils import get_loaders
from utils.modelutils import build_fp16_model, build_tokenizer
from intactkv.utils import gen_bos_kv
from intactkv.inference_model import inject_intactkv_inference_model, inject_quantized_kv_model
from eval_utils.ppl import ppl_eval
from eval_utils.mmlu import run_mmlu_eval

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, choices=["ppl", "mmlu", "qa"], default=None, help="eval task name")
    parser.add_argument("--mmlu_num_few_shots", type=str, default="0,5", help="#MMLU few shots")
    parser.add_argument("--fp16_model_path", type=str, default=None, help="full-precision model path")
    parser.add_argument("--gptq_path", type=str, default="./modelzoo/autogptq", help="GPTQ model path")
    parser.add_argument("--awq_path", type=str, default="./modelzoo/llm-awq", help="AWQ model path")
    parser.add_argument("--quant_method", type=str, default="fp16", choices=["fp16", "rtn", "gptq", "awq"],
                        help="which quantization method to use")
    parser.add_argument("--intactkv", action="store_true", help="whether to use intactKV")
    parser.add_argument("--bits", type=int, default=3, help="weight #quantization bits")
    parser.add_argument("--kv_bits", type=int, default=16, help="KV cache #quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="group size of quantized model")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="output path")

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.tasks == "qa":
        task_names = [
            ###############  zero-shot tasks   #################
            "openbookqa",       # 500	acc, acc_norm
            "winogrande",       # 1267	acc
            "arc_challenge",    # 1172	acc, acc_norm
            "arc_easy",         # 2376	acc, acc_norm
            "boolq",            # 3270	acc
            "hellaswag",        # 10042	acc, acc_norm
            "lambada_openai",   # 5153	ppl, acc
        ]
    
    args.fp16_model_path = args.fp16_model_path[:-1] if args.fp16_model_path.endswith("/") else args.fp16_model_path
    args.model_name = args.fp16_model_path.split("/")[-1]
    args.model_type = "-".join(args.model_name.split("-")[:-1])
    if args.quant_method == "fp16":
        args.pretrained = args.fp16_model_path
        args.gptq_path = None
        args.awq_path = None
        args.intactkv = False
    elif args.quant_method == "rtn":
        args.pretrained = args.fp16_model_path
        args.gptq_path = None
        args.awq_path = None
    elif args.quant_method == "awq":
        args.pretrained = args.fp16_model_path
        args.gptq_path = None
        args.awq_path = os.path.join(args.awq_path, args.model_type, f"{args.model_name}-w{args.bits}-g{args.group_size}.pt")
    elif args.quant_method == "gptq":
        args.pretrained = os.path.join(args.gptq_path, args.model_type, f"{args.model_name}-{args.bits}bit-{args.group_size}g")
        args.gptq_path = f"gptq_model-{args.bits}bit-{args.group_size}g.safetensors"
        args.awq_path = None

    # output path
    if args.quant_method == "fp16":
        args.output_dir = os.path.join(args.output_dir, "fp16", args.model_name)
    else:
        args.output_dir = os.path.join(args.output_dir, args.quant_method, args.model_name, f"{args.bits}-bits",
                                       f"g{args.group_size}", "quantized_model")
    args.output_dir = os.path.join(args.output_dir, args.tasks)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.intactkv:
        model, tokenizer = build_fp16_model(args)
        intactkv, intactkv_ids = gen_bos_kv(model, tokenizer)
        with torch.no_grad():
            bos_ids = tokenizer(tokenizer.bos_token, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
            intactkv_logits = model(bos_ids).logits
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        intactkv, intactkv_ids, intactkv_logits = None, None, None

    if torch.cuda.device_count() > 1:
        from accelerate import Accelerator
        accelerator = Accelerator()
        # do naive model parallel for models larger than 60B, otherwise activate DDP
        parallelize = (accelerator.num_processes == 1)
    else:
        parallelize = False

    model = HFLM(
        args,
        pretrained=args.pretrained,
        dtype=torch.float16,
        quant_method=args.quant_method,
        gptq=args.gptq_path,
        awq=args.awq_path,
        gptq_use_triton=False,
        parallelize=parallelize,    
        use_fast_tokenizer=False,
    )
    model.tokenizer = build_tokenizer(args, model.model)

    # inject intactKV model
    model = inject_intactkv_inference_model(args, model, model.tokenizer, intactkv, intactkv_ids, intactkv_logits)

    # quantize KV cache
    if args.kv_bits < 16:
        model = inject_quantized_kv_model(args, model)

    for param in model.model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    model.model.eval()

    save_name = args.model_name + ("-intactkv" if args.intactkv else "")

    if args.tasks == "ppl":
        # evaluate ppl following gptq
        datasets = ['wikitext2', 'c4-new']
        results = {}
        for dataset in datasets:
            logging.info("loading and tokenizing dataset...")
            testloader = get_loaders(
                args, dataset, tokenizer=model.tokenizer, seqlen=2048,    # fix seq len
            )
            logging.info(dataset)
            ppl = ppl_eval(model.model, model.tokenizer, testloader)
            results[dataset] = ppl
    elif args.tasks == "mmlu":
        args.mmlu_num_few_shots = [int(n) for n in args.mmlu_num_few_shots.split(",")]
        for num_few_shots in args.mmlu_num_few_shots:
            save_dir = os.path.join(args.output_dir, f"{num_few_shots}-shot-{save_name}")
            run_mmlu_eval(model.model, model.tokenizer, args.model_name,
                          num_few_shots, "./datasets/mmlu/data", save_dir)
    else:
        results = evaluator.simple_evaluate(
            model=model,
            tasks=task_names,
            num_fewshot=0,
            batch_size=1,
        ) 

    if args.tasks != "mmlu" and results is not None:     # rank 0
        if args.tasks == "qa":
            # print summary table
            with open(os.path.join(args.output_dir, f"accuracies_{save_name}.md"), "w") as f:
                acc_table = evaluator.make_table(results)
                f.write(acc_table)
                print(acc_table)
        dumped = json.dumps(results, indent=2)
        with open(os.path.join(args.output_dir, f"results_{save_name}.json"), "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
