import argparse
import gc
import json
import os
import random
import time
import shortuuid
from tqdm import tqdm

import torch
from transformers import set_seed
from fastchat.model import get_conversation_template

from utils.modelutils import build_fp16_model, print_parameters_dtype
from intactkv.utils import gen_bos_kv, gen_vicuna_prompt_kv
from intactkv.train_model import get_quantized_model


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}


def run_eval(
    args,
    model_path,
    model_id,
    question_file,
    answer_file,
    max_new_token,
    num_choices,
):
    questions = load_questions(question_file)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                args,
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
            )
        )


@torch.inference_mode()
def get_model_answers(
    args,
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
):
    # fp16 model and tokenizer
    args.fp16_model_path = model_path
    model, tokenizer = build_fp16_model(args)

    # get intactkv
    intactkv, intactkv_ids = None, None
    if args.intactkv:
        print("Getting intactkv_[P]...")
        intactkv, intactkv_ids = gen_vicuna_prompt_kv(model, tokenizer)

    if args.quant_method != "fp16":
        # clean up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # for compatibility
        args.writer = None
        args.gradient_accumulation_steps = None

        if args.intactkv_path is not None:
            print("Loading intactkv from checkpoint.")
            loaded_intactkv = torch.load(os.path.join(args.intactkv_path, "intactkv.pth"))
            list_intactkv = []
            for layer, layer_kv in enumerate(loaded_intactkv):
                list_intactkv.append(tuple([layer_kv[0], layer_kv[1]]))
            intactkv = tuple(list_intactkv)

        model = get_quantized_model(
            args=args,
            tokenizer=tokenizer,
            acts_list=None,
            intactkv=intactkv,
            intactkv_ids=intactkv_ids,
        )
    
    # fp16 training
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    # Verifying the datatypes
    print_parameters_dtype(model)

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                # try:
                output_ids = model.generate(
                    inputs=torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
                # except RuntimeError as e:
                #     print("ERROR question ID: ", question["question_id"])
                #     output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--bench_name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max_new_token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num_choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs", help="output path")
    parser.add_argument("--quant_method", type=str, default="fp16", choices=["fp16", "rtn", "gptq", "awq"], help="which quantization method to use")
    parser.add_argument("--gptq_path", type=str, default="./modelzoo/autogptq", help="path to GPTQ model")
    parser.add_argument("--awq_path", type=str, default="./modelzoo/llm-awq", help="path to AWQ model")
    parser.add_argument("--bits", type=int, default=3, help="#quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="group size of quantized model")
    parser.add_argument("--intactkv", action="store_true", help="whether to use intactKV")
    parser.add_argument("--intactkv_path", type=str, default=None, help="path to calibrated intactkv")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    question_file = f"fastchat/data/{args.bench_name}/question.jsonl"

    # output path
    args.model_name = args.model_path.split("/")[-1]
    args.model_type = "-".join(args.model_name.split("-")[:-1])
    args.gptq_path = os.path.join(args.gptq_path, args.model_type, f"{args.model_name}-{args.bits}bit-{args.group_size}g")
    args.awq_path = os.path.join(args.awq_path, args.model_type, f"{args.model_name}-w{args.bits}-g{args.group_size}.pt")
    if args.quant_method == "fp16":
        args.output_dir = os.path.join(args.output_dir, args.quant_method, args.model_name)
    else:
        args.output_dir = os.path.join(args.output_dir, args.quant_method, args.model_name, f"{args.bits}-bits", f"g{args.group_size}")
        if args.intactkv_path:
            args.output_dir = os.path.join(args.output_dir, args.intactkv_path)
            max_step = 0
            for filename in os.listdir(args.output_dir):
                if os.path.isdir(os.path.join(args.output_dir, filename)) and filename.startswith('checkpoint'):
                    max_step = max(max_step, int(filename.replace('checkpoint-', '')))
            args.intactkv_path = os.path.join(args.output_dir, f'checkpoint-{max_step}')
            print(f"Loading calibrated intactkv from {args.intactkv_path}...")
            args.output_dir = os.path.join(args.output_dir, "eval")
        else:
            args.output_dir = os.path.join(args.output_dir, "quantized_model")
    args.output_dir = os.path.join(args.output_dir, args.bench_name)

    # answer file name
    args.model_id = args.model_name
    json_name = args.model_id
    if args.quant_method != "fp16":
        json_name += f"-{args.quant_method}-w{args.bits}g{args.group_size}" + ("-intactkv" if args.intactkv else "")
    json_name += f"-seed_{args.seed}"
    answer_file = os.path.join(args.output_dir, f"{json_name}.jsonl")
    if os.path.exists(answer_file):
        os.remove(answer_file)

    print(f"Output to {answer_file}")

    # set seed
    set_seed(args.seed)

    run_eval(
        args=args,
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
    )

    reorg_answer_file(answer_file)
