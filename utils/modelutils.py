import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import (
    infer_auto_device_map,
    dispatch_model,
)
from accelerate.utils.modeling import get_balanced_memory

from llm_awq.quantize.pre_quant import apply_scale

from intactkv.utils import gen_bos_kv
from intactkv.inference_model import inject_intactkv_inference_model


def build_tokenizer(args, model):
    tokenizer = AutoTokenizer.from_pretrained(
        args.fp16_model_path,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        if model.config.pad_token_id is not None:
            tokenizer.pad_token_id = model.config.pad_token_id
        elif model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer

    
def auto_dispatch_model(model):
    kwargs = {
        "max_memory": get_balanced_memory(
            model, None
        )
    }
    device_map = infer_auto_device_map(
        model,
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "MistralDecoderLayer",
        ],
        **kwargs,
    )
    model = dispatch_model(model, device_map=device_map)
    return model


def build_fp16_model(args):
    config = AutoConfig.from_pretrained(args.fp16_model_path)
    config._attn_implementation_internal = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.fp16_model_path,
        config=config,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    tokenizer = build_tokenizer(args, model)

    # apply AWQ scale
    if args.quant_method == "awq":
        print("Loading pre-computed AWQ results from", args.awq_path)
        awq_results = torch.load(args.awq_path, map_location="cpu")
        apply_scale(model, awq_results["scale"])
    model = auto_dispatch_model(model)

    # prepend bos token to input
    model = inject_intactkv_inference_model(args, model, tokenizer, None, None, None)
    
    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")


def print_parameters_dtype(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: 
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v/total)
