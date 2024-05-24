import argparse
import pprint
import torch
import os
from datetime import datetime
import logging

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


supported_models = [
            './modelzoo/llama/llama-7b',
            './modelzoo/llama/llama-13b',
            './modelzoo/llama-2/llama-2-7b',
            './modelzoo/llama-2/llama-2-13b',
            './modelzoo/llama-2/llama-2-70b',
            './modelzoo/llama-3/llama-3-8b'
            ]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--hf_token', type=str, default=None)

    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')

    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # IntactKV
    parser.add_argument('--intactkv', action=argparse.BooleanOptionalAction, default=False)

    # Experiments Arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="output path")

    args = parser.parse_args()
    
    assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'

    # output path
    args.model_name = args.model.split("/")[-1]
    args.model_type = "-".join(args.model_name.split("-")[:-1])
    args.quant_method = "quarot"
    args.cache_dir = os.path.join(args.output_dir, args.quant_method, args.model_name, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.quant_method, args.model_name, f"w{args.w_bits}a{args.a_bits}")
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
