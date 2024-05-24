import os

import torch
import transformers

import quarot.utils as utils
import quarot.model_utils as model_utils
import quarot.quant_utils as quant_utils
import quarot.rotation_utils as rotation_utils
import quarot.hadamard_utils as hadamard_utils
from intactkv.utils import gen_bos_kv


def get_intactkv(args):
    print("Generating IntactKV...")
    intactkv_dict_path = os.path.join(args.cache_dir, "intactkv.pth")
    if os.path.exists(intactkv_dict_path):
        return torch.load(intactkv_dict_path)

    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'down_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'o_proj' in name:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.k_bits < 16:
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                        "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn, 
                        rope_function_name, 
                        config=model.config,
                        **k_quant_config)

    model.to(utils.DEV)
    quant_utils.set_quantizer_state(model, activate_quantizer=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    intactkv, intactkv_ids = gen_bos_kv(model, tokenizer)
    with torch.no_grad():
        bos_ids = tokenizer(tokenizer.bos_token, return_tensors='pt',
                            add_special_tokens=False).input_ids.to(model.device)
        intactkv_logits = model(bos_ids).logits
    model.cpu()
    quant_utils.set_quantizer_state(model, activate_quantizer=True)
    utils.cleanup_memory(verbos=True)

    intactkv_dict = {
        "intactkv": intactkv,
        "intactkv_ids": intactkv_ids,
        "intactkv_logits": intactkv_logits,
    }
    torch.save(intactkv_dict, intactkv_dict_path)

    return intactkv_dict
