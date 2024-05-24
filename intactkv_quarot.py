import os
import json

import torch
import transformers

import quarot.utils as utils
import quarot.model_utils as model_utils
import quarot.quant_utils as quant_utils
import quarot.rotation_utils as rotation_utils
import quarot.gptq_utils as gptq_utils
import quarot.hadamard_utils as hadamard_utils
import quarot.intactkv_utils as intactkv_utils

from utils.datautils import get_loaders
from intactkv.inference_model import inject_intactkv_inference_model
from eval_utils.ppl import ppl_eval


def main():
    args = utils.parser_gen()

    if args.intactkv:
        intactkv_dict = intactkv_utils.get_intactkv(args)
        intactkv, intactkv_ids, intactkv_logits = \
            intactkv_dict["intactkv"], intactkv_dict["intactkv_ids"], intactkv_dict["intactkv_logits"]
        
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
      
    if args.w_bits < 16:
        save_dict = {}

        if not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model, "Only llama is supported for GPTQ!"
            gptq_path = os.path.join(args.output_dir, "gptq.pth")
            if os.path.exists(gptq_path):
                print(f"Load quantized model from {gptq_path}")
                save_dict = torch.load(gptq_path)
                model.load_state_dict(save_dict["model"])
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
                trainloader = get_loaders(
                    args, args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, tokenizer=tokenizer,
                    seqlen=model.seqlen, eval_mode=False
                )
                quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
                save_dict["w_quantizers"] = quantizers
                save_dict["model"] = model.state_dict()
                print(f"Save quantized model to {gptq_path}")
                torch.save(save_dict, gptq_path)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

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
        
    # Evaluating on dataset
    save_name = args.model_name + ("-intactkv" if args.intactkv else "")
    datasets = ['wikitext2', 'c4-new']
    results = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    model.to(utils.DEV)
    if args.intactkv:
        model = inject_intactkv_inference_model(args, model, tokenizer, intactkv, intactkv_ids, intactkv_logits)
    for dataset in datasets:
        print("loading and tokenizing dataset...")
        testloader = get_loaders(
            args, dataset, tokenizer=tokenizer, seqlen=2048,    # fix seq len
        )
        print(dataset)
        ppl = ppl_eval(model, tokenizer, testloader)
        results[dataset] = ppl
    model.cpu()

    dumped = json.dumps(results, indent=2)
    with open(os.path.join(args.output_dir, f"results_{save_name}.json"), "w") as f:
        f.write(dumped)


if __name__ == '__main__':
    main()