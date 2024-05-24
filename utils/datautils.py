import pickle
import os
import random
from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np
import torch
from datasets import load_dataset

from utils.sharegpt import make_sharegpt_data_module


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=True):
    if eval_mode:
        testdata = load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc.input_ids
    else:
        traindata = load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode=True):
    if eval_mode:
        testdata = load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc.input_ids
    else:
        traindata = load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode=True):
    if eval_mode:
        valdata = load_dataset(
            './datasets/allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc.input_ids
    else:
        traindata = load_dataset(
            './datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    args, name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, eval_mode=True
):
    cache_dir = os.path.join("./outputs", ".cache", args.model_type, name)
    os.makedirs(cache_dir, exist_ok=True)
    cached_dataset = os.path.join(cache_dir, "testset.pkl" if eval_mode else "trainset.pkl")
    if os.path.exists(cached_dataset):
        print("Loading cached tokenized dataset...")
        with open(cached_dataset, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode)
    if 'ptb' in name:
        dataset = get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode)
    if 'c4' in name:
        dataset = get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode)
    with open(cached_dataset, "wb") as f:
        print("Saving cached tokenized dataset...")
        pickle.dump(dataset, f)
    return dataset


@dataclass
class DataCollatorForCausalLM(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # TODO. support bs=1 only
        assert len(instances) == 1
        return instances[0]


def make_data_module(args, tokenizer):
    print("Loading dataset...")
    dataset = make_sharegpt_data_module(args, tokenizer)

    data_collator = DataCollatorForCausalLM()
    return dict(
        train_dataset=dataset,
        data_collator=data_collator
    )
