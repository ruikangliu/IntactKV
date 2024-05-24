import json
import random
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers

from fastchat.model.model_adapter import get_conversation_template


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_len,
) -> Dict:
    conv = get_conversation_template("vicuna")
    system_prompt = conv.system_template.format(system_message=conv.system_message)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    assert len(sources) == 1
    source = sources[0]

    # Skip to the first human prompt
    start = 0
    while roles[source[start]["from"]] != conv.roles[0]:
        start += 1
    source = source[start:]
    assert len(source) > 0

    # get multi-round conversation string
    messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{j}, {role}, {conv.roles[j % 2]}"
        # Apply prompt templates
        message = system_prompt + conv.sep if j == 0 else ""
        if sentence["value"]:
            if role == roles["human"]:
                message += f'{roles["human"]}: {sentence["value"]} {roles["gpt"]}:'
            else:
                message += f'{sentence["value"]}'
        messages.append((role, message))

    # tokenize multi-round conversation
    bos_token_tensor = tokenizer(tokenizer.bos_token, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = [bos_token_tensor]
    gpt_masks = []
    sentence_total_len = 1  # bos token
    for role, message in messages:
        sentence_input_ids = tokenizer(
            message,
            return_tensors="pt",
            max_length=min(tokenizer.model_max_length, 4096),
            truncation=True,
            add_special_tokens=False,
        ).input_ids
        input_ids.append(sentence_input_ids)
        # Only compute loss on the assistant outputs.
        sentence_len = sentence_input_ids.shape[-1]
        if role == roles["gpt"]:
            gpt_mask = torch.arange(sentence_len) + sentence_total_len
            gpt_masks.append(gpt_mask)
        sentence_total_len += sentence_len
    input_ids = torch.cat(input_ids, dim=-1).contiguous()
    gpt_masks = torch.cat(gpt_masks, dim=-1).contiguous()
    if input_ids.shape[-1] >= max_seq_len:
        input_ids = input_ids[:, :max_seq_len]
        gpt_masks = gpt_masks[gpt_masks < max_seq_len].contiguous()

    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        gpt_masks=gpt_masks,
    )


class ShareGPTDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_seq_len):
        super(ShareGPTDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data

        self.cached_data_dict = []
        for i in range(len(self.raw_data)):
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, max_seq_len)
            ret = dict(
                input_ids=ret["input_ids"],
                attention_mask=ret["attention_mask"],
                gpt_masks=ret["gpt_masks"],
                data_index=i,
            )
            self.cached_data_dict.append(ret)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.cached_data_dict[i]


def make_sharegpt_data_module(args, tokenizer, min_gpt_len=50, seed=42):
    train_json = json.load(open(args.dataset_path, "r"))

    # select calibration set
    random.seed(seed)
    max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)
    random.shuffle(train_json)
    cal_json = []
    i = 0
    while len(cal_json) < args.max_train_samples:
        raw_data = train_json[i]
        ret = preprocess([raw_data["conversations"]], tokenizer, max_seq_len)
        if len(ret["gpt_masks"]) > min_gpt_len:
            cal_json.append(raw_data)
        i += 1

    train_dataset = ShareGPTDataset(cal_json, tokenizer, max_seq_len)

    return train_dataset
