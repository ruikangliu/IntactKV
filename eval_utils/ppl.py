from tqdm import tqdm

import torch
import torch.nn as nn


@torch.no_grad()
def ppl_eval(model, tokenizer, testenc):
    print('Evaluating ppl...')
    model.eval()
    max_length = 2048   # fix model max length

    nsamples = testenc.numel() // max_length

    dev = model.lm_head.weight.device

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]
        batch[:, 0] = tokenizer.bos_token_id  # ensure the 1st token is <bos>
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * max_length): ((i + 1) * max_length)
        ][:, 1:]

        # ignore next token prediction after <bos>
        shift_logits = shift_logits[:, 1:, :]
        shift_labels = shift_labels[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * max_length))
    print(ppl.item())
    return ppl.item()
