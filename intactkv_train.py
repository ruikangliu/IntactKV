import gc
import glob
import os
import logging
import shutil
import argparse
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import set_seed, Seq2SeqTrainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.datautils import make_data_module
from utils.modelutils import build_fp16_model, print_trainable_parameters, print_parameters_dtype
from intactkv.utils import gen_vicuna_prompt_kv, get_acts_list
from intactkv.train_model import get_quantized_model


logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    fp16_model_path: Optional[str] = field(
        default="./vicuna-v1.5-7b",
        metadata={"help": "path to full-precision model"}
    )
    bits: Optional[int] = field(
        default=3,
        metadata={"help": "#bits of quantized model"}
    )
    group_size: Optional[int] = field(
        default=128,
        metadata={"help": "group size of quantized model"}
    )
    quant_method: Optional[str] = field(
        default="awq",
        metadata={"help": "which quantization method to use"}
    )
    gptq_path: Optional[str] = field(
        default="./modelzoo/autogptq",
        metadata={"help": "path to GPTQ model"}
    )
    awq_path: Optional[str] = field(
        default="./modelzoo/llm-awq",
        metadata={"help": "path to AWQ model"}
    )
    intactkv_size: Optional[int] = field(
        default=None,
        metadata={"help": "size of IntactKV"}
    )


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=128,
        metadata={"help": "calset size"},
    )
    max_seq_len: Optional[int] = field(
        default=1024,
        metadata={"help": "max length of input ids"},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to calset"},
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default='./outputs',
        metadata={"help": 'The output dir for logs and checkpoints'}
    )
    exp_name: Optional[str] = field(
        default='intactkv',
        metadata={"help": 'experiment name'}
    )
    optim: Optional[str] = field(
        default='paged_adamw_32bit',
        metadata={"help": 'The optimizer to be used'}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": 'The training batch size per GPU.'}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16,
        metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'}
    )
    max_steps: Optional[int] = field(
        default=160,
        metadata={"help": 'How many optimizer update steps to take'}
    )
    learning_rate: Optional[float] = field(
        default=2e-4,
        metadata={"help": 'The learning rate'}
    )
    max_grad_norm: Optional[float] = field(
        default=0.3,
        metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'}
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": 'To train or not to train, that is the question?'}
    )
    lr_scheduler_type: Optional[str] = field(
        default='constant',
        metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'}
    )
    warmup_ratio: Optional[float] = field(
        default=0.03,
        metadata={"help": 'Fraction of steps to do a warmup for'}
    )
    logging_strategy: Optional[str] = field(
        default="epoch",
        metadata={"help": 'When to log the loss'}
    )
    save_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": 'When to save checkpoints'}
    )
    save_steps: Optional[int] = field(
        default=999,
        metadata={"help": 'How often to save a model'}
    )


class SaveIntactKVCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving intactKV...')
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)

        save_path = os.path.join(checkpoint_folder, "intactkv.pth")
        torch.save(kwargs["model"].intactkv, save_path)

        # delete saved full model
        def os_remove(pattern):
            for name in glob.glob(pattern):
                os.remove(name)
        os_remove(os.path.join(checkpoint_folder, "*.bin"))
        os_remove(os.path.join(checkpoint_folder, "*.safetensors"))

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # output path
    args.cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    num_epochs = args.max_steps * args.gradient_accumulation_steps // args.max_train_samples
    args.fp16_model_path = args.fp16_model_path[:-1] if args.fp16_model_path.endswith("/") else args.fp16_model_path
    model_name = args.fp16_model_path.split("/")[-1]
    args.model_name = model_name
    assert "vicuna" in args.model_name  # TODO. support Vicuna models only
    args.model_type = "-".join(model_name.split("-")[:-1])
    args.gptq_path = os.path.join(args.gptq_path, args.model_type, f"{model_name}-{args.bits}bit-{args.group_size}g")
    args.awq_path = os.path.join(args.awq_path, args.model_type, f"{model_name}-w{args.bits}-g{args.group_size}.pt")
    assert args.quant_method in ["rtn", "gptq", "awq"]
    args.output_dir = os.path.join(args.output_dir, args.quant_method, model_name, f"{args.bits}-bits", f"g{args.group_size}",
                                   f"{args.exp_name}", f"epoch_{num_epochs}")
    args.writer = SummaryWriter(os.path.join(args.output_dir, "log")) 
    training_args.output_dir = args.output_dir

    # find latest checkpoint to resume training
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    # save .py
    if not completed_training:
        os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
        shutil.copy(__file__, os.path.join(args.output_dir, "code", __file__.split("/")[-1]))
    else:
        logging.log("Detected that training was already completed!")
        return

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
    intactkv, intactkv_ids = gen_vicuna_prompt_kv(model, tokenizer)

    # clean up GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # load intactkv
    if checkpoint_dir is not None:
        logging.info("Loading intactkv from checkpoint...")
        loaded_intactkv = torch.load(join(checkpoint_dir, "intactkv.pth"))
        intactkv = []
        for layer, layer_kv in enumerate(loaded_intactkv):
            intactkv.append(tuple([layer_kv[0], layer_kv[1]]))
        intactkv = tuple(intactkv)
    
    # get quantized model
    logging.info("Loading quantized model...")
    model = get_quantized_model(args, tokenizer, acts_list, intactkv, intactkv_ids)
    model.quant_method = args.quant_method
    training_args.skip_loading_checkpoint_weights = True

    # only intactkv is trainable
    for name, param in model.named_parameters():
        if "intactkv" in name:
            param.requires_grad = True
        elif param.requires_grad:
            param.requires_grad = False
    print_trainable_parameters(model)

    # fp16 inference
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    set_seed(args.seed)

    # Verifying the datatypes
    print_parameters_dtype(model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Callbacks
    trainer.add_callback(SaveIntactKVCallback)

    # Training
    train_result = trainer.train(resume_from_checkpoint=False)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    train()
