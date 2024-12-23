"""
SFT or continued pretraining on hypernymy data with LoRA to quickly test what it does
to generator/discriminator gap in various settings.

Example use:

CUDA_VISIBLE_DEVICES=3 python fine_tune_lora.py --epochs 5 --style generator --shots zero --negate
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import random
from datetime import datetime
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EvalPrediction,
)
from transformers.integrations import WandbCallback
from trl import SFTTrainer
from datasets import Dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from utils import (
    load_noun_pair_data,
    make_prompt,
    split_train_test,
    make_and_format_data,
)


# NOTE: this is currently too slow to be useful, but leaving in in case
# I want to troubleshoot this again later.
def accuracy_metrics_disc(p: EvalPrediction):

    predictions = p.predictions  # logits [batch_size, seq_len, vocab]
    labels = p.label_ids  # [batch_size, seq_len]

    # get batch_len size array with index of last entry before padding (-100)
    mask = labels == -100
    has_neg_100 = np.any(mask, axis=1)
    first_neg_100 = np.argmax(mask, axis=1)
    last_valid_index = np.where(has_neg_100, first_neg_100 - 1, labels.shape[1] - 1)

    # get batch_size x vocab array containing the vocabs for the last valid index in each batch
    batch_size = predictions.shape[0]
    selected_predictions = predictions[np.arange(batch_size), last_valid_index, :]

    assert all(
        [
            tokenizer.decode(labels[ii, last_valid_index[ii]]) in [" Yes", " No"]
            for ii in range(len(last_valid_index))
        ]
    )

    yesind = tokenizer.encode(" Yes")[-1]
    noind = tokenizer.encode(" No")[-1]
    ygn = selected_predictions[:, yesind] - selected_predictions[:, noind] > 0

    gold = labels[np.arange(batch_size), last_valid_index] == yesind

    accuracy = sum(ygn == gold) / len(gold)

    return {"accuracy": accuracy}


# NOTE: using wandb callbacks is slightly faster, but still too slow to iterate quickly!
class AccuracyCallback(WandbCallback):
    def __init__(
        self, trainer, eval_dataset, tokenizer, yes_token=" Yes", no_token=" No"
    ):
        super().__init__()
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.yes_token_id = self.tokenizer.encode(yes_token)[-1]
        self.no_token_id = self.tokenizer.encode(no_token)[-1]

    def on_evaluate(self, args, state, control, **kwargs):
        # Call the superclass method to ensure proper Wandb state
        super().on_evaluate(args, state, control, **kwargs)

        # Run predictions on the evaluation dataset to get logits and labels
        # predict() returns a namedtuple with .predictions and .label_ids
        predictions_output = self.trainer.predict(self.eval_dataset)
        predictions = (
            predictions_output.predictions
        )  # shape (batch_size, seq_len, vocab)
        labels = predictions_output.label_ids  # shape (batch_size, seq_len)

        # NOTE careful, passing the entire input_ids, which includes the completion!
        # so offset from last is -2, not -1
        offset = 2

        # get batch_len size array with index of last entry before padding (-100)
        mask = labels == -100
        has_neg_100 = np.any(mask, axis=1)
        first_neg_100 = np.argmax(mask, axis=1)
        last_valid_index = np.where(
            has_neg_100, first_neg_100 - offset, labels.shape[1] - offset
        )

        batch_size = predictions.shape[0]
        selected_predictions = predictions[np.arange(batch_size), last_valid_index, :]

        # Compute whether model predicts "Yes" over "No"
        yes_scores = selected_predictions[:, self.yes_token_id]
        no_scores = selected_predictions[:, self.no_token_id]
        model_pred_yes = (yes_scores - no_scores) > 0

        gold_yes = labels[np.arange(batch_size), last_valid_index] == self.yes_token_id
        accuracy = np.mean(model_pred_yes == gold_yes)

        self._wandb.log({"accuracy": accuracy})


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune (LoRA) models on hypernym task."
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2-2b", help="model to train"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for shuffling the data."
    )
    parser.add_argument(
        "--subsample",
        action="store_true",
        default=False,
        help="Use 1/10th of the data to train (for quick testing)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        help="how to filter the training data; can be 'all', 'pos', 'neg'",
    )

    parser.add_argument("--style", type=str, help="'discriminator' vs 'generator'")
    parser.add_argument("--shots", type=str, help="'zero' vs 'few'")
    parser.add_argument(
        "--both",
        type=str,
        default="none",
        help="Train using 'union', 'joint', 'none'. Union: both discriminator and generator examples. Joint: each training example combines both forms. None: just train on one type, given by 'style'.",
    )
    parser.add_argument(
        "--negate",
        action="store_true",
        default=False,
        help="Use negation for generator form in negative examples.",
    )
    parser.add_argument(
        "--instruction-mask",
        action="store_true",
        default=True,
        help="Use instruction masking (SFT) or not.",
    )
    args = parser.parse_args()

    # Model & training arguments
    model_id = args.model
    num_epochs = args.epochs

    # Data shuffling, subsampling and filtering arguments
    seed = args.seed
    subsample = args.subsample
    train_filter = args.filter
    num_train = 3000

    # Prompt construction arguments
    style = args.style
    shots = args.shots
    both = args.both
    negate = args.negate
    instruction_mask = args.instruction_mask

    #########################################################
    # LOAD MODEL AND TOKENIZER
    #########################################################
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",  # use this due to gemma-2 bug
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    ###########################################################
    # LOAD AND FORMAT DATA
    ###########################################################
    L = load_noun_pair_data()
    L_train, L_test = split_train_test(
        L, seed=seed, subsample=subsample, num_train=num_train
    )
    if train_filter == "all":
        print("Training with all data\n")
    elif train_filter == "pos":
        L_train = [i for i in L_train if i.taxonomic == "yes"]
        print("Training with positive data only\n")
    elif train_filter == "neg":
        L_train = [i for i in L_train if i.taxonomic == "no"]
        print("Training with negative data only\n")
    else:
        raise ValueError("!")

    print("Gen or disc: ", style)
    print("Shots: ", shots)
    print("Use negation? ", negate)
    print("Train variation? ", both)
    print("Instruction masking?", instruction_mask)
    print("\nTrain dataset size: ", len(L_train))

    p_train, hf_train = make_and_format_data(
        L_train,
        tokenizer,
        style=style,
        shots=shots,
        neg=negate,
        both=both,
        instruction_masking=instruction_mask,
    )
    p_test, hf_test = make_and_format_data(
        L_test,
        tokenizer,
        style=style,
        shots=shots,
        neg=negate,
        both=both,
        instruction_masking=instruction_mask,
    )

    ################################################################
    # CONFIG FOR LoRA
    ################################################################
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # From: https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    negstr = "--negation" if negate else ""
    if both == "none":
        output_dir = "ftmodel--{}--{}--{}--{}{}".format(
            model_id.split("/")[-1], style, shots, train_filter, negstr
        )
    else:
        output_dir = "ftmodel--{}--{}--{}--{}{}".format(
            model_id.split("/")[-1], both, shots, train_filter, negstr
        )
    output_dir = os.path.join("../models/", output_dir)

    #If this already exists, make sure not to overwrite it
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

    merge_dir = output_dir + "/merged"

    train_args = TrainingArguments(
        output_dir=output_dir,  # directory to save and repository id
        num_train_epochs=num_epochs,  # number of training epochs
        per_device_train_batch_size=64,  # $128, #2          # batch size per device during training
        per_device_eval_batch_size=7,  # 7#4
        # gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        # gradient_checkpointing=True,            # use gradient checkpointing to save memory
        # optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="wandb",
        do_eval=True,
        eval_strategy="epoch",
        # eval_steps=10,
        # eval_accumulation_steps=30 #works but slow
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=hf_train,
        eval_dataset=hf_test,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=False,
    )

    # TODO this requires a little more tinkering to get working
    # wandb_callback = AccuracyCallback(trainer, hf_test, tokenizer, yes_token=" Yes", no_token=" No")
    # trainer.add_callback(wandb_callback)

    trainer.train()
    trainer.save_model()

    # Keep train config in same directory as model
    with open(os.path.join(output_dir, "config_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Loading saved model...")
    model2 = AutoPeftModelForCausalLM.from_pretrained(output_dir)
    print("Merging model...")
    merged_model = model2.merge_and_unload()
    print("Saving merged model...")
    merged_model.save_pretrained(
        merge_dir, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(merge_dir)


if __name__ == "__main__":
    main()
