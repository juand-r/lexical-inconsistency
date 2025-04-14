"""
SFT or continued pretraining on hypernymy data with LoRA to quickly test what it does
to generator/discriminator gap in various settings.

Example use:

CUDA_VISIBLE_DEVICES=3 python fine_tune_lora.py --epochs 5 --style generator --shots zero --negate
CUDA_VISIBLE_DEVICES=6 python fine_tune_lora.py --epochs 2 --shots zero --both union

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
    split_train_test,
    make_and_format_data,
    get_L_prompt,
    filtering_hypernym,
    filtering_swords,
    filtering_triviaqa,
    filtering_lambada
)

filtering_function_map = {
    "hypernym": filtering_hypernym,
    "swords": filtering_swords,
    "trivia-qa": filtering_triviaqa,
    "lambada": filtering_lambada
    }


# def accuracy_metrics_disc(p: EvalPrediction):

#     predictions = p.predictions  # logits [batch_size, seq_len, vocab]
#     labels = p.label_ids  # [batch_size, seq_len]

#     # get batch_len size array with index of last entry before padding (-100)
#     mask = labels == -100
#     has_neg_100 = np.any(mask, axis=1)
#     first_neg_100 = np.argmax(mask, axis=1)
#     last_valid_index = np.where(has_neg_100, first_neg_100 - 1, labels.shape[1] - 1)

#     # get batch_size x vocab array containing the vocabs for the last valid index in each batch
#     batch_size = predictions.shape[0]
#     selected_predictions = predictions[np.arange(batch_size), last_valid_index, :]

#     assert all(
#         [
#             tokenizer.decode(labels[ii, last_valid_index[ii]]) in [" Yes", " No"]
#             for ii in range(len(last_valid_index))
#         ]
#     )

#     yesind = tokenizer.encode(" Yes")[-1]
#     noind = tokenizer.encode(" No")[-1]
#     ygn = selected_predictions[:, yesind] - selected_predictions[:, noind] > 0

#     gold = labels[np.arange(batch_size), last_valid_index] == yesind

#     accuracy = sum(ygn == gold) / len(gold)

#     return {"accuracy": accuracy}


# class AccuracyCallback(WandbCallback):
#     def __init__(
#         self, trainer, eval_dataset, tokenizer, yes_token=" Yes", no_token=" No"
#     ):
#         super().__init__()
#         self.trainer = trainer
#         self.eval_dataset = eval_dataset
#         self.tokenizer = tokenizer
#         self.yes_token_id = self.tokenizer.encode(yes_token)[-1]
#         self.no_token_id = self.tokenizer.encode(no_token)[-1]

#     def on_evaluate(self, args, state, control, **kwargs):
#         # Call the superclass method to ensure proper Wandb state
#         super().on_evaluate(args, state, control, **kwargs)

#         # Run predictions on the evaluation dataset to get logits and labels
#         # predict() returns a namedtuple with .predictions and .label_ids
#         predictions_output = self.trainer.predict(self.eval_dataset)
#         predictions = (
#             predictions_output.predictions
#         )  # shape (batch_size, seq_len, vocab)
#         labels = predictions_output.label_ids  # shape (batch_size, seq_len)

#         # NOTE careful, passing the entire input_ids, which includes the completion!
#         # so offset from last is -2, not -1
#         offset = 2

#         # get batch_len size array with index of last entry before padding (-100)
#         mask = labels == -100
#         has_neg_100 = np.any(mask, axis=1)
#         first_neg_100 = np.argmax(mask, axis=1)
#         last_valid_index = np.where(
#             has_neg_100, first_neg_100 - offset, labels.shape[1] - offset
#         )

#         batch_size = predictions.shape[0]
#         selected_predictions = predictions[np.arange(batch_size), last_valid_index, :]

#         # Compute whether model predicts "Yes" over "No"
#         yes_scores = selected_predictions[:, self.yes_token_id]
#         no_scores = selected_predictions[:, self.no_token_id]
#         model_pred_yes = (yes_scores - no_scores) > 0

#         gold_yes = labels[np.arange(batch_size), last_valid_index] == self.yes_token_id
#         accuracy = np.mean(model_pred_yes == gold_yes)

#         self._wandb.log({"accuracy": accuracy})


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune (LoRA) models on hypernym task."
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-2-2b", help="model to train"
    )
    parser.add_argument("--split_type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument(
        "--task", type=str, default="hypernym", help="task to train on: hypernym, trivia-qa, swords, etc."
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
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--shots", type=str, help="'zero' vs 'few'")
    parser.add_argument("--sample_negative", action="store_true", default=False, help="whether to sample negative examples when loading trivia-qa or lambada")
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

    if "instruct" in model_id.lower():
        model_is_chat = True
    else:
        model_is_chat = False
    
    print("Model is chat: ", model_is_chat)
    # Data shuffling, subsampling and filtering arguments
    seed = args.seed
    subsample = args.subsample
    train_filter = args.filter
    if train_filter == 'pos':
        filtering_func = filtering_function_map[args.task]
    else:
        filtering_func = None
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ###########################################################
    # LOAD AND FORMAT DATA
    ###########################################################
    L_train, L_test, make_prompt = get_L_prompt(args.task, args.split_type, seed, sample_negative=args.sample_negative)
    print(f"len L_train: {len(L_train)}")
    print(f"len L_test: {len(L_test)}")

    print("Gen or disc: ", style)
    print("Shots: ", shots)
    print("Use negation? ", negate)
    print("Train variation? ", both)
    print("Instruction masking?", instruction_mask)
    print("\nTrain dataset size: ", len(L_train))

    p_train, hf_train, prompt_completion_train = make_and_format_data(
        make_prompt,
        L=L_train,
        tokenizer=tokenizer,
        style=style,
        shots=shots,
        neg=negate,
        both=both,
        instruction_masking=instruction_mask,
        filtering=filtering_func,
        is_chat=model_is_chat
    )
    p_test, hf_test, prompt_completion_test = make_and_format_data(
        make_prompt,
        L=L_test,
        tokenizer=tokenizer,
        style=style,
        shots=shots,
        neg=negate,
        both=both,
        instruction_masking=instruction_mask,
        is_chat=model_is_chat
    )
    print("!!!fine_tune_lora after pos filtering:") # 4514 -> 2010(1005); 3778 -> 838(419)
    print("Train dataset size: ", len(prompt_completion_train))
    print("Test dataset size: ", len(prompt_completion_test))

    for i in range(10):
        print(prompt_completion_train[i])
        # print(tokenizer.apply_chat_template(prompt_completion_train[i]))

    ################################################################
    # CONFIG FOR LoRA
    ################################################################

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    negstr = "--negation" if negate else ""
    if both == "none":
        output_dir = "ftmodel--{}--{}--{}--{}--{}{}".format(
            model_id.split("/")[-1], args.task, style, shots, train_filter, negstr
        )
    else:
        output_dir = "ftmodel--{}--{}--{}--{}--{}{}".format(
            model_id.split("/")[-1], args.task, both, shots, train_filter, negstr
        )
    if args.split_type != 'random':
        output_dir += f"--{args.split_type}"
    if args.sample_negative:
        output_dir += "--pn"
    
    output_dir = os.path.join("/datastor1/wenxuand/output/sft/", output_dir)

    #If this already exists, make sure not to overwrite it
    if os.path.exists(output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir}_{timestamp}"

    merge_dir = output_dir + "/merged"

    train_args = TrainingArguments(
        output_dir=output_dir,  # directory to save and repository id
        num_train_epochs=num_epochs,  # number of training epochs
        per_device_train_batch_size=16,  # $128, #2          # batch size per device during training
        per_device_eval_batch_size=7,  # 7#4
        # gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        # gradient_checkpointing=True,            # use gradient checkpointing to save memory
        # optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        learning_rate=args.lr,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="wandb",
        do_eval=True,
        eval_strategy="epoch",
        eval_steps=10,
        # eval_accumulation_steps=30 #works but slow
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=prompt_completion_train,
        eval_dataset=prompt_completion_test,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # packing=False,
    )


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

