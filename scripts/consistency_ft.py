"""
SFT or continued pretraining on hypernymy data with LoRA to quickly test what it does
to generator/discriminator gap in various settings.

Example use:

CUDA_VISIBLE_DEVICES=3 python consistency_ft.py --epochs 5 --style generator --shots zero --negate
CUDA_VISIBLE_DEVICES=6 python consistency_ft.py --epochs 2 --shots zero --both union

"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import random
import gc
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
    make_and_format_data,
    get_L_prompt,
    filtering_hypernym,
    filtering_swords,
    filtering_triviaqa,
    filtering_lambada,
    get_final_logit_prob
)
from logitlens import compute_logodds_final_layer, get_logodds_gen, get_logodds_disc

yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
no_words = ["No", " No", "NO", "no", " no"]

filtering_function_map = {
    "hypernym": filtering_hypernym,
    "swords": filtering_swords,
    "trivia-qa": filtering_triviaqa,
    "lambada": filtering_lambada
    }

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
    parser.add_argument("--shots", type=str, default = 'zero', help="'zero' vs 'few'")
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
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="Round of consistent fine-tuning",
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
    # L = load_noun_pair_data()
    # L_train, L_test = split_train_test(
    #     L, seed=seed, subsample=subsample, num_train=num_train
    # )
    # if train_filter == "all":
    #     print("Training with all data\n")
    # elif train_filter == "pos":
    #     L_train = [i for i in L_train if i.taxonomic == "yes"]
    #     print("Training with positive data only\n")
    # elif train_filter == "neg":
    #     L_train = [i for i in L_train if i.taxonomic == "no"]
    #     print("Training with negative data only\n")
    # else:
    #     raise ValueError("!")
    L_train, L_test, make_prompt = get_L_prompt(args.task, 'random', seed, sample_negative=args.sample_negative)
    print(f"len L_train: {len(L_train)}")
    print(f"len L_test: {len(L_test)}")

    print("Gen or disc: ", style)
    print("Shots: ", shots)
    print("Use negation? ", negate)
    print("Train variation? ", both)
    print("Instruction masking?", instruction_mask)
    print("\nTrain dataset size: ", len(L_train))

    '''
    filtering consistent data
    '''
    LL = L_train
    P_gen = []
    P_disc = []
    json_list = []
    task = args.task
    first_sw_token = 2

    model_is_chat = False
    if 'instruct' in model_id.lower():
        model_is_chat = True
        first_sw_token = 1
        print("Model is chat model!")
    if "gpt" in model_id.lower():
        raise ValueError("If you are using GPT then rewrite this bit!")

    yestoks = [tokenizer.encode(i)[-1] for i in yes_words]
    notoks = [tokenizer.encode(i)[-1] for i in no_words]

    for item in tqdm(LL):
        prompt_gen = make_prompt(item, style='generator', shots=shots).prompt
        prompt_disc = make_prompt(item, style='discriminator', shots=shots).prompt
        probs_gen = get_final_logit_prob(prompt_gen, model, tokenizer, device, is_chat = model_is_chat) 
        P_gen.append(probs_gen)
        probs_disc = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat = model_is_chat) 
        P_disc.append(probs_disc)

        prefix = " " if not model_is_chat else ""
        if task == 'hypernym':
            json_list.append({"noun1":item.noun1, "noun2":item.noun2, "taxonomic":item.taxonomic, "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0, 
                            "generator-completion": prefix + item.noun2.strip(), "discriminator-gold-completion": prefix + item.taxonomic.strip().capitalize()})
        elif task == 'trivia-qa':
            # print(item.keys())
            if args.sample_negative:
                    json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + item['correct']})
            else:
                json_list.append({"question":item['question'], "answer":prefix + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                            "generator-completion": prefix + item['answers'][0].strip(), "discriminator-gold-completion": prefix + 'Yes'})
        elif task == 'lambada':
            if args.sample_negative:
                    json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + item['correct']})
            else:
                json_list.append({"context":item['context'], "completion":prefix + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                            "generator-completion": prefix + item['final_word'].strip(), "discriminator-gold-completion": prefix + 'Yes'})
        elif task == 'swords':
            json_list.append({"context":item.context, "target":item.target, "replacement":item.replacement, "synonym":item.synonym,
                            "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                            "generator-completion": item.replacement.strip(), "discriminator-gold-completion": prefix + item.synonym.strip().capitalize()})
        else:
            raise NotImplementedError("Not a task")
            # print(json_list[-1])
   
    logodds_gen = [get_logodds_gen(P_gen, LL, ii, tokenizer, first_sw_token, task, is_chat = model_is_chat, use_lgo=False) for ii in range(len(P_gen))]
    logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, None) for ii in range(len(P_disc))]
    
    consistent_LL = []
    gen_threshold = sum(logodds_gen)/len(logodds_gen)
    disc_threshold = sum(logodds_disc)/len(logodds_disc)

    for jj, item in tqdm(enumerate(LL)):
        score_gen = logodds_gen[jj]
        score_disc = logodds_disc[jj]
        if (score_gen > gen_threshold) and (score_disc > disc_threshold):
            consistent_LL.append(item)
    if type(LL) != list:
        consistent_LL = Dataset.from_list(consistent_LL)
    print(f"!!!!!!after filtering:")
    print(f"len consistent_LL: {len(consistent_LL)}")
    print(f"len LL: {len(LL)}")

    gc.collect()
    torch.cuda.empty_cache()






    print("Preparing and formating data for training...")
    p_train, hf_train, prompt_completion_train = make_and_format_data(
        make_prompt,
        L=consistent_LL,
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
    print("Train dataset size: ", len(prompt_completion_train))
    print("Test dataset size: ", len(prompt_completion_test))
    # raise ValueError("STOP")
    for i in range(5):
        print(prompt_completion_train[i])
    # raise ValueError("STOP")
        # print(tokenizer.apply_chat_template(prompt_completion_train[i]))
    # raise ValueError("STOP")
    ################################################################
    # CONFIG FOR LoRA
    ################################################################
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # From: https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb
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
        output_dir = "conftmodel_{}--{}--{}--{}--{}--{}{}--size_{}-{}".format(
            args.round,
            model_id.split("/")[-1], args.task, style, shots, train_filter, negstr, len(L_train), len(consistent_LL)
        )
    else:
        output_dir = "conftmodel_{}--{}--{}--{}--{}--{}{}--size_{}-{}".format(
            args.round,
            model_id.split("/")[-1], args.task, both, shots, train_filter, negstr, len(L_train), len(consistent_LL)
        )
    output_dir = os.path.join("/datastor1/wenxuand/output/consistent_sft/", output_dir)
    if args.sample_negative:
        output_dir += "--pn"
    
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

'''
CUDA_VISIBLE_DEVICES=5 python consistency_ft.py --epochs 2 --shots zero --both union --filter pos --task lambada --model google/gemma-2-2b
CUDA_VISIBLE_DEVICES=6 python consistency_ft.py --epochs 2 --shots zero --both union --filter pos --task lambada --model meta-llama/Llama-3.2-3B

CUDA_VISIBLE_DEVICES=2 python consistency_ft.py --epochs 2 --shots zero --both union --filter pos --task trivia-qa --model google/gemma-2-2b
CUDA_VISIBLE_DEVICES=7 python consistency_ft.py --epochs 2 --shots zero --both union --filter pos --task swords --model meta-llama/Llama-3.2-3B-Instruct
CUDA_VISIBLE_DEVICES=7 python consistency_ft.py --epochs 2 --shots zero --both union --filter pos --task swords --model meta-llama/Llama-3.2-3B


CUDA_VISIBLE_DEVICES={} python consistency_ft.py --epochs 2 --shots zero --both union  --task trivia-qa --model {} --sample_negative
CUDA_VISIBLE_DEVICES={} python consistency_ft.py --epochs 2 --shots zero --both union  --task lambada --model {} --sample_negative

'''