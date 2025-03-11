import json
import torch
from tqdm import tqdm
from datasets import Dataset
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from datetime import datetime
import argparse
import wandb
import os
import numpy as np
from collections import defaultdict
import itertools
from peft import LoraConfig, get_peft_model
current_time = datetime.now()
time_string = current_time.strftime("%d[%H_%M_%S]")

# (1)
def load_data_instruction_qa(file):
    with open(file, 'r') as f:
        data = json.load(f)
    d = data[0]['winner']
    splits = d.split('\n\n')
    instruction = '\n\n'.join(splits[:-1])
    ind = len(instruction)
    prompts = [instruction] * len(data)
    chosen = []
    rejected = []
    for i in tqdm(range(len(data))):
        d_win = data[i]['winner']
        d_lose = data[i]['loser']
        chosen.append(d_win[ind:].strip() + ' Yes')
        rejected.append(d_lose[ind:].strip() + ' Yes')
    dataset_dict = {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

# (2)
def load_hypernym_disc_to_gen(file = "../data/hypernymy-train.json", threshold = -11.3):
    with open(file, 'r') as f:
        data_hypernym = json.load(f)

    if threshold == None:
        threshold = np.mean([d['generator-log-prob'] for d in data_hypernym])
    prompts = [d['discriminator-prompt'] for d in data_hypernym]
    chosen = []
    rejected = []
    for i in tqdm(range(len(data_hypernym))):
        d = data_hypernym[i]
        if d['generator-log-prob'] > threshold:
            chosen.append('Yes')
            rejected.append('No')
        else:
            chosen.append('No')
            rejected.append('Yes')

    dataset_dict = {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset
    
def load_hypernym_gen_to_disc(file = "../data/hypernymy-train.json"):
    with open(file, 'r') as f:
        data_hypernym = json.load(f)
    same_hyponym_dicts = defaultdict(list)
    for d in data_hypernym:
        same_hyponym_dicts[d['noun1']].append(d)
    # print(len(same_hyponym_dicts)) 1135
    num_hypernym_lists = [len(v) for v in same_hyponym_dicts.values()]
    # print(np.mean(num_hypernym_lists)) 2.64

    prompts = []
    chosen = []
    rejected = []
    for k, v in same_hyponym_dicts.items():
        if len(v) > 1:
            pairs_combinations = list(itertools.combinations(v, 2))
            pairs_combinations = sorted(pairs_combinations, key=lambda pair: abs(pair[0]['discriminator-log-prob'] - pair[1]['discriminator-log-prob']))
            for (a,b) in pairs_combinations[:len(v)]:
                if a['discriminator-log-prob'] >= b['discriminator-log-prob']:
                    prompts.append(v[0]['generator-prompt'])
                    chosen.append(a['noun2'])
                    rejected.append(b['noun2'])
                else:
                    prompts.append(v[0]['generator-prompt'])
                    chosen.append(b['noun2'])
                    rejected.append(a['noun2'])
    dataset_dict = {"prompt": prompts, "chosen": chosen, "rejected": rejected}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

data_loader = {'hypernym':
               {'d2g': load_hypernym_disc_to_gen,
                'g2d': load_hypernym_gen_to_disc},
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--task", type=str, default='hypernym')
    parser.add_argument("--direction", type=str, default='d2g', help="d2g or g2d")
    parser.add_argument("--dataset_dir", type=str, default="../data/hypernymy-train.json")
    parser.add_argument("--epochs", type=int, default=1)
    # parser.add_argument("--device", type=str, default="7", required=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default='google/gemma-2-2b')#default=) 'meta-llama/Meta-Llama-3-8B-Instruct'
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/datastor1/wenxuand/output") 
    
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)
    # print("Using cuda: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    args.device = "cuda"

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = data_loader[args.task][args.direction](args.dataset_dir)

    output_dir = os.path.join(args.output_dir, args.task, args.model.split('/')[-1].strip(), f"dpo_{args.direction}_{time_string}")
    os.makedirs(output_dir, exist_ok=True)
    training_args = DPOConfig(
        beta = args.beta,
        output_dir = output_dir,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        num_train_epochs = args.epochs,
        save_strategy = 'epoch',
        report_to = 'wandb',
        logging_steps = 20,
        seed = args.seed,
        do_train = True,
        bf16 = True,
        learning_rate = args.lr,

    )

    with open(os.path.join(output_dir, 'args_dict.txt'), 'w') as f:
        for arg, value in vars(training_args).items():
            f.write(f'{arg}: {value}\n')
    
    run = wandb.init(
        mode='online',
        project = 'dpo_std',
        config = training_args,
        name = f"distill_{args.task}_{time_string}"
    )

    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=dataset)

    trainer.train()

