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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--task", type=str, default='lexical')
    parser.add_argument("--dataset_dir", type=str, default="/u/wenxuand/generator_validator/DPO/data/win-lose-delta-5.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="7", required=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default='google/gemma-2-2b')#default=) 'meta-llama/Meta-Llama-3-8B-Instruct'
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/datastor1/wenxuand/output") 
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)
    print("Using cuda: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    args.device = "cuda:0"

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_data_instruction_qa(args.dataset_dir)

    output_dir = os.path.join(args.output_dir, args.task, args.model.split('/')[-1].strip(), f"dpo_{time_string}")
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
        project = 'dpo_v1',
        config = training_args,
        name = f"distill_{args.task}_{time_string}"
    )

    trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=dataset)

    trainer.train()

