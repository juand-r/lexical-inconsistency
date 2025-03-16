"""
Compute probabilities via tuned-lens or logit-lens; obtain log-odds over layers
and save to disk to analyze later.

Log-odds are computed for both discriminator and generator prompts.

Usage
=====

CUDA_VISIBLE_DEVICES=1 python logodds.py --model ftmodel--gemma-2-2b--generator--zero--all--negation

Tensors will be saved in `outputs/logodds`; the end of the file will be in the form {disc,gen}-{few,zero}.pt
indicating whether test samples were in discriminator or generator form (zero or few shot).
"""
#TODO should also track train or test set, in case I want to do this on the train set

from pathlib import Path
import os
import sys
import argparse
from tqdm import tqdm
import torch
import gc
import json
from datasets import load_dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from utils import load_noun_pair_data, split_train_test, split_train_test_no_overlap, split_train_test_no_overlap_both, make_prompt_hypernymy, make_prompt_triviaqa, make_prompt_swords, load_swords_data, get_final_logit_prob

from logitlens import get_logitlens_output, load_model_nnsight, compute_logodds
from tunedlens import init_lens, obtain_prob_tensor

device = "cuda"
yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
no_words = ["No", " No", "NO", "no", " no"]


def main():
    parser = argparse.ArgumentParser(description="Compute log-odds on test data")
    parser.add_argument("--model", type=str, help="model directory to process (this should contain merged/ subdirectory) or hf model")
    parser.add_argument("--tunedlens", action="store_true", default=False, help="")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling the data.")
    parser.add_argument("--disc-shots", type=str, default='few', help="'zero' vs 'few'")
    parser.add_argument("--gen-shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--train", action="store_true", default=False, help="log-odds of train or test set?")
    parser.add_argument("--split-type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument("--task", type=str, default='hypernym', help="hypernym, trivia-qa, etc")

    args = parser.parse_args()

    task = args.task
    modelname = args.model
    do_tunedlens = args.tunedlens
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    train_flag = args.train
    split_type = args.split_type

    if task=='hypernym':
        L = load_noun_pair_data()
        if split_type=='hyper':
            L_train, L_test = split_train_test_no_overlap(L, seed=seed)
        elif split_type=='random':
            L_train, L_test = split_train_test(L, seed=seed, subsample=False, num_train=3000)
        elif split_type=='both':
            L_train, L_test = split_train_test_no_overlap_both(L, seed=2)
        else:
            raise ValueError("Wrong value for split-type")
        make_prompt = make_prompt_hypernymy
    elif task=='trivia-qa':
        # load data here
        L = load_dataset('lucadiliello/triviaqa') #TODO check if this is correct version.
        #USE SUBSET FOR NOW
        L_train =  L['train'].shuffle(seed=42).select(range(3000))
        L_test = L['validation'].shuffle(seed=42).select(range(1000))

        #NOTE assumes this takes same arguments in each case
        make_prompt = make_prompt_triviaqa
    elif task=='swords':
        L_train, L_test = load_swords_data(seed=0)
        make_prompt = make_prompt_swords
    else:
        raise NotImplementedError("Not a task")

    device = "cuda"

    # Load model, lens, and tokenizer
    if do_tunedlens:
        if modelname not in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
            raise NotImplementedError("Model not supported for tuned-lens yet.")
        init_lens(modelname, device=device)
    else: # logit-lens
        modeldir = os.path.join("../models", modelname)
        if Path(os.path.join(modeldir, "merged")).is_dir():
            # in this case it is a merged model via lora
            modeldir_merged = os.path.join(modeldir, "merged")
            modelname = modeldir_merged
            modelname_short = os.path.basename(modeldir).split("--")[1]
        elif Path(modeldir).is_dir():
            #TODO clean up name later
            modeldir_merged =  modeldir
            modelname = modeldir_merged
            modelname_short = "gemma-2-2b" #os.path.basename(modeldir)
        else: # it's a huggingface model id
            modelname_short = modelname.split("/")[1]
            modeldir = modelname_short

        model = load_model_nnsight(modelname, device)
        tokenizer = model.tokenizer

    yestoks = [tokenizer.encode(i)[-1] for i in yes_words]
    notoks = [tokenizer.encode(i)[-1] for i in no_words]

    # NOTE: the first subword token of interest is different because gemma-2 adds <bos>
    if "gpt" in modelname_short:
        first_sw_token = 1
    elif "gemma" in modelname_short or "Llama" in modelname_short:
        first_sw_token = 2
    else:
        raise ValueError("!?")

    # Calculate probabilities via tuned/logit-lens: generator and discriminator
    if train_flag:
        LL = L_train
        train_suffix = "--train"
    else:
        LL = L_test
        train_suffix = ""

    if split_type=='random':
        split_suffix = ""
    elif split_type=='hyper':
        split_suffix = "--hyper"
    elif split_type=='both':
        split_suffix = "--both"
    else:
        raise ValueError()

    P_gen = []
    P_gen_final = []
    for item in tqdm(LL):

        prompt = make_prompt(item, style='generator', shots=gen_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = get_logitlens_output(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
            #print(probs.shape)
            #print(probs[-1, :10])
            probs_gen = get_final_logit_prob(prompt, model, tokenizer, device, is_chat = False) # TODO: change is_chat to True if instruction-tuned model
            #print(probs_gen.shape)
            #print(probs_gen[:10])
            #print("sum probs:", probs_gen.sum())
            #print("----")
            #del X 
        P_gen.append(probs)
        P_gen_final.append(probs_gen) # TODO: P_gen_final does not match with P_gen with gemma-2-2b

    print("NOW DOING DISCRIMINATOR")
    gc.collect()
    torch.cuda.empty_cache()

    P_disc = []
    for item in tqdm(LL):
        prompt = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = get_logitlens_output(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
            del X
        P_disc.append(probs)


    ranks, logodds_gen, logodds_disc, corr, disc_accuracy, gen_accuracies = compute_logodds(task,
        P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks, layer_gen=-1, layer_disc=-1)

    tensordir_disc = os.path.join("../outputs/logodds", os.path.basename(modeldir) + "--" + task + "--disc-"+disc_shots + train_suffix + split_suffix + ".pt")
    tensordir_gen = os.path.join("../outputs/logodds", os.path.basename(modeldir) + "--" + task + "--gen-"+gen_shots + train_suffix + split_suffix + ".pt")
    torch.save(logodds_disc, tensordir_disc)
    torch.save(logodds_gen, tensordir_gen)
    ranks_gen = os.path.join("../outputs/logodds", os.path.basename(modeldir) + "--" + task + "--gen-"+gen_shots + train_suffix + split_suffix + "--ranks.json")
    with open(ranks_gen, 'w') as f:
        json.dump(ranks, f, indent=4)

    #print("\n\n")

if __name__ == "__main__":
    main()
