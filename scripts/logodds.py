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

import os
import sys
import argparse
from tqdm import tqdm
import torch
import json

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from utils import load_noun_pair_data, split_train_test, make_prompt
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
    args = parser.parse_args()

    modelname = args.model
    do_tunedlens = args.tunedlens
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    train_flag = args.train

    L = load_noun_pair_data()
    L_train, L_test = split_train_test(L, seed=seed, subsample=False, num_train=3000)
    device = "cuda"

    # Load model, lens, and tokenizer
    if do_tunedlens:
        if modelname not in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
            raise NotImplementedError("Model not supported for tuned-lens yet.")
        init_lens(modelname, device=device)
    else: # logit-lens
        modeldir = os.path.join("../models", modelname)
        modeldir_merged = os.path.join(modeldir, "merged")
        if os.path.isdir(modeldir_merged):
            modelname = modeldir_merged
            modelname_short = os.path.basename(modeldir).split("--")[1]
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
    P_gen = []
    if train_flag:
        LL = L_train
        train_suffix = "--train"
    else:
        LL = L_test
        train_suffix = ""

    for item in tqdm(LL[:]):
        prompt = make_prompt(item, style='generator', shots=gen_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = get_logitlens_output(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
        P_gen.append(probs)

    P_disc = []
    for item in tqdm(LL[:]):
        prompt = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = get_logitlens_output(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
        P_disc.append(probs)


    ranks, logodds_gen, logodds_disc, corr = compute_logodds(
        P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks, layer_gen=-1, layer_disc=-1
    )

    tensordir_disc = os.path.join("../outputs/logodds", os.path.basename(modeldir)+ "--disc-"+disc_shots + train_suffix+".pt")
    tensordir_gen = os.path.join("../outputs/logodds", os.path.basename(modeldir)+ "--gen-"+gen_shots + train_suffix+".pt")
    torch.save(logodds_disc, tensordir_disc)
    torch.save(logodds_gen, tensordir_gen)
    ranks_gen = os.path.join("../outputs/logodds", os.path.basename(modeldir)+ "--gen-"+gen_shots + train_suffix + "--ranks.json")
    with open(ranks_gen, 'w') as f:
        json.dump(ranks, f, indent=4)

    print("\n\n")

if __name__ == "__main__":
    main()
