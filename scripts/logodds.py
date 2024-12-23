"""
Compute probabilities via tuned-lens or logit-lens; obtain log-odds over layers
and save to disk to analyze later.
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)

from utils import load_noun_pair_data, split_train_test, make_prompt
from logitlens import logitlens, load_model_nnsight, compute_logodds
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
    args = parser.parse_args()

    modelname = args.model
    do_tunedlens = args.tunedlens
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots

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
    for item in tqdm(L_test[:]):
        prompt = make_prompt(item, style='generator', shots=gen_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = logitlens(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
        P_gen.append(probs)

    P_disc = []
    for item in tqdm(L_test[:]):
        prompt = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        if do_tunedlens:
            input_ids = tokenizer.encode(prompt)
            probs = obtain_prob_tensor(input_ids, token_pos=-1)
        else: #do_logitlens:
            X = logitlens(prompt, model, modelname_short)
            probs = X[0][:, -1, :].detach().cpu()
        P_disc.append(probs)


    ranks, logodds_gen, logodds_disc = compute_logodds(
        P_gen, None, None, P_disc, L_test, tokenizer, first_sw_token, yestoks, notoks
    )

    tensordir_disc = os.path.join("../outputs/logodds", os.path.basename(modeldir)+ "--disc-"+disc_shots+".pt")
    tensordir_gen = os.path.join("../outputs/logodds", os.path.basename(modeldir)+ "--gen-"+gen_shots+".pt")
    torch.save(logodds_disc, tensordir_disc)
    torch.save(logodds_gen, tensordir_gen)
    print("\n\n")

if __name__ == "__main__":
    main()
