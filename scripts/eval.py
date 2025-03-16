# model, dataset
# -> pearson correlation (all, pos, neg)
# -> acc (disc, gen, threshold)
# ROC for disc / gen accuract
# MRR for 
'''
CUDA_VISIBLE_DEVICES={} python eval.py --model {path to model or model name} --task {}
'''
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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
#Gemma3ForCausalLM
from utils import get_L_prompt, get_final_logit_prob
from logitlens import compute_logodds_final_layer, get_logodds_gen, get_logodds_disc
# from tunedlens import init_lens, obtain_prob_tensor

device = "cuda"
yes_words = ["Yes", " Yes", "YES", "yes", " yes"]
no_words = ["No", " No", "NO", "no", " no"]

# combine with logodds.py
# and viz.py piece of code

def init_model(model_name, device):
    global model
    global tokenizer
    global terminators 
    torch_dtype = "auto"#torch.bfloat16
    # if 'gemma-3' in model_name:
    #     model = Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    # el
    if 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch_dtype).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    print("model.config.torch_dtype:", model.config.torch_dtype)  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if "llama" in model_name:
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# def get_L_prompt(task, split_type, seed):
#     if task=='hypernym':
#         L = load_noun_pair_data()
#         if split_type=='hyper':
#             L_train, L_test = split_train_test_no_overlap(L, seed=seed)
#         elif split_type=='random':
#             L_train, L_test = split_train_test(L, seed=seed, subsample=False, num_train=3000)
#         elif split_type=='both':
#             L_train, L_test = split_train_test_no_overlap_both(L, seed=2)
#         else:
#             raise ValueError("Wrong value for split-type")
#         make_prompt = make_prompt_hypernymy
#     elif task=='trivia-qa':
#         # load data here
#         L = load_dataset('lucadiliello/triviaqa') #TODO check if this is correct version.
#         #USE SUBSET FOR NOW
#         L_train =  L['train'].shuffle(seed=42).select(range(3000))
#         L_test = L['validation'].shuffle(seed=42).select(range(1000))

#         #NOTE assumes this takes same arguments in each case
#         make_prompt = make_prompt_triviaqa
#     elif task=='swords':
#         L_train, L_test = load_swords_data(seed=0)
#         make_prompt = make_prompt_swords
#     else:
#         raise NotImplementedError("Not a task")
#     return L_train, L_test, make_prompt

def get_base_model_name(modelname):
    # TODO: improve this
    modelname = modelname.split("output")[-1]
    return modelname.replace('/', '-')

def main(args):
    task = args.task
    modelname = args.model
    # do_tunedlens = args.tunedlens
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    train_flag = args.train
    split_type = args.split_type

    L_train, L_test, make_prompt = get_L_prompt(task, split_type, seed)

    device = "cuda"

    init_model(modelname, device)
    
    # TODO: test with base model, check the values
    if "gpt" in modelname.lower():
        first_sw_token = 1
    elif "gemma" in modelname.lower() or "llama" in modelname.lower():
        first_sw_token = 2
    else:
        raise ValueError("!?")
    
    yestoks = [tokenizer.encode(i)[-1] for i in yes_words]
    notoks = [tokenizer.encode(i)[-1] for i in no_words]

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
    P_disc = []

    json_list = []
    for item in tqdm(LL):
        prompt_gen = make_prompt(item, style='generator', shots=gen_shots).prompt
        prompt_disc = make_prompt(item, style='discriminator', shots=disc_shots).prompt
        probs_gen = get_final_logit_prob(prompt_gen, model, tokenizer, device, is_chat = False) # TODO: change is_chat to True if instruction-tuned model
        P_gen.append(probs_gen)
        probs_disc = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat = False) # TODO: change is_chat to True if instruction-tuned model
        P_disc.append(probs_disc)
        if args.train:
            if task == 'hypernym':
                json_list.append({"noun1":item.noun1, "noun2":item.noun2, "taxonomic":item.taxonomic, "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0, 
                                "generator-completion": ' ' + item.noun2.strip(), "discriminator-gold-completion": ' ' + item.taxonomic.strip().capitalize()})
            elif task == 'trivia-qa':
                # print(item.keys())
                json_list.append({"question":item['question'], "answer":' ' + item['answers'][0].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": ' ' + item['answers'][0].strip(), "discriminator-gold-completion": ' Yes'})
            elif task == 'lambada':
                json_list.append({"context":item['context'], "completion":' ' + item['final_word'].strip(), "generator-prompt":prompt_gen, "discriminator-prompt":prompt_disc, "generator-log-prob":0, "discriminator-log-prob":0,
                                "generator-completion": ' ' + item['final_word'].strip(), "discriminator-gold-completion": ' Yes'})
            else:
                raise NotImplementedError("Not a task")
            # print(json_list[-1])
    if args.train:
        logodds_gen = [get_logodds_gen(P_gen, LL, ii, tokenizer, first_sw_token, task, use_lgo=False) for ii in range(len(P_gen))]
        logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, None) for ii in range(len(P_disc))]
        for jj in range(len(json_list)):
            json_list[jj]["generator-log-prob"] = float(logodds_gen[jj])
            json_list[jj]["discriminator-log-prob"] = float(logodds_disc[jj])
            # print(json_list[jj])
        print(f"Saving train data to ../data/{task}-train-{modelname.split('/')[-1]}.json")
        with open(f"../data/{task}-train-{modelname.split('/')[-1]}.json", 'w') as f:
            json.dump(json_list, f, indent=4)
        return

    gc.collect()
    torch.cuda.empty_cache()

    
    
    res_dict = compute_logodds_final_layer(task,
        P_gen, P_disc, LL, tokenizer, first_sw_token, yestoks, notoks)

    
    basename = get_base_model_name(modelname)
    # # os.path.basename(modeldir)
    # tensordir_disc = os.path.join("../outputs/logodds", basename + "--" + task + "--disc-"+disc_shots + train_suffix + split_suffix + ".pt")
    # tensordir_gen = os.path.join("../outputs/logodds", basename + "--" + task + "--gen-"+gen_shots + train_suffix + split_suffix + ".pt")
    # torch.save(logodds_disc, tensordir_disc)
    # torch.save(logodds_gen, tensordir_gen)
    # ranks_gen = os.path.join("../outputs/logodds", basename + "--" + task + "--gen-"+gen_shots + train_suffix + split_suffix + "--ranks.json")
    # with open(ranks_gen, 'w') as f:
    #     json.dump(ranks, f, indent=4)


    summary_file = os.path.join("../outputs/eval_results.csv")
    with open(summary_file, 'a') as f:
        # if file is empty:
        if os.stat(summary_file).st_size == 0:
            f.write("model,task,corr_all,corr_pos,corr_neg,disc_acc,disc_roc, gen_acc_5, gen_acc_10, gen_acc_40, gen_acc_100, gen_acc_1000,gen_mrr_pos, gen_mrr_neg, gen_shots,disc_shots,split,split_type,seed\n")
        split = "train" if args.train else "test"
        f.write(f"{modelname},{task},{res_dict['corr_all']},{res_dict['corr_pos']},{res_dict['corr_neg']},{res_dict['disc_acc']},{res_dict['disc_roc']},{res_dict['gen_acc_dict'][5]},{res_dict['gen_acc_dict'][10]},{res_dict['gen_acc_dict'][40]},{res_dict['gen_acc_dict'][100]},{res_dict['gen_acc_dict'][1000]},{res_dict['gen_mrr_pos']},{res_dict['gen_mrr_neg']},{gen_shots},{disc_shots},{split},{split_type},{seed}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log-odds on test data")
    parser.add_argument("--model", type=str, help="model directory to process (this should contain merged/ subdirectory) or hf model")
    # parser.add_argument("--tunedlens", action="store_true", default=False, help="")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling the data.")
    parser.add_argument("--disc-shots", type=str, default='few', help="'zero' vs 'few'")
    parser.add_argument("--gen-shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--train", action="store_true", default=False, help="log-odds of train or test set?")
    parser.add_argument("--split_type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument("--task", type=str, default='hypernym', help="hypernym, trivia-qa, etc")

    args = parser.parse_args()
    main(args)

'''
get data for dpo...
CUDA_VISIBLE_DEVICES=5 python eval.py --model google/gemma-3-4b-pt --train --task hypernym
CUDA_VISIBLE_DEVICES=6 python eval.py --model google/gemma-2-2b --train --task trivia-qa

CUDA_VISIBLE_DEVICES=3 python eval.py --model google/gemma-2-2b --train --task lambada
CUDA_VISIBLE_DEVICES=3 python eval.py --model meta-llama/Llama-3.2-3B --train --task lambada

CUDA_VISIBLE_DEVICES=3 python eval.py --model google/gemma-2-2b --train --task swords
CUDA_VISIBLE_DEVICES=3 python eval.py --model meta-llama/Llama-3.2-3B --train --task swords

model = {'google/gemma-2-2b', 'google/gemma-3-4b-pt', 'meta-llama/Llama-3.2-3B'}
'''