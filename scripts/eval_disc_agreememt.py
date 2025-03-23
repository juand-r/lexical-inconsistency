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
from utils import get_L_prompt, get_final_logit_prob, get_response
from logitlens import compute_logodds_final_layer, get_logodds_gen, get_logodds_disc, compute_disc_accuracy
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
    print("loading model_name:", model_name)
    if 'gemma' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch_dtype).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    print("model.config.torch_dtype:", model.config.torch_dtype)  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
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

def clear_response(response):
    response = response.strip()
    response = response.split(' ')[0]
    if response[-1] == ".":
        response = response[:-1]
    return response


def main(args):
    task = args.task
    modelname = args.model
    # do_tunedlens = args.tunedlens
    seed = args.seed
    gen_shots = args.gen_shots
    disc_shots = args.disc_shots
    print(f"gen_shots: {gen_shots}, disc_shots: {disc_shots}")
    # raise ValueError("Check this!")
    train_flag = args.train
    split_type = args.split_type

    L_train, L_test, make_prompt = get_L_prompt(task, split_type, seed)

    device = "cuda"

    init_model(modelname, device)
    
    # TODO: test with base model, check the values
#    if "gpt" in modelname.lower():
#        first_sw_token = 1
#    elif "gemma" in modelname.lower() or "llama" in modelname.lower():
#        first_sw_token = 2
#    else:
#        raise ValueError("!?")

    #NOTE assume we just do llama or gemma. Same situation in both:
    first_sw_token = 2

    model_is_chat = False
    if 'instruct' in modelname.lower():
        model_is_chat = True
        first_sw_token = 1
        print("Model is chat model!")
    if "gpt" in modelname.lower():
        raise ValueError("If you are using GPT then rewrite this bit!")

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
    
    response_gen_list = []
    response_disc_list = []
    P_disc = []
    dict_list = []
    json_list = []
    # LL = LL[:10]
    n = 0
    for item in tqdm(LL):
        prompt_gen = make_prompt(item, style='generator', shots=gen_shots).prompt
        # print(f"prompt_gen: {prompt_gen}")
        response_gen = get_response(prompt_gen, model, tokenizer, device, is_chat = model_is_chat) # TODO: change is_chat to True if instruction-tuned model
        # print(f"response_gen: {response_gen}")
        response_gen = clear_response(response_gen)
        response_gen_list.append(response_gen)

        prompt_disc = make_prompt(item, style='discriminator', shots=disc_shots, gen_response = response_gen).prompt
        # print(f"prompt_disc: {prompt_disc}")
        response_disc = get_response(prompt_disc, model, tokenizer, device, is_chat = model_is_chat)
        # print(f"response_disc: {response_disc}")
        prob_disc = get_final_logit_prob(prompt_disc, model, tokenizer, device, is_chat = model_is_chat)
        P_disc.append(prob_disc)
        response_disc_list.append(response_disc)

        dict_list.append({"item": item, 
                          "prompt_gen": prompt_gen, "prompt_disc": prompt_disc,
                          "response_gen": response_gen, "response_disc": response_disc})


        # n += 1
        # if n > 10:
        #     break
    
    
    

    logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]
    for i, item in enumerate(logodds_disc):
        dict_list[i]['logodds_disc']  = float(logodds_disc[i])
    # print(f"dics_list: {dict_list}")
    golds = [1 for i in range(len(P_disc))]
    disc_acc, disc_roc = compute_disc_accuracy(golds, logodds_disc)

    file_name = os.path.join("../outputs/disc_agreement", f"disc_agg_{get_base_model_name(args.model)}_{args.task}.json")
    print("Storing json file to {}".format(file_name))

    with open(file_name, 'w') as json_file:
        json.dump(dict_list, json_file, indent=4)

    print(f"disc_acc: {disc_acc}, disc_roc: {disc_roc}")
    # print(f"response_disc_list: {response_disc_list}")

    summary_file = os.path.join("../outputs/disc_agreement/eval_aggr_results.csv")
    with open(summary_file, 'a') as f:
        # if file is empty:
        if os.stat(summary_file).st_size == 0:
            f.write("model,task,disc_agg,gen_shots,disc_shots\n")
        f.write(f"{get_base_model_name(args.model)},{args.task},{disc_acc},{args.gen_shots},{args.disc_shots}\n")




if __name__ == "__main__":
    print("Before parser")
    parser = argparse.ArgumentParser(description="Compute log-odds on test data")
    parser.add_argument("--model", type=str, help="model directory to process (this should contain merged/ subdirectory) or hf model")
    # parser.add_argument("--tunedlens", action="store_true", default=False, help="")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling the data.")
    parser.add_argument("--disc_shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--gen_shots", type=str, default='zero', help="'zero' vs 'few'")
    parser.add_argument("--train", action="store_true", default=False, help="log-odds of train or test set?")
    parser.add_argument("--split_type", type=str, default='random', help="'random' vs 'hyper' vs 'both' ")
    parser.add_argument("--task", type=str, default='hypernym', help="hypernym, trivia-qa, etc")

    args = parser.parse_args()
    print("Before main")
    main(args)

'''
CUDA_VISIBLE_DEVICES=1 python eval_disc_agreememt.py --model meta-llama/Llama-3.2-3B-Instruct  --task trivia-qa
'''