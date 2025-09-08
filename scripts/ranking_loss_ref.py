"""
This script is used to train a model to rank discriminator prompts to match the ranking of log-probabilities of generator prompts.

Usage:
python ranking_loss_ref.py --model google/gemma-2-2b --task hypernym --with_ref --num_epochs 10 --learning_rate 1e-5 --delta 5 --total_samples 5110 --save_steps 1

"""
import os
import sys
import itertools
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from peft import LoraConfig, get_peft_model
import math
import random
import argparse

from datasets import load_dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils
from utils import make_prompt_triviaqa, make_prompt_hypernymy, make_prompt_swords, make_prompt_lambada, get_final_logit_prob

# good_pair, alpha_fun_1, get_alpha are used for "both" mode
def good_pair(log_prob_i, log_prob_j, label_i, label_j):
    """Determine if a pair is good based on log probabilities and labels."""
    if log_prob_i > log_prob_j and label_i == 1 and label_j == 0:
        return True
    elif log_prob_i <= log_prob_j and label_i == 0 and label_j == 1:
        return True
    else:
        return False

def alpha_fun_1(gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j):
    """Determine alpha based on which model has the good pair."""
    if good_pair(gen_log_prob_i, gen_log_prob_j, label_i, label_j) and not good_pair(disc_log_prob_i, disc_log_prob_j, label_i, label_j):
        return 1.0  # g->v direction
    elif good_pair(disc_log_prob_i, disc_log_prob_j, label_i, label_j) and not good_pair(gen_log_prob_i, gen_log_prob_j, label_i, label_j):
        return 0.0  # v->g direction
    else:
        return 0.5  # equal weighting

def get_alpha(alpha_arg, gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j):
    """Get alpha value based on argument and sample characteristics."""
    if isinstance(alpha_arg, (int, float)):
        return float(alpha_arg)
    elif alpha_arg == "alpha_fun_1":
        return alpha_fun_1(gen_log_prob_i, gen_log_prob_j, disc_log_prob_i, disc_log_prob_j, label_i, label_j)
    else:
        raise ValueError(f"Unknown alpha function: {alpha_arg}")

def main(args):
    model_name = args.model
    task = args.task
    with_ref = args.with_ref
    num_epochs = args.num_epochs
    lr = args.learning_rate
    delta = args.delta
    #TODO set delta automatically based on data?
    total_samples = args.total_samples
    save_steps = args.save_steps
    use_all = args.all  # New flag for using all examples
    train_g_or_d = args.train_g_or_d
    split_type = args.split_type
    alpha = args.alpha  # New alpha parameter
    use_lora = args.lora
    gradient_checkpointing = args.gradient_checkpointing
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    WITH_REF = with_ref

    if 'Instruct' in model_name or 'instruct' in model_name:
        with_chat = True
        print(f"Detected instruct model: {model_name}")
        print("Using chat template formatting for prompts")
        disc_shots = "zero"
        space_prefix = ""
    else:
        with_chat = False
        disc_shots = "few"
        space_prefix = " "
        print(f"Using standard formatting for model: {model_name}")


    # Define device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_model_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use memory-efficient loading for large models
        if 'gemma' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="eager", 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif 'llama' in model_name.lower() or '8B' in model_name or '7B' in model_name:
            # For large Llama models, use more aggressive memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
                # Note: flash_attention_2 requires separate installation
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    tokenizer, model = load_model_tokenizer(model_name)
    # Note: device_map="auto" in load_model_tokenizer handles device placement
    
    # Conditionally add LoRA for memory-efficient fine-tuning
    if use_lora:
        print("Setting up LoRA for memory-efficient fine-tuning...")
        lora_config = LoraConfig(
            r=16,  # Low-rank dimension
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Llama target modules
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Show how many parameters we're actually training
    else:
        print("Using full model fine-tuning (no LoRA)")
        # For full model fine-tuning, ensure model is on correct device if device_map didn't handle it
        if not hasattr(model, 'hf_device_map'):
            model.to(device)
    
    # Enable gradient checkpointing to save memory (optional)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Gradient checkpointing disabled")

    if WITH_REF:
        print("Loading reference model with memory optimizations...")
        if 'gemma' in model_name.lower():
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                attn_implementation="eager", 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif 'llama' in model_name.lower() or '8B' in model_name or '7B' in model_name:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            model_ref = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
    else:
        model_ref = None

    # see if padding right works??
    tokenizer.padding_side = 'left'

    # assume label of 1 meaning left < right in ground truth

    # NOTE first use ground truth ranking from generator.
    # Will now use ranking loss on *discriminator* prompts to try to match it!

    if task=='hypernym':
        L = utils.load_noun_pair_data()
        if split_type=='hyper':
            L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
        elif split_type=='random':
            L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
        elif split_type=='both':
            L_train, L_test = utils.split_train_test_no_overlap_both(L, seed=2)
        else:
            raise ValueError("Wrong value for split-type")
        #L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
        #L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
        #L_train, L_test = utils.split_train_test_no_overlap_both(L)
    elif task=='trivia-qa':
        #USE SUBSET FOR NOW
#        L_train =  L['train'].shuffle(seed=42).select(range(3000))
#        L_test = L['validation'].shuffle(seed=42).select(range(1000))
        L_train, L_test, _ = utils.get_L_prompt('trivia-qa', split_type, seed=0)
    elif task=='swords':
        L_train, L_test = utils.load_swords_data(seed=0)
    elif task=='lambada':
        #L_train, L_test = utils.load_lambada_data(seed=0)
        # experiment with negatives -- recent version of get_L_prompt does this
        L_train, L_test, _ = utils.get_L_prompt('lambada', split_type, seed=0)
    else:
        raise NotImplementedError("Task not implemented!")

    print("Computing log-probabilities on the fly...")
    print(f"Using device: {device}")

    if train_g_or_d=='d':
        # Assume the generator is absolutely correct and try to match it.
        gold_prompt_style = 'generator'
        gold_prompt_shots = 'zero' #always using zero-shot for generator prompts

        tune_prompt_style = 'discriminator'
        tune_prompt_shots = disc_shots #depends on whether is instruct tuned model

    elif train_g_or_d =='g':
        # Assume the discriminator is absolutely correct and try to match generator to it.
        gold_prompt_style = 'discriminator'
        gold_prompt_shots = disc_shots #always using zero-shot for generator prompts

        tune_prompt_style = 'generator'
        tune_prompt_shots = 'zero'

    elif train_g_or_d == 'both':
        # For this mode, we train on a combination of generator and discriminator prompts
        # NOTE: gold_prompt_style and tune_prompt_style might be misleading in this setting.
        gold_prompt_style = 'generator'
        gold_prompt_shots = 'zero'
        
        tune_prompt_style = 'discriminator'
        tune_prompt_shots = disc_shots

    elif train_g_or_d == 'i':
        raise NotImplementedError("TODO implement this")
    else:
        raise NotImplementedError("train_g_or_d needs to be 'g', 'd', 'both', or 'i'.")


    if task=='hypernym':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.taxonomic == "yes"]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for presumed "gold truth" prompts (when training discriminator, these are generator prompts)
        # if trainin disc, log_probs_last_layer_pos are for generator prompt
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (noun2)
            # For hypernymy, we want the probability of the noun2 token
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx].noun2
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx].noun2
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts if train_g_or_d == 'd'.  Previously was p_train_disc
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]

    elif task=='trivia-qa':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i['correct']=='yes']

        #L_train_all = L_train  # Already using all examples for trivia-qa
        # Generate generator prompts
        #p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='generator', shots='zero', both=None)
        #prompts_gen = [i.prompt for i in p_train_gen]
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            if train_g_or_d=='d':
                # Get the log probability for the target token (answer)
                target_text = space_prefix + L_train_all[idx]['answers'][0].capitalize()
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['answers'][0].capitalize()
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts
        #p_train_disc, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)

    elif task=='swords':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.synonym=='yes']
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (replacement)
            #target_text = space_prefix + L_train_all[idx].replacement
            #target_tokens = tokenizer.encode(target_text)
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx].replacement
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx].replacement
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)
        # Generate discriminator prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]

    elif task=='lambada':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i['correct']=='yes']


        #L_train_all = L_train  # Already using all examples for lambada
        # Generate generator prompts
        p_train_gold, hf_train_gold, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, both=None)
 
        prompts_gold = [i.prompt for i in p_train_gold]

        # Compute log-probabilities for generator prompts
        logprobs_last_layer = []
        for idx, prompt in enumerate(tqdm(prompts_gold)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (final_word)
            #target_text = space_prefix + L_train_all[idx]['final_word']
            #target_tokens = tokenizer.encode(target_text)
            if train_g_or_d=='d':
                target_text = space_prefix + L_train_all[idx]['final_word']
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='g':
                target_text = space_prefix +"Yes"
                target_tokens = tokenizer.encode(target_text)
            elif train_g_or_d=='both':
                # For both mode, we use the same target as discriminator mode
                target_text_d = space_prefix + L_train_all[idx]['final_word']
                target_tokens_d = tokenizer.encode(target_text_d)

                target_text_g = space_prefix + "Yes"
                target_tokens_g = tokenizer.encode(target_text_g)
            else:
                raise ValueError("No.")
            # Use the first token after the space
            if train_g_or_d == 'both':
                ind_d = target_tokens_d[0] if len(target_tokens_d) == 1 else target_tokens_d[1]
                ind_g = target_tokens_g[0] if len(target_tokens_g) == 1 else target_tokens_g[1]
                log_prob_d = math.log(probs[ind_d].item() + 1e-12)
                log_prob_g = math.log(probs[ind_g].item() + 1e-12)
                logprobs_last_layer.append((log_prob_d, log_prob_g))
                #NOTE careful these contain tuples of (log_prob_d, log_prob_g)
            else:
                ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
                log_prob = math.log(probs[ind].item() + 1e-12)
                logprobs_last_layer.append(log_prob)
        # Generate discriminator prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
    else:
        raise ValueError("Task unsupported!")

    if with_chat:
        # Process discriminator prompts (p_train_tune)
        ms_tune = [ [ {"role": "system", "content": "Answer directly without explanation."},  {"role": "user", "content": i.prompt.strip()} ] for i in p_train_tune]
        toks_tune = tokenizer.apply_chat_template(ms_tune, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
        max_context_length = toks_tune.shape[1]
        
        # If mode is 'both', also process generator prompts (p_train_gold) and take the maximum
        if train_g_or_d == 'both':
            ms_gold = [ [ {"role": "system", "content": "Answer directly without explanation."},  {"role": "user", "content": i.prompt.strip()} ] for i in p_train_gold]
            toks_gold = tokenizer.apply_chat_template(ms_gold, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
            max_context_length = max(max_context_length, toks_gold.shape[1])
    else:
        #TODO later should make this cleaner in utils.make_and_format_data
        max_context_length = len(hf_train[0]['input_ids'])
        if train_g_or_d == 'both':
            max_context_length = max(len(hf_train_gold[0]['input_ids']), max_context_length)
    print("MAX CONTEXT LENGTH: ", max_context_length)

    if train_g_or_d == 'both':
        # Create tuples of (discriminator_prompt, generator_prompt, logprobs)
        # Note: logprobs_last_layer contains tuples of (log_prob_d, log_prob_g)
        Z = list(zip(p_train_tune, p_train_gold, logprobs_last_layer))
        
        # Sort based on discriminator logprob (first element of the logprobs tuple)
        Z = sorted(Z, key=lambda i: i[2][0])  # Using i[2][0] to get the discriminator logprob

        # Calculate delta based on range of discriminator logprobs
        min_logprob = Z[0][2][0]  # Minimum discriminator logprob
        max_logprob = Z[-1][2][0]  # Maximum discriminator logprob

        print(f"Delta (minimum separation): {delta}")
        if delta!=0:
            NN = (max_logprob - min_logprob) / delta
            print(f"NN: {NN}")
        print(f"Min logprob: {min_logprob}")
        print(f"Max logprob: {max_logprob}")

        indices = range(len(Z))
        pair_inds = list(itertools.product(indices, repeat=2))
        pair_inds = [i for i in pair_inds if i[0] < i[1]]
        pair_inds = random.sample(pair_inds, total_samples)
        
        # Create pairs with all the information
        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]
    else:
        #Z = list(zip(prompts_pos, gen_logprobs_last_layer))
        Z = list(zip(p_train_tune, logprobs_last_layer))
        Z = sorted(Z, key = lambda i: i[-1])

        # Calculate delta based on range of logprobs
        min_logprob = Z[0][1]
        max_logprob = Z[-1][1]

        print(f"Delta (minimum separation): {delta}")
        if delta!=0:
            NN = (max_logprob - min_logprob) / delta
            print(f"NN: {NN}")
        print(f"Min logprob: {min_logprob}")
        print(f"Max logprob: {max_logprob}")

        indices = range(len(Z))
        pair_inds = list(itertools.product(indices, repeat=2))
        pair_inds = [i for i in pair_inds if i[0] < i[1]]
        pair_inds = random.sample(pair_inds, total_samples)
        pairs_ = [(Z[i[0]], Z[i[1]]) for i in pair_inds]


    def format_with_inst(prompt):
        message = [
            {"role": "system", "content": "Answer directly without explanation."},
            {"role": "user", "content": prompt},]
        toks = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')[0]
        return tokenizer.decode(toks[1:])


    if train_g_or_d=='d':
        #NOTE in this case the tokens we are targeting are the "Yes" tokens in both cases.
        token_id = tokenizer.encode(space_prefix +"Yes")[-1]

        if with_chat:
             pairs = [(  ( format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)),  (token_id, token_id) ) for pair in pairs_
                if pair[1][1] - pair[0][1] > delta]
        else:
            pairs = [((pair[0][0].prompt ,pair[1][0].prompt),  (token_id, token_id) ) for pair in pairs_
                if pair[1][1] - pair[0][1] > delta]
    elif train_g_or_d=='g':
        #NOTE in this case the ranking is derived from the log-probs of Yes under both prompts but we are targetting
        # the log-odds (hopefully log-prob is fine here) of the *generator completion*, so not the same in each item of the pair!
        if with_chat:
             pairs = [(  ( format_with_inst(pair[0][0].prompt),  format_with_inst(pair[1][0].prompt)),  (tokenizer.encode(pair[0][0].completion)[1], tokenizer.encode(pair[1][0].completion)[1] )     ) for pair in pairs_
                if pair[1][1] - pair[0][1] > delta]
        else:
            pairs = [(   (pair[0][0].prompt, pair[1][0].prompt) , (tokenizer.encode(pair[0][0].completion)[1], tokenizer.encode(pair[1][0].completion)[1] )   ) for pair in pairs_  if pair[1][1] - pair[0][1] > delta]
    elif train_g_or_d == 'both':
        # For both mode, we create pairs for both generator and discriminator training
        # First create discriminator pairs (targeting "Yes" tokens)
        token_id = tokenizer.encode(space_prefix +"Yes")[-1]
        if with_chat:
            # Create pairs with both discriminator and generator prompts, applying chat formatting
            # NOTE verify fixed
            pairs = [
                (
                    ((format_with_inst(pair[0][0].prompt), format_with_inst(pair[1][0].prompt)), (token_id, token_id)),  # discriminator pair
                    ((format_with_inst(pair[0][1].prompt), format_with_inst(pair[1][1].prompt)), (tokenizer.encode(pair[0][1].completion)[1], tokenizer.encode(pair[1][1].completion)[1])),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   )
                ) for pair in pairs_ if pair[1][-1][0] - pair[0][-1][0] > delta
            ]
        else:
            # Create pairs with both discriminator and generator prompts
            pairs = [
                (
                    ((pair[0][0].prompt, pair[1][0].prompt), (token_id, token_id)),  # discriminator pair
                    ((pair[0][1].prompt, pair[1][1].prompt), (tokenizer.encode(pair[0][1].completion)[1], tokenizer.encode(pair[1][1].completion)[1])),  # generator pair
                    (pair[0][0].completion.strip().lower()   , pair[1][0].completion.strip().lower()   )
                ) for pair in pairs_ if pair[1][-1][0] - pair[0][-1][0] > delta
            ]

    else:
        raise ValueError("TODO!")

    print(pairs[0])
    print("\n\n")
    print(pairs[1])
    print("\n\nNum Samples: ", len(pairs))

    class PairwiseDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_length=128, device='cuda'):
            """
            pairs: list of ((prompt_i, prompt_j), (token_i, token_j))
            tokenizer: Hugging Face tokenizer
            device: device to place tensors on
            """
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device

            # Debug print first pair
            #print("\nDebugging PairwiseDataset initialization:")
            #print("First pair:", pairs[0])
            #print("Token types:", type(pairs[0][1][0]), type(pairs[0][1][1]))
            #print("Tokens:", pairs[0][1][0], pairs[0][1][1])

            # Try to encode the tokens
            #print("\nTrying to encode tokens:")
            #try:
            #    print("Encoding first token:", tokenizer.encode(pairs[0][1][0]))
            #    print("Encoding second token:", tokenizer.encode(pairs[0][1][1]))
            #except Exception as e:
            #    print("Error encoding tokens:", e)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            if train_g_or_d == 'both':
                ((prompt_i_disc, prompt_j_disc), (token_i_disc, token_j_disc)), ((prompt_i_gen, prompt_j_gen), (token_i_gen, token_j_gen)), (label_i, label_j) = self.pairs[idx]
            else:
                (prompt_i, prompt_j), (token_i, token_j) = self.pairs[idx]
            # Debug print
            #print(f"\nProcessing item {idx}:")
            #print("Token types:", type(token_i), type(token_j))
            #print("Tokens:", token_i, token_j)
            if train_g_or_d == 'both':
                # Tokenize discriminator prompts
                enc_i_disc = self.tokenizer(
                    prompt_i_disc,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                enc_j_disc = self.tokenizer(
                    prompt_j_disc,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Tokenize generator prompts
                enc_i_gen = self.tokenizer(
                    prompt_i_gen,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                enc_j_gen = self.tokenizer(
                    prompt_j_gen,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
            # Get labels from prompts
            #    label_i = "yes" if "yes" in prompt_i.lower() else "no"
            #    label_j = "yes" if "yes" in prompt_j.lower() else "no"



            else:
                # Tokenize prompt i
                enc_i = self.tokenizer(
                    prompt_i,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                # Tokenize prompt j
                enc_j = self.tokenizer(
                    prompt_j,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

            if train_g_or_d != 'both':
                # Squeeze to remove the batch dimension (shape: [seq_len])
                item = {
                    'input_ids_i': enc_i['input_ids'].squeeze(0),
                    'attention_mask_i': enc_i['attention_mask'].squeeze(0),
                    'token_id_i': torch.tensor(token_i, dtype=torch.long),
                    'input_ids_j': enc_j['input_ids'].squeeze(0),
                    'attention_mask_j': enc_j['attention_mask'].squeeze(0),
                    'token_id_j': torch.tensor(token_j, dtype=torch.long),
                    'label': torch.tensor(1.0, dtype=torch.float)
                }
            else:
                item = {
                    'input_ids_i_disc': enc_i_disc['input_ids'].squeeze(0),
                    'attention_mask_i_disc': enc_i_disc['attention_mask'].squeeze(0),
                    'token_id_i_disc': torch.tensor(token_i_disc, dtype=torch.long),
                    'input_ids_j_disc': enc_j_disc['input_ids'].squeeze(0),
                    'attention_mask_j_disc': enc_j_disc['attention_mask'].squeeze(0),
                    'token_id_j_disc': torch.tensor(token_j_disc, dtype=torch.long),
                    'input_ids_i_gen': enc_i_gen['input_ids'].squeeze(0),
                    'attention_mask_i_gen': enc_i_gen['attention_mask'].squeeze(0),
                    'token_id_i_gen': torch.tensor(token_i_gen, dtype=torch.long),
                    'input_ids_j_gen': enc_j_gen['input_ids'].squeeze(0),
                    'attention_mask_j_gen': enc_j_gen['attention_mask'].squeeze(0),
                    'token_id_j_gen': torch.tensor(token_j_gen, dtype=torch.long),
                    'label_i': torch.tensor(1.0 if label_i == "yes" else 0.0, dtype=torch.float),
                    'label_j': torch.tensor(1.0 if label_j == "yes" else 0.0, dtype=torch.float)
            }
            return item


    #18 fine for zero-shot
    if with_ref:
        if task=='swords':
            batch_size = 2
        elif task=='trivia-qa':
            batch_size = 2
        elif task=='lambada':
            batch_size = 2
        elif task =='hypernym':
            batch_size = 1 #4
        else:
            raise ValueError("define batch size for this case")
    else:
        if task=='swords':
            batch_size = 1#6
        elif task=='trivia-qa':
            batch_size = 2#6
        elif task=='lambada':
            batch_size = 2#6
        elif task =='hypernym':
            batch_size = 2#6#1  # Reduced from 32 to 1 for large models
        else:
            raise ValueError("define batch size for this case")

    if max_context_length > 90:
        max_context_length = 90

    dataset = PairwiseDataset(pairs, tokenizer, max_length=max_context_length, device=device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("\n\nDone making dataloader\n\n")
    optimizer = AdamW(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        #if True:#epoch % save_steps==1:
        if epoch!=0:
            with_ref_str = "-with-ref" if with_ref else ""
            all_str = "-all" if use_all else ""

            if train_g_or_d == 'd':
                direction_str = '--g2d'
            elif train_g_or_d == 'g':
                direction_str = '--d2g'
            elif train_g_or_d == 'iter':
                direction_str = '--iter'
            elif train_g_or_d == 'both':
                direction_str = '--both'
            else:
                raise ValueError("not supported")

            split_type_str = "--"+ split_type

            alpha_str = "--alpha" + str(alpha) if isinstance(alpha, (int, float)) else "--alpha-" + str(alpha)
            save_directory = "../models/v5-" + model_name.replace('/','--')  + "-delta"+str(delta)+"-epoch"+str(epoch) + "--" + task + with_ref_str + all_str + direction_str + split_type_str + alpha_str
            print("Saving to ", save_directory)
            
            if use_lora:
                # For LoRA: Save adapters first, then merge and save full model
                print("Saving LoRA adapters...")
                model.save_pretrained(save_directory)
                
                print("Loading saved LoRA model...")
                from peft import AutoPeftModelForCausalLM
                model_peft = AutoPeftModelForCausalLM.from_pretrained(save_directory)
                
                print("Merging LoRA into base model...")
                merged_model = model_peft.merge_and_unload()
                
                merge_dir = save_directory + "_merged"
                print(f"Saving merged full model to {merge_dir}")
                merged_model.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="2GB")
                tokenizer.save_pretrained(merge_dir)
                
                # Clean up merged model from memory
                del model_peft, merged_model
                torch.cuda.empty_cache()
            else:
                # For full model fine-tuning: Save normally
                model.save_pretrained(save_directory)
                tokenizer.save_pretrained(save_directory)

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            # Move all inputs to device
            if train_g_or_d == 'both':
                # Get discriminator inputs
                input_ids_i_disc = batch["input_ids_i_disc"].to(device)
                attention_mask_i_disc = batch["attention_mask_i_disc"].to(device)
                token_id_i_disc = batch["token_id_i_disc"].to(device)
                input_ids_j_disc = batch["input_ids_j_disc"].to(device)
                attention_mask_j_disc = batch["attention_mask_j_disc"].to(device)
                token_id_j_disc = batch["token_id_j_disc"].to(device)

                # Get generator inputs
                input_ids_i_gen = batch["input_ids_i_gen"].to(device)
                attention_mask_i_gen = batch["attention_mask_i_gen"].to(device)
                token_id_i_gen = batch["token_id_i_gen"].to(device)
                input_ids_j_gen = batch["input_ids_j_gen"].to(device)
                attention_mask_j_gen = batch["attention_mask_j_gen"].to(device)
                token_id_j_gen = batch["token_id_j_gen"].to(device)

                label_i = batch["label_i"].to(device)
                label_j = batch["label_j"].to(device)

                # Forward pass for discriminator prompts
                outputs_i_disc = model(input_ids=input_ids_i_disc, attention_mask=attention_mask_i_disc)
                outputs_j_disc = model(input_ids=input_ids_j_disc, attention_mask=attention_mask_j_disc)
                
                # Forward pass for generator prompts
                outputs_i_gen = model(input_ids=input_ids_i_gen, attention_mask=attention_mask_i_gen)
                outputs_j_gen = model(input_ids=input_ids_j_gen, attention_mask=attention_mask_j_gen)

                # Get logits for discriminator
                last_idx_disc = attention_mask_i_disc.size(1) - 1
                selected_logits_i_disc = torch.cat([outputs_i_disc.logits[b, last_idx_disc, :].unsqueeze(0) for b in range(outputs_i_disc.logits.size(0))], dim=0)
                selected_logits_j_disc = torch.cat([outputs_j_disc.logits[b, last_idx_disc, :].unsqueeze(0) for b in range(outputs_j_disc.logits.size(0))], dim=0)
                
                # Get logits for generator
                last_idx_gen = attention_mask_i_gen.size(1) - 1
                selected_logits_i_gen = torch.cat([outputs_i_gen.logits[b, last_idx_gen, :].unsqueeze(0) for b in range(outputs_i_gen.logits.size(0))], dim=0)
                selected_logits_j_gen = torch.cat([outputs_j_gen.logits[b, last_idx_gen, :].unsqueeze(0) for b in range(outputs_j_gen.logits.size(0))], dim=0)

                # Compute log probabilities
                log_probs_i_disc = F.log_softmax(selected_logits_i_disc, dim=-1)
                log_probs_j_disc = F.log_softmax(selected_logits_j_disc, dim=-1)
                log_probs_i_gen = F.log_softmax(selected_logits_i_gen, dim=-1)
                log_probs_j_gen = F.log_softmax(selected_logits_j_gen, dim=-1)

                # Get scores
                score_i_disc = log_probs_i_disc[torch.arange(log_probs_i_disc.size(0), device=device), token_id_i_disc]
                score_j_disc = log_probs_j_disc[torch.arange(log_probs_j_disc.size(0), device=device), token_id_j_disc]
                score_i_gen = log_probs_i_gen[torch.arange(log_probs_i_gen.size(0), device=device), token_id_i_gen]
                score_j_gen = log_probs_j_gen[torch.arange(log_probs_j_gen.size(0), device=device), token_id_j_gen]

                # Use frozen reference model if needed
                if WITH_REF:
                    raise ValueError("Do LATER")
                else:
                    diff_ref = 0

                # Compute both G->V and V->G losses
                g2v_diff = score_j_disc - score_i_disc - diff_ref
                v2g_diff = score_j_gen - score_i_gen - diff_ref

                # Get alpha for each sample in the batch
                alphas = []
                for b in range(batch["input_ids_i_disc"].size(0)):
                    alpha_val = get_alpha(alpha, 
                                    score_i_gen[b].item(), score_j_gen[b].item(),
                                    score_i_disc[b].item(), score_j_disc[b].item(),
                                    label_i[b].item(), label_j[b].item())
                    alphas.append(alpha_val)
                alphas = torch.tensor(alphas, device=device)

                # Compute weighted loss
                g2v_loss = -torch.log(torch.sigmoid(g2v_diff) + 1e-12)
                v2g_loss = -torch.log(torch.sigmoid(v2g_diff) + 1e-12)
                loss = (alphas * g2v_loss + (1 - alphas) * v2g_loss).mean()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Clear cache to prevent memory accumulation
                torch.cuda.empty_cache()
            else:
                input_ids_i = batch["input_ids_i"].to(device)
                attention_mask_i = batch["attention_mask_i"].to(device)
                token_id_i = batch["token_id_i"].to(device)

                input_ids_j = batch["input_ids_j"].to(device)
                attention_mask_j = batch["attention_mask_j"].to(device)
                token_id_j = batch["token_id_j"].to(device)

                label = batch["label"].to(device)

                # Forward pass for prompt i
                outputs_i = model(input_ids=input_ids_i, attention_mask=attention_mask_i)
                # logits_i: [batch_size, seq_len, vocab_size]
                logits_i = outputs_i.logits

                last_idx_i = attention_mask_i.size(1) - 1

                # Gather the logits for the chosen position and compute log-softmax
                # shape [B, vocab_size]
                selected_logits_i = []
                for b in range(logits_i.size(0)):
                    #selected_logits_i.append(logits_i[b, last_idx_i[b], :].unsqueeze(0))
                    selected_logits_i.append(logits_i[b, last_idx_i, :].unsqueeze(0))
                selected_logits_i = torch.cat(selected_logits_i, dim=0)

                log_probs_i = F.log_softmax(selected_logits_i, dim=-1)  # [B, vocab_size]

                # Score for example i is the log-prob of token_id_i
                # shape [B]
                #score_i = log_probs_i[torch.arange(log_probs_i.size(0)), token_id_i]
                score_i = log_probs_i[torch.arange(log_probs_i.size(0), device=device), token_id_i]

                # Forward pass for prompt j
                outputs_j = model(input_ids=input_ids_j, attention_mask=attention_mask_j)
                logits_j = outputs_j.logits

                #last_idx_j = attention_mask_j.sum(dim=1) - 1
                last_idx_j = attention_mask_j.size(1) - 1 # assumes LEFT padding
                #print(last_idx_i)
                #print(last_idx_j)

                selected_logits_j = []
                for b in range(logits_j.size(0)):
                    #selected_logits_j.append(logits_j[b, last_idx_j[b], :].unsqueeze(0))
                    selected_logits_j.append(logits_j[b, last_idx_j, :].unsqueeze(0))

                selected_logits_j = torch.cat(selected_logits_j, dim=0)
                log_probs_j = F.log_softmax(selected_logits_j, dim=-1)  # [B, vocab_size]
                #score_j = log_probs_j[torch.arange(log_probs_j.size(0)), token_id_j]
                score_j = log_probs_j[torch.arange(log_probs_j.size(0), device=device), token_id_j]

                # Use frozen reference model
                if WITH_REF:
                    with torch.no_grad():
                        outputs_i_ref = model_ref(input_ids=input_ids_i, attention_mask=attention_mask_i)
                        # logits_i: [batch_size, seq_len, vocab_size]
                        logits_i_ref = outputs_i_ref.logits
                        last_idx_i = attention_mask_i.size(1) - 1
                        selected_logits_i_ref = []
                        for b in range(logits_i_ref.size(0)):
                            selected_logits_i_ref.append(logits_i_ref[b, last_idx_i, :].unsqueeze(0))
                        selected_logits_i_ref = torch.cat(selected_logits_i_ref, dim=0)
                        log_probs_i_ref = F.log_softmax(selected_logits_i_ref, dim=-1)  # [B, vocab_size]
                        #score_i_ref = log_probs_i_ref[torch.arange(log_probs_i_ref.size(0)), token_id_i]
                        score_i_ref = log_probs_i_ref[torch.arange(log_probs_i_ref.size(0), device=device), token_id_i]

                        # Forward pass for prompt j
                        outputs_j_ref = model_ref(input_ids=input_ids_j, attention_mask=attention_mask_j)
                        logits_j_ref = outputs_j_ref.logits

                        last_idx_j = attention_mask_j.size(1) - 1 # assumes LEFT padding
                        selected_logits_j_ref = []
                        for b in range(logits_j_ref.size(0)):
                            selected_logits_j_ref.append(logits_j_ref[b, last_idx_j, :].unsqueeze(0))
                        selected_logits_j_ref = torch.cat(selected_logits_j_ref, dim=0)
                        log_probs_j_ref = F.log_softmax(selected_logits_j_ref, dim=-1)  # [B, vocab_size]
                        #score_j_ref = log_probs_j_ref[torch.arange(log_probs_j_ref.size(0)), token_id_j]
                        score_j_ref = log_probs_j_ref[torch.arange(log_probs_j_ref.size(0), device=device), token_id_j]
                    diff_ref = score_j_ref - score_i_ref
                else:
                    diff_ref = 0

                # Pairwise logistic loss: - log( sigmoid( (score_j) - (score_i) ) )
                diff = score_j - score_i - diff_ref
                loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Clear cache to prevent memory accumulation
                torch.cuda.empty_cache()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name/path")
    parser.add_argument("--task", type=str, choices=["hypernym", "trivia-qa", "swords", "lambada"], help="Task to run")
    parser.add_argument("--with_ref", default=False, action="store_true", help="Whether to use reference model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--delta", type=float, default=10, help="Delta")
    parser.add_argument("--total_samples", type=int, default=5110, help="Total samples")
    parser.add_argument("--save_steps", type=int, default=1, help="Save steps")
    parser.add_argument("--all", default=False, action="store_true", help="Whether to use all examples or just positive ones")
    parser.add_argument("--train_g_or_d", type=str, default='d', choices=["d","g","iter","both"], help="Train generator or discriminator.")
    parser.add_argument("--split_type", type=str, default='random', choices=["random","hyper","both"], help="How to do train/test split. Only applies to hypernymy.")
    parser.add_argument("--alpha", type=str, default='1.0', help="Alpha value or function name. Can be a number between 0 and 1, or 'alpha_fun_1'")
    parser.add_argument("--lora", action='store_true', help="Use LoRA for memory-efficient fine-tuning")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory (trades compute for memory)")
    args = parser.parse_args()
    
    # Convert alpha to float if it's a number
    try:
        alpha_val = float(args.alpha)
        if 0 <= alpha_val <= 1:
            args.alpha = alpha_val
    except ValueError:
        if args.alpha not in ["alpha_fun_1"]:
            raise ValueError("Alpha must be a number between 0 and 1, or one of: alpha_fun_1")
    
    main(args)
