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
import math
import random
import argparse

from datasets import load_dataset

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils
from utils import make_prompt_triviaqa, make_prompt_hypernymy, make_prompt_swords, make_prompt_lambada, get_final_logit_prob

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
        if 'gemma' in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    tokenizer, model = load_model_tokenizer(model_name)
    model.to(device)  # Move model to device right after creation

    if WITH_REF:
        model_ref = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
        model_ref.to(device)  # Move model_ref to the same device as model
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

    elif train_g_or_d == 'i':
        raise NotImplementedError("TODO implement this")
    else:
        raise NotImplementedError("train_g_or_d needs to be 'g', 'd', or 'i'.")


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
            else:
                raise ValueError("No.")
            # Use the first token after the space
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

        breakpoint()
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
            else:
                raise ValueError("No.")

            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts
        #p_train_disc, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
        #breakpoint()
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)

    elif task=='swords':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.synonym=='yes']
        # Generate generator prompts
        p_train_gold, hf_train_gen, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, neg=False, both=None)
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
            else:
                raise ValueError("No.")

            # Use the first token after the space
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

        breakpoint()

        #L_train_all = L_train  # Already using all examples for lambada
        # Generate generator prompts
        p_train_gold, hf_train_gen, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=gold_prompt_style, shots=gold_prompt_shots, both=None)
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
            else:
                raise ValueError("No.")
            # Use the first token after the space

            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            logprobs_last_layer.append(log_prob)

        # Generate discriminator prompts
        p_train_tune, hf_train, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style=tune_prompt_style, shots=tune_prompt_shots, neg=False, both=None)
        #prompts_pos = [i.prompt for i in p_train]
    else:
        raise ValueError("Task unsupported!")

    if with_chat:
        ms = [ [ {"role": "system", "content": "Answer directly without explanation."},  {"role": "user", "content": i.prompt.strip()} ] for i in p_train_tune]
        toks = tokenizer.apply_chat_template(ms, add_generation_prompt=True, padding=True, truncation=True, return_tensors='pt')
        max_context_length = toks.shape[1]
    else:
        #TODO later should make this cleaner in utils.make_and_format_data
        max_context_length = len(hf_train[0]['input_ids'])
    print("MAX CONTEXT LENGTH: ", max_context_length)

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
            (prompt_i, prompt_j), (token_i, token_j) = self.pairs[idx]
            # Debug print
            #print(f"\nProcessing item {idx}:")
            #print("Token types:", type(token_i), type(token_j))
            #print("Tokens:", token_i, token_j)

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
            batch_size = 6
        elif task=='trivia-qa':
            batch_size = 6
        elif task=='lambada':
            batch_size = 6
        elif task =='hypernym':
            batch_size = 32
        else:
            raise ValueError("define batch size for this case")

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
            else:
                raise ValueError("not supported")

            split_type_str = "--"+ split_type

            save_directory = "../models/v5-" + model_name.replace('/','--')  + "-delta"+str(delta)+"-epoch"+str(epoch) + "--" + task + with_ref_str + all_str + direction_str + split_type_str
            print("Saving to ", save_directory)
            model.save_pretrained(save_directory)
            tokenizer.save_pretrained(save_directory)

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            # Move all inputs to device
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
    parser.add_argument("--train_g_or_d", type=str, default='d', choices=["d","g","iter"], help="Train generator or discriminator.")
    parser.add_argument("--split_type", type=str, default='random', choices=["random","hyper","both"], help="How to do train/test split. Only applies to hypernymy.")
    args = parser.parse_args()
    main(args)
