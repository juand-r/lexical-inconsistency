"""
This script is used to train a model to rank discriminator prompts to match the ranking of log-probabilities of generator prompts.

Usage:
python ranking_loss_ref.py --model google/gemma-2-2b --task hypernym --with_ref --num_epochs 10 --learning_rate 1e-5 --delta 5 --total_samples 5110 --save_steps 1

TODO
- Currently this is using the ground truth ranking from the generator.
  We should also try to use the ranking from the discriminator to align the generator.

- Also, currently we are using the log-probabilities of the last layer.
  We should also try to use the log-probabilities of the second to last layer.

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
    #TODO load model also, use function above
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
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    WITH_REF = with_ref

    if 'Instruct' in model_name or 'instruct' in model_name:
        with_chat = True
        print(f"Detected instruct model: {model_name}")
        print("Using chat template formatting for prompts")
        disc_shots = "zero"
    else:
        with_chat = False
        disc_shots = "few"
        print(f"Using standard formatting for model: {model_name}")

    #TODO generalize to other models other than llama!

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

    # Verify the model has chat template support
    try:
        test_message = [{"role": "user", "content": "Hello"}]
        tokenizer.apply_chat_template(test_message, add_generation_prompt=True)
        print("Chat template verified and working")
    except Exception as e:
        print(f"Warning: Model {model_name} doesn't support chat templates: {e}")
        print("Falling back to standard formatting")
        with_chat = False
        disc_shots = "few"

    if WITH_REF:
        model_ref = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
        model_ref.to(device)  # Move model_ref to the same device as model
    else:
        model_ref = None

    # see if padding right works??
    tokenizer.padding_side = 'left'

    #TODO assume generator is ground truth
    #TODO assume discriminator is ground truth

    # assume label of 1 meaning left < right in ground truth


    # NOTE first use ground truth ranking from generator.
    # Will now use ranking loss on *discriminator* prompts to try to match it!


    if task=='hypernym':
        L = utils.load_noun_pair_data()
        L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
        #L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
        #L_train, L_test = utils.split_train_test_no_overlap_both(L)
    elif task=='trivia-qa':
        L = load_dataset('lucadiliello/triviaqa') #TODO check if this is correct version.
        #USE SUBSET FOR NOW
        L_train =  L['train'].shuffle(seed=42).select(range(3000))
        L_test = L['validation'].shuffle(seed=42).select(range(1000))
    elif task=='swords':
        L_train, L_test = utils.load_swords_data(seed=0)
    elif task=='lambada':
        L_train, L_test = utils.load_lambada_data(seed=0)
    else:
        raise NotImplementedError("Task not implemented!")

    # Compute log-probabilities on the fly instead of loading from disk
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)

    print("Computing log-probabilities on the fly...")
    print(f"Using device: {device}")

    if task=='hypernym':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.taxonomic == "yes"]
        # Generate generator prompts
        p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style='generator', shots='zero', neg=False, both=None, is_chat=with_chat)
        prompts_gen = [i.prompt for i in p_train_gen]

        # Compute log-probabilities for generator prompts
        gen_logprobs_last_layer_pos = []
        for idx, prompt in enumerate(tqdm(prompts_gen)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (noun2)
            # For hypernymy, we want the probability of the noun2 token
            target_text = " " + L_train_all[idx].noun2
            target_tokens = tokenizer.encode(target_text)
            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            gen_logprobs_last_layer_pos.append(log_prob)

        # Generate discriminator prompts
        p_train, hf_train, _ = utils.make_and_format_data(make_prompt_hypernymy, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        prompts_pos = [i.prompt for i in p_train]

    elif task=='trivia-qa':
        L_train_all = L_train  # Already using all examples for trivia-qa
        # Generate generator prompts
        p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='generator', shots='zero', both=None)
        prompts_gen = [i.prompt for i in p_train_gen]
        # Compute log-probabilities for generator prompts
        gen_logprobs_last_layer_pos = []
        for idx, prompt in enumerate(tqdm(prompts_gen)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (answer)
            target_text = " " + L_train_all[idx]['answers'][0].capitalize()
            target_tokens = tokenizer.encode(target_text)
            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            gen_logprobs_last_layer_pos.append(log_prob)

        # Generate discriminator prompts
        p_train, hf_train, _ = utils.make_and_format_data(make_prompt_triviaqa, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        prompts_pos = [i.prompt for i in p_train]

    elif task=='swords':
        if use_all:
            L_train_all = L_train
        else:
            L_train_all = [i for i in L_train if i.synonym=='yes']
        # Generate generator prompts
        p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style='generator', shots='zero', neg=False, both=None)
        prompts_gen = [i.prompt for i in p_train_gen]

        # Compute log-probabilities for generator prompts
        gen_logprobs_last_layer_pos = []
        for idx, prompt in enumerate(tqdm(prompts_gen)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=False)
            # Get the log probability for the target token (replacement)
            target_text = " " + L_train_all[idx].replacement
            target_tokens = tokenizer.encode(target_text)
            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            gen_logprobs_last_layer_pos.append(log_prob)

        # Generate discriminator prompts
        p_train, hf_train, _ = utils.make_and_format_data(make_prompt_swords, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        prompts_pos = [i.prompt for i in p_train]

    elif task=='lambada':
        L_train_all = L_train  # Already using all examples for lambada
        # Generate generator prompts
        p_train_gen, hf_train_gen, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style='generator', shots='zero', both=None)
        prompts_gen = [i.prompt for i in p_train_gen]

        # Compute log-probabilities for generator prompts
        gen_logprobs_last_layer_pos = []
        for idx, prompt in enumerate(tqdm(prompts_gen)):
            probs = get_final_logit_prob(prompt, model, tokenizer, device, is_chat=with_chat)
            # Get the log probability for the target token (final_word)
            target_text = " " + L_train_all[idx]['final_word']
            target_tokens = tokenizer.encode(target_text)
            # Use the first token after the space
            ind = target_tokens[0] if len(target_tokens) == 1 else target_tokens[1]
            log_prob = math.log(probs[ind].item() + 1e-12)
            gen_logprobs_last_layer_pos.append(log_prob)

        # Generate discriminator prompts
        p_train, hf_train, _ = utils.make_and_format_data(make_prompt_lambada, L_train_all, tokenizer, style='discriminator', shots=disc_shots, neg=False, both=None)
        prompts_pos = [i.prompt for i in p_train]
    else:
        raise ValueError("!!")

    max_context_length = len(hf_train[0]['input_ids'])
    print("MAX CONTEXT LENGTH: ", max_context_length)


    #Z = list(zip(prompts_pos, gen_logprobs_last_layer_pos))
    Z = list(zip(p_train, gen_logprobs_last_layer_pos))
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
    pairs = [(Z[i[0]], Z[i[1]]) for i in pair_inds]

    #breakpoint()
    token_id = tokenizer.encode(" Yes")[-1]
    #first_token_id = tokenizer.encode(pairs[0][0].completion)[-1]
    #second_token_id = tokenizer.encode(pairs[1][0].completion)[-1]

    # Filter pairs that maintain minimum separation of delta
#    pairs = [((pair[0][0].prompt ,pair[1][0].prompt),  (tokenizer.encode(pair[0][0].completion)[-1], tokenizer.encode(pair[1][0].completion)[-1] )  ) for pair in pairs
#            if pair[1][1] - pair[0][1] > delta]

    pairs = [((pair[0][0].prompt ,pair[1][0].prompt),  (token_id, token_id) ) for pair in pairs
            if pair[1][1] - pair[0][1] > delta]

    #if not use_all:
    #    assert all([pair[0][0].completion== " Yes" for pair in pairs])
    #    assert all([pair[1][0].completion== " Yes" for pair in pairs])
    
    # Debug prints
    #print("\nDebugging pairs structure:")
    #print("First pair structure:", pairs[0])
    #print("First pair token types:", type(pairs[0][1][0]), type(pairs[0][1][1]))
    #print("First pair tokens:", pairs[0][1][0], pairs[0][1][1])
    
    #breakpoint()

    print(pairs[0])
    print("\n\n")
    print(pairs[1])
    print("\n\nNum Samples: ", len(pairs))

    class PairwiseDataset(Dataset):
        def __init__(self, pairs, tokenizer, max_length=128, device='cuda', is_chat=False):
            """
            pairs: list of ((prompt_i, prompt_j), (token_i, token_j))
            tokenizer: Hugging Face tokenizer
            device: device to place tensors on
            is_chat: whether to use chat template formatting
            """
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.device = device
            self.is_chat = is_chat
            
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
            if self.is_chat:
                # Use chat template for instruct models
                message_i = [
                    {"role": "system", "content": "Answer directly without explanation."},
                    {"role": "user", "content": prompt_i},
                ]
                input_ids_i = self.tokenizer.apply_chat_template(
                    message_i, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                )
                
                # Handle truncation if needed
                if input_ids_i.size(1) > self.max_length:
                    input_ids_i = input_ids_i[:, :self.max_length]
                    attention_mask_i = torch.ones_like(input_ids_i)
                # Handle padding if needed (left padding to match the tokenizer's padding_side)
                elif input_ids_i.size(1) < self.max_length:
                    pad_length = self.max_length - input_ids_i.size(1)
                    if self.tokenizer.padding_side == 'left':
                        input_ids_i = torch.cat([
                            torch.full((1, pad_length), self.tokenizer.pad_token_id, device=input_ids_i.device),
                            input_ids_i
                        ], dim=1)
                        attention_mask_i = torch.cat([
                            torch.zeros(1, pad_length, device=input_ids_i.device),
                            torch.ones(1, input_ids_i.size(1) - pad_length, device=input_ids_i.device)
                        ], dim=1)
                    else:
                        input_ids_i = torch.cat([
                            input_ids_i,
                            torch.full((1, pad_length), self.tokenizer.pad_token_id, device=input_ids_i.device)
                        ], dim=1)
                        attention_mask_i = torch.ones_like(input_ids_i)
                else:
                    attention_mask_i = torch.ones_like(input_ids_i)
                
                enc_i = {
                    "input_ids": input_ids_i,
                    "attention_mask": attention_mask_i
                }
                
                # Same for prompt j
                message_j = [
                    {"role": "system", "content": "Answer directly without explanation."},
                    {"role": "user", "content": prompt_j},
                ]
                input_ids_j = self.tokenizer.apply_chat_template(
                    message_j, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                )
                
                # Handle truncation if needed
                if input_ids_j.size(1) > self.max_length:
                    input_ids_j = input_ids_j[:, :self.max_length]
                    attention_mask_j = torch.ones_like(input_ids_j)
                # Handle padding if needed (left padding to match the tokenizer's padding_side)
                elif input_ids_j.size(1) < self.max_length:
                    pad_length_j = self.max_length - input_ids_j.size(1)
                    if self.tokenizer.padding_side == 'left':
                        input_ids_j = torch.cat([
                            torch.full((1, pad_length_j), self.tokenizer.pad_token_id, device=input_ids_j.device),
                            input_ids_j
                        ], dim=1)
                        attention_mask_j = torch.cat([
                            torch.zeros(1, pad_length_j, device=input_ids_j.device),
                            torch.ones(1, input_ids_j.size(1) - pad_length_j, device=input_ids_j.device)
                        ], dim=1)
                    else:
                        input_ids_j = torch.cat([
                            input_ids_j,
                            torch.full((1, pad_length_j), self.tokenizer.pad_token_id, device=input_ids_j.device)
                        ], dim=1)
                        attention_mask_j = torch.ones_like(input_ids_j)
                else:
                    attention_mask_j = torch.ones_like(input_ids_j)
                
                enc_j = {
                    "input_ids": input_ids_j,
                    "attention_mask": attention_mask_j
                }
            else:
                # Regular tokenization for non-instruct models
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
            batch_size = 4
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

    dataset = PairwiseDataset(pairs, tokenizer, max_length=max_context_length, device=device, is_chat=with_chat)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("\n\nDone making dataloader\n\n")
    optimizer = AdamW(model.parameters(), lr=lr)

    #num_epochs = 10
    #save_steps = 1
    losses = []


    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        #if True:#epoch % save_steps==1:
        if epoch!=0:
            #save_directory = "../models/v3-delta5-epoch"+str(epoch)
            #save_directory = "../models/v3-delta5-no-overlap-both-epoch"+str(epoch)
            with_ref_str = "-with-ref" if with_ref else ""
            all_str = "-all" if use_all else ""
            save_directory = "../models/v4-" + model_name.replace('/','--')  + "-delta"+str(delta)+"-epoch"+str(epoch) + "--" + task + with_ref_str + all_str
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

            # We'll take the last timestep for each prompt (seq_len - 1).
            # If you want something different (e.g. next token's logit), adjust accordingly.
    #        last_idx_i = attention_mask_i.sum(dim=1) - 1  # last non-pad index for each example
            last_idx_i = attention_mask_i.size(1) - 1
            # Alternatively, if you want the absolute last index in the tensor: last_idx_i = logits_i.size(1) - 1

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

            # TODO: with torch.no_grad(): compute ref logits diff

            # Forward pass for prompt i
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
    args = parser.parse_args()
    main(args)
