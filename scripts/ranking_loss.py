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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, "src")
sys.path.append(src_path)
import utils

#TODO load model also, use function above
model_name = "google/gemma-2-2b"
#tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_model_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

tokenizer, model = load_model_tokenizer(model_name)

# see if padding right works??
tokenizer.padding_side = 'left'

#TODO assume generator is ground truth
#TODO assume discriminator is ground truth

# assume label of 1 meaning left < right in ground truth


# NOTE first use ground truth ranking from generator.
# Will now use ranking loss on *discriminator* prompts to try to match it!
L = utils.load_noun_pair_data()
L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
#L_train, L_test = utils.split_train_test_no_overlap(L, seed=0)
#L_train, L_test = utils.split_train_test_no_overlap_both(L)


#TODO load the positive *train* set, not test set! Use generator order for ranking
gen_logodds = torch.load('../outputs/logodds/gemma-2-2b--gen-zero--train.pt', weights_only=True)
#gen_logodds = torch.load('../outputs/logodds/gemma-2-2b--gen-zero--train--hyper.pt', weights_only=True)
#gen_logodds = torch.load('../outputs/logodds/gemma-2-2b--gen-zero--train--both.pt', weights_only=True)


gen_logprobs_last_layer = [-math.log(1+math.exp(-l[-1].tolist())) for l in gen_logodds[:]]
gen_logprobs_last_layer_pos = [i for ii,i in enumerate(gen_logprobs_last_layer) if L_train[ii].taxonomic=='yes']

#NOTE train only on positive examples for now
L_train_pos = [i for i in L_train if i.taxonomic == "yes"]
#TODO do this again on "few" shot setting which is the one I actually want.. but need memory
p_train, hf_train = utils.make_and_format_data(L_train_pos, tokenizer, style='discriminator', shots='few', neg=False, both=None)
prompts_pos = [i.prompt for i in p_train]
#NOTE in discriminator case the indices we're looking for is just index of " Yes"... for generator it will depend on each prompt
# of course.

Z = list(zip(prompts_pos, gen_logprobs_last_layer_pos))
Z = sorted(Z, key = lambda i: i[-1])
k = 5
# sample k points uniformly after each j, for j from 0 to len(Z)-k
#NOTE since we sorted them previously, they are in the right order, first one < second one


delta = 5 #maybe 10

#NOTE this is too sparse at the start! Don't do this
#pairs = [(Z[j],  Z[j+1:][i*( len(Z)-(j+1) -1)//(k-1)]  ) for j in range(len(Z)-k-1) for i in range(k)]
#total_samples = 20000 #used 20k for when delta=10 6000
total_samples = 5110
#Uniform random maybe better
indices = range(len(Z))
pair_inds = list(itertools.product(indices, repeat=2))
pair_inds = [i for i in pair_inds if i[0] < i[1]]
pair_inds = random.sample(pair_inds, total_samples)
pairs = [ (Z[i[0]] , Z[i[1]]) for i in pair_inds]
#TODO confirm mass is mostly on Yes and No
token_id = tokenizer.encode(" Yes")[-1]
#NOTE can ignore the actual values now! The order is what matters.

#pairs = [((pair[0][0],pair[1][0]), (token_id, token_id)) for pair in pairs]
pairs = [((pair[0][0],pair[1][0]), (token_id, token_id)) for pair in pairs if abs(pair[0][1] - pair[1][1]) > delta]
# train pairs -- need to get index of each in vocab
#pairs = [
#    (("Hello world", "Hello Mars"), (13, 1312)),
#    (("What's your name?", "Where do you live?"), (412, 5840)),
#    (("Hello world", "Hello Mars"), (13, 1312)),
#    (("Hello world", "Hello Mars"), (13, 1312)),
#]


print(pairs[0])
print("\n\n")
print(pairs[1])
print("\n\nNum Samples: ", len(pairs))

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=128):
        """
        pairs: list of ((prompt_i, prompt_j), (token_i, token_j))
        tokenizer: Hugging Face tokenizer
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (prompt_i, prompt_j), (token_i, token_j) = self.pairs[idx]
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
            # For simplicity, label=1 means "i < j"
            # If you have a different label scheme, store that accordingly.
            'label': torch.tensor(1.0, dtype=torch.float)
        }
        return item

#18 fine for zero-shot
dataset = PairwiseDataset(pairs, tokenizer, max_length=64)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("\n\nDone making dataloader\n\n")
optimizer = AdamW(model.parameters(), lr=1e-5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
save_steps = 1
losses = []


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    if True:#epoch % save_steps==1:
        #save_directory = "../models/v3-delta5-epoch"+str(epoch)
        #save_directory = "../models/v3-delta5-no-overlap-both-epoch"+str(epoch)
        save_directory = "../models/v3-delta5"+str(epoch)
        print("Saving to ", save_directory)
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        # Move inputs to device
        input_ids_i = batch["input_ids_i"].to(device)
        attention_mask_i = batch["attention_mask_i"].to(device)
        token_id_i = batch["token_id_i"].to(device)

        input_ids_j = batch["input_ids_j"].to(device)
        attention_mask_j = batch["attention_mask_j"].to(device)
        token_id_j = batch["token_id_j"].to(device)

        label = batch["label"].to(device)  # typically 1.0 for "i < j"

        # Forward pass for prompt i
        outputs_i = model(input_ids=input_ids_i, attention_mask=attention_mask_i)
        # logits_i: [batch_size, seq_len, vocab_size]
        logits_i = outputs_i.logits

        # We'll take the last timestep for each prompt (seq_len - 1).
        # If you want something different (e.g. next token’s logit), adjust accordingly.
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
        score_i = log_probs_i[torch.arange(log_probs_i.size(0)), token_id_i]

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
        score_j = log_probs_j[torch.arange(log_probs_j.size(0)), token_id_j]

        # Pairwise logistic loss: - log( sigmoid( (score_j) - (score_i) ) )
        diff = score_j - score_i
        loss = -torch.log(torch.sigmoid(diff) + 1e-12).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
