import json
from tokenizer import AutoTokenizer

import utils
from utils import make_prompt_hypernymy, make_prompt_swords, make_prompt_triviaqa

model_name = "google/gemma-2-2b"
task == 'hypernym'

tokenizer = AutoTokenizer.from_pretrained(model_name)

if task == 'hypernym':
    prompt_func = make_prompt_hypernymy
    L = utils.load_noun_pair_data()
    L_train, L_test = utils.split_train_test(L, seed=0, subsample=False, num_train=3000)
    #TODO calculate the log-probs directly
    disc = torch.load('../outputs/logodds/gemma-2-2b--hypernym--disc-few--train.pt', weights_only=True)
    gen = torch.load('../outputs/logodds/gemma-2-2b--hypernym--gen-zero--train.pt', weights_only=True)
    gen_logprobs_last_layer = [-math.log(1+math.exp(-l[-1].tolist())) for l in gen[:]]
    disc_logprobs_last_layer = [-math.log(1+math.exp(-l[-1].tolist())) for l in disc[:]]

if task == 'swords':
    prompt_func = make_prompt_swords
    L_train, L_test = utils.load_swords_data(seed=0)
    #TODO
if task == 'triviaqa':
    prompt_func = make_prompt_triviaqa
    #TODO

prompts_train_d, hf_train_d = utils.make_and_format_data(prompt_func, L_train, tokenizer, style="discriminator", shots="few", neg=False, both=None)
prompts_train_g, hf_train_g = utils.make_and_format_data(prompt_func, L_train, tokenizer, style="generator", shots="zero", neg=False, both=None)

#TODO maybe generalize this to work with attributes of L_train available from other tasks e.g., like swords
D = [{"noun1":L_train[i].noun1,
      "noun2":L_train[i].noun2,
      "taxonomic": L_train[i].taxonomic,
      "generator-prompt": prompts_train_g[i].prompt,
      "generator-completion": prompts_train_g[i].completion,
      "discriminator-prompt": prompts_train_d[i].prompt,
      "discriminator-gold-completion": prompts_train_d[i].completion,
      "generator-log-prob":gen_logprobs_last_layer[i],
      "discriminator-log-prob":disc_logprobs_last_layer[i]} for i in range(len(L_train))]

#TODO save to json

