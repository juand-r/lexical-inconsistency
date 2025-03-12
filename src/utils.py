"""
Utilities to:

- Load the hypernym dataset:
    L = load_noun_pair_data()

- Make a train/test split:
    L_train, L_test = split_train_test(L, seed=0, subsample=False, num_train=3000)

- Make a prompt to evaluate:
    # item is an element of L
    prompt = make_prompt_hypernymy(item, style="generator", shots="zero", neg=False)

- Make prompts and format them into a tokenized, padded, huggingface Dataset
    prompts_train, hf_train = make_and_format_data(make_prompt, L_train, tokenizer, style="discriminator", shots="few", neg=False, both="union")

- Load huggingface or peft model:
    model, tokenizer = load_model(peft_model_id, device)

Note: some of this code adapted from
https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy

"""

import re
import math
import json
import random
from collections import namedtuple
from string import Template
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from find_disjoint_sets import partition_items_kernighan_lin


def load_noun_pair_data():
    """
    Load the noun pair (hypernymy) data, with information including whether
    taxonomic relation exists ("taxonomic"), and high or low similarity ("sim")
    """
    with open("../data/ranks.txt", "r") as fd:
        x = fd.readlines()

    x = [i.strip().split("\t") for i in x]
    for i in x:
        i[4] = int(i[4])
    Item = namedtuple(
        "Item",
        ["noun1", "noun2", "taxonomic", "sim", "gen_rank", "yesgreater", "argmax"],
    )
    out = [Item(*i) for i in x]
    return out


def load_swords_data(seed=0):
    """Load the swords dataset, and makes positive and negative pairs from it."""
    #NOTE for now pick the top highest ranked item in list of substitutes as positive case
    #TODO generalize later!

    with open("../data/swords-data-v1.1_test.json", "r") as fd:
        test = json.load(fd)

    with open("../data/swords-data-v1.1_dev.json", "r") as fd:
        dev = json.load(fd)

    Item = namedtuple(
        "Item",
        ["context", "target", "replacement", "synonym"],
    )

    #NOTE use their dev as my train to keep things sane.
    testset = []
    for item in test:
        #TODO make this change permanent in the file
        item['context'] = re.sub(r'\s+', ' ', item['context'])

        pos_replacement = item['substitutes'][0][0]
        neg_replacement = item['substitutes'][-1][0]

        context = item['context'].replace(item['target'], "*" + item['target'] + "*")

        testset.append(Item(context, item['target'], pos_replacement, 'yes') )
        testset.append(Item(context, item['target'], neg_replacement, 'no') )

    trainset = []
    for item in dev:
        item['context'] = re.sub(r'\s+', ' ', item['context'])

        pos_replacement = item['substitutes'][0][0]
        neg_replacement = item['substitutes'][-1][0]

        context = item['context'].replace(item['target'], "*" + item['target'] + "*")

        trainset.append(Item(context, item['target'], pos_replacement, 'yes') )
        trainset.append(Item(context, item['target'], neg_replacement, 'no') )

    random.seed(seed)
    random.shuffle(trainset)
    random.shuffle(testset)
    return trainset, testset

#TODO generalize better to DRY!

def make_prompt_triviaqa(item, with_context=False, style='generator', shots='zero'):
    """ Only positive examples for now! Also, when multiple acceptable answers are available in the dataset,
    use the first one for now. """

    example_context = """[DOC] [TLE] brindisi | Italian music | Britannica.combrindisi | Italian music | Britannica.com [PAR] Italian music [PAR] THIS IS A DIRECTORY PAGE. Britannica does not currently have an article on this topic. [PAR] Learn about this topic in these articles: [PAR]   [PAR] in drinking song [PAR] ...in certain types of 19th-century opera and operetta, frequently involving not only a soloist but also a chorus joining in with choral repeats or refrains. In Italy the drinking song is known as brindisi (Italian: "toast"). In Giuseppe Verdi's operas drinking songs range from the cheerful "Libiamo" ("Let Us Drink") in La traviata (1853), to.."""

    example_question = "What kind of song is a Brindisi?"
    example_answer = "drinking song"

    if style=='generator':
        if with_context:
            example = "Context: " + example_context + "\n\nQuestion: " + example_question + "\n\nAnswer: " + example_answer + "\n\n"
        else:
            example = "Question: " + example_question + "\n\nAnswer: " + example_answer + "\n\n"

        query = "Question: " + item['question'] + "\n\nAnswer: "# + item['answer'] + "\n\n"
        completion = item['answers'][0]
        if with_context:
            query = "Context: "+ item['context'] + "\n\n" + query

        if shots =='zero':
            prompt = query
        else:
            prompt = example + query
    else: #discriminator
        completion = "Yes"
        example = "Is the correct answer to the question \"" + example_question + "\" given by \""+ example_answer + "\"? Answer Yes or No: " + completion + "\n\n"
        if with_context:
            example = "Context: " + example_context + "\n\n" + example + "\n\n"

        query = "Is the correct answer to the question \"" + item['question'] + "\" given by \""+ item['answers'][0] + "\"? Answer Yes or No: "
        if with_context:
            query = "Context: "+ item['context']  + "\n\n" + query

        if shots == 'zero':
            prompt = query
        else:
            prompt = example + query
    
    prompt = prompt.strip()
    completion = " " + completion.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)


def make_prompt_swords(item, style="generator", shots="zero", neg=False):
    """
    Make a prompt based on the item.
    """

    if neg and item.synonym=='no':
        negation_word = " not"
    else:
        negation_word = ""

    if style == "generator":
        generator_prompt = 'Notice the word "$target" used in the context: "$context". In this context, the word "$target" is$optional_negation synonymous with "'
        prompt = Template(generator_prompt).substitute(context=item.context, target=item.target, optional_negation=negation_word)
        #completion = " " + item.replacement
        completion = "" + item.replacement
        #TODO should few-shot negation case have different example..??
        if shots == "few":
            #examples = "In the following sentence: 'I thought as much. Now leave, before I call the rats on you.', the word call is a synonym of the word summon\n\n"
            #TODO make "neg" and regular version of this example??
            examples = 'Notice the word "artists" used in the context: "Many painters, sculptors, and other *artists* were inspired by Duchamp.". In this context, the word "artists" is not synonymous with "character".\n\nNotice the word "happen" used in the context: "I could free Tasha. If I did, one of three things would *happen*. Most likely: she would be meat..." In this context, the word "happen" is synonymous with "transpire".\n\n'
            prompt = examples + prompt

    elif style == "discriminator":
        instruction =  'Determine whether the word in context can be replaced by another word or expression without changing the meaning of the sentence.\n\n'
        examples = 'Notice the word "artists" used in the context: "Many painters, sculptors, and other *artists* were inspired by Duchamp.". In this context, is "artists" synonymous with "character"? Answer: No\n\nNotice the word "happen" used in the context: "I could free Tasha. If I did, one of three things would *happen*. Most likely: she would be meat..." In this context, is "happen" synonymous with "transpire"? Answer: Yes\n\n'
        template_string = 'Notice the word "$target" used in the context: "$context". In this context, is "$target" synonymous with "$replacement"? Answer:'

        if shots == 'zero':
            prompt = Template(
                instruction + template_string
                ).substitute(context=item.context, target=item.target, replacement=item.replacement)
            completion = " " + item.synonym.capitalize()

        #TODO more than two shots??
        if shots == "few":
            #example1 = "In the following sentence: 'Well, kid, what do you think? Remember, this is your quest.', can the word think be replaced by the word mind? Answer: No\n\n"
            #example2 = "In the following sentence: 'I thought as much. Now leave, before I call the rats on you.', can the word call be replaced by the word summon? Answer: Yes\n\n"
            prompt = Template(
                instruction + examples + template_string
                ).substitute(context=item.context, target=item.target, replacement=item.replacement)
            completion = " " + item.synonym.capitalize()
    else:
        raise ValueError("!?")

    prompt = prompt.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)


def make_prompt_hypernymy(item, style="generator", shots="zero", neg=False):
    """
    Make a prompt based on the item.
    """
    if style == "generator":
        if shots == "zero":
            if neg:
                if item.taxonomic == "no":
                    prompt = Template(
                        "Complete the sentence: $word are not a kind of"
                    ).substitute(word=item.noun1, hypernym=item.noun2)
                else:
                    prompt = Template(
                        "Complete the sentence: $word are a kind of"
                    ).substitute(word=item.noun1, hypernym=item.noun2)
            else:
                prompt = Template(
                    "Complete the sentence: $word are a kind of"
                ).substitute(word=item.noun1, hypernym=item.noun2)
            completion = " " + item.noun2
        else:
            if neg:
                if item.taxonomic == "no":
                    prompt = Template(
                        "Complete the sentence: bees are not a kind of furniture\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are not a kind of fruit\n\nComplete the sentence: $word are not a kind of"
                    ).substitute(word=item.noun1)
                else:
                    prompt = Template(
                        "Complete the sentence: bees are a kind of insect\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are a kind of bird\n\nComplete the sentence: $word are a kind of"
                    ).substitute(word=item.noun1)
            else:
                prompt = Template(
                    "Complete the sentence: bees are a kind of insect\n\nComplete the sentence: corgis are a kind of dog\n\nComplete the sentence: robins are a kind of bird\n\nComplete the sentence: $word are a kind of"
                ).substitute(word=item.noun1)
            completion = " " + item.noun2
    elif style == "discriminator":
        if shots == "zero":
            prompt = Template("Do you think $word are a $hypernym? Answer:").substitute(
                word=item.noun1, hypernym=item.noun2
            )
            completion = " " + item.taxonomic.capitalize()
        else:
            prompt = Template(
                "Do you think bees are furniture? Answer: No\n\nDo you think corgis are dogs? Answer: Yes\n\nDo you think trucks are a fruit? Answer: No\n\nDo you think robins are birds? Answer: Yes\n\nDo you think $word are a $hypernym? Answer:"
            ).substitute(word=item.noun1, hypernym=item.noun2)
            completion = " " + item.taxonomic.capitalize()
    else:
        raise ValueError("!?")

    prompt = prompt.strip()
    Pt = namedtuple("PromptCompletion", ["prompt", "completion"])
    return Pt(prompt, completion)


def split_train_test(L, seed=0, subsample=False, num_train=3000):
    """
    After loading the data into a list L, use this to shuffle, subsample (optional),
    and split into train and test sets.
    """
    random.seed(seed)
    random.shuffle(L)

    if subsample:
        L = L[9::10]
        if num_train > len(L):
            num_train = math.floor(len(L) * 3 / 4)
            print("Reset num_train to ", num_train)

    L_train = L[:num_train]
    L_test = L[num_train:]
    return L_train, L_test


def split_train_test_no_overlap(L, seed=0):
    #TODO later flag for option no overlap in hyper, in hypo, or both
    test_concepts_hyper = ['jewelry',
                           'home decor',
                           'vehicle',
                           'musical instrument',
                           'tool',
                           'container',
                           'auto part',
                           'kitchen equipment',
                           'kitchen tool',
                           'garden tool']
    random.seed(seed)
    random.shuffle(L)
    L_train = [i for i in L if i.noun2 not in test_concepts_hyper]
    L_test = [i for i in L if i.noun2 in test_concepts_hyper]
    return L_train, L_test

def f1(x): return [i.noun1 for i in x]
def f2(x): return [i.noun2 for i in x]

def split_train_test_no_overlap_both(L, seed=2):
    random.seed(seed)
    random.shuffle(L)
    c1, c2, exc = partition_items_kernighan_lin(L)
    assert len(c1) == 419
    assert len(c2) == 2676 # train
    assert len(set(f1(c1)).intersection(f1(c2)))==0
    assert len(set(f2(c1)).intersection(f2(c2)))==0
    L_train = c2
    L_test = c1
    return L_train, L_test


def make_and_format_data(
    make_prompt,
    L,
    tokenizer,
    style="discriminator",
    shots="few",
    neg=False,
    both="union",
    instruction_masking=True,
):
    """
    Make prompts and completions, and tokenize and pad into a HF dataset.

    Note on `both`:
    - combines generator and discriminator, this ignores other parameters
    `style`, `shots` and `neg`
    """

    items = [make_prompt(i, style=style, shots=shots, neg=neg) for i in L]
    #items = [make_prompt(i, style=style, shots=shots) for i in L]

    if both == "union":
        items1 = [
            #make_prompt(i, style="discriminator", shots="zero", neg=False) for i in L
            make_prompt(i, style="discriminator", shots="zero") for i in L
        ]
        items2 = [make_prompt(i, style="generator", shots="zero") for i in L]
        #items2 = [make_prompt(i, style="generator", shots="zero", neg=True) for i in L]
        items = items1 + items2
        random.shuffle(items)

    if both == "joint":
        items1 = [
            make_prompt(i, style="discriminator", shots="zero", neg=False) for i in L
        ]
        items2 = [make_prompt(i, style="generator", shots="zero", neg=True) for i in L]

        discfirst = [
            namedtuple("PromptCompletion", ["prompt", "completion"])(
                items1[ii].prompt + items1[ii].completion + "\n\n" + items2[ii].prompt,
                items2[ii].completion,
            )
            for ii in range(len(items1))
        ]

        genfirst = [
            namedtuple("PromptCompletion", ["prompt", "completion"])(
                items2[ii].prompt + items2[ii].completion + "\n\n" + items1[ii].prompt,
                items1[ii].completion,
            )
            for ii in range(len(items1))
        ]

        items = discfirst + genfirst
        random.shuffle(items)

    print("EXAMPLE items: ")
    print("prefix: ", items[0].prompt)
    print("completion: ", items[0].completion)

    instructions = []
    completions = []
    for ii in range(len(items)):
        instruction = tokenizer(items[ii].prompt)["input_ids"]
        output = tokenizer(items[ii].completion, add_special_tokens=False)["input_ids"]
        instructions.append(instruction)
        completions.append(output)

    inputs = [instructions[ii] + completions[ii] for ii in range(len(completions))]

    # NOTE: The huggingface Training takes care of shifting tokens by 1.
    if instruction_masking:
        outputs = [
            [-100] * len(instructions[ii]) + completions[ii]
            for ii in range(len(completions))
        ]
    else:
        outputs = inputs

    padded_inputs = tokenizer.pad(
        {"input_ids": inputs}, padding=True, return_tensors="pt"
    )
    padded_outputs = tokenizer.pad(
        {"input_ids": outputs}, padding=True, return_tensors="pt"
    )

    hf_dataset = Dataset.from_dict(
        {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_outputs["input_ids"],
        }
    )

    return items, hf_dataset


def load_model(peft_model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        peft_model_id, torch_dtype=torch.float16
    )
    model.to(device)
    return model, tokenizer


def get_final_logit_prob(prompt, model, tokenizer, device = 'cuda', is_chat=False):
    with torch.no_grad():
        if is_chat:
            message = [
                {"role": "system", "content": "Answer directly without explanation."},
                {"role": "user", "content": prompt},]
            input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True,return_tensors="pt", tokenize=True, return_dict=False)[0].tolist()
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
        outputs = model(torch.tensor([input_ids]).to(device), output_hidden_states=True)
    # print("outputs.logits.shape",outputs.logits.shape)
    model_log_probs = (
            outputs.logits[..., :]
            .log_softmax(-1)
            .squeeze()
            .detach()
            .cpu()
            .float()
        )
    # print("model_log_probs.shape",model_log_probs.shape)
    # print(model_log_probs.shape) # (seq_len, vocab_size)
    # print(torch.exp(model_log_probs).shape, type(torch.exp(model_log_probs))) # (seq_len, vocab_size)
    model_log_probs = model_log_probs[-1, :]
    # print(model_log_probs.shape) # (seq_len, vocab_size)
    # raise ValueError("!?")
    # get the maximum indixe of model_log_probs
    max_ind = torch.argmax(model_log_probs)
    print("max_ind:", max_ind, torch.exp(model_log_probs[max_ind]))
    print(f"max token--{tokenizer.decode([max_ind])}--")
    print(torch.sum(torch.exp(model_log_probs)))
    return torch.exp(model_log_probs)
        
