"""
Utilities to:

- Load the hypernym dataset:
    L = load_noun_pair_data()

- Make a train/test split:
    L_train, L_test = split_train_test(L, seed=0, subsample=False, num_train=3000)

- Make a prompt to evaluate:
    # item is an element of L
    prompt = make_prompt(item, style="generator", shots="zero", neg=False)

- Make prompts and format them into a tokenized, padded, huggingface Dataset
    prompts_train, hf_train = make_and_format_data(L_train, tokenizer, style="discriminator", shots="few", neg=False, both="union")

- Load huggingface or peft model:
    model, tokenizer = load_model(peft_model_id, device)

Note: some of this code adapted from
https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy

"""

import math
import random
from collections import namedtuple
from string import Template
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def make_prompt(item, style="generator", shots="zero", neg=False):
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


def make_and_format_data(
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

    if both == "union":
        items1 = [
            make_prompt(i, style="discriminator", shots="zero", neg=False) for i in L
        ]
        items2 = [make_prompt(i, style="generator", shots="zero", neg=True) for i in L]
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

