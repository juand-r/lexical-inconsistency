"""
Implementation of logit lens using nnsight.
For more information, see https://nnsight.net/notebooks/tutorials/logit_lens/

"""

from nnsight import LanguageModel
import torch
import seaborn as sns
import numpy as np
import sklearn.metrics
from scipy.stats import pearsonr
from tqdm import tqdm

device = "cuda:0"


def load_model_nnsight(modelname, device):
    """Load the model into an nnsight.LanguageModel object

    Usage:
    model = load_model_nnsight("google/gemma-2b", "cuda:0")
    """
    model = LanguageModel(modelname, device_map=device, dispatch=True)
    return model


def logitlens(prompt, model, modelname_short):
    """
    Usage:
    model = load_model_nnsight("google/gemma-2b", "cuda:0")
    prompt = "The Eiffel Tower is in the city of"
    probs, max_probs, tokens, words, input_words = logitlens(prompt, model)
    """
    if modelname_short in ["gpt2-xl"]:
        layers = model.transformer.h
    if modelname_short in ["gemma-2-2b", "Meta-Llama-3-8B-Instruct"]:
        layers = model.model.layers

    probs_layers = []

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                if modelname_short in ["gpt2-xl"]:
                    layer_output = model.lm_head(
                        model.transformer.ln_f(layer.output[0])
                    )
                elif modelname_short in [
                    "gemma-2-2b",
                    "Meta-Llama-3-8B-Instruct",
                ]:
                    layer_output = model.lm_head(model.model.norm(layer.output[0]))
                else:
                    raise NotImplementedError("Model not implemented.")

                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    probs = torch.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)

    # Decode token IDs to words for each layer
    words = [
        [
            model.tokenizer.decode(t).encode("unicode_escape").decode()
            for t in layer_tokens
        ]
        for layer_tokens in tokens
    ]

    # Access the 'input_ids' attribute of the invoker object to get the input words
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0]["input_ids"][0]]
    return probs, max_probs, tokens, words, input_words


def get_rank(L, ind):
    value = L[ind]
    sorted_L, sorted_indices = torch.sort(L, descending=True)
    rank = (sorted_L == value).nonzero(as_tuple=True)[0][0].item() + 1
    return rank


def logodds_yn(Ps, ii, yestoks, notoks):
    what = torch.log(torch.sum(Ps[ii][:, yestoks], dim=-1)) - torch.log(
        torch.sum(Ps[ii][:, notoks], dim=-1)
    )
    what[torch.isinf(what)] = 35  # truncate infs
    return what


def logodds(Ps, L_, ii, tokenizer, first_sw_token):
    ind = tokenizer.encode("a " + L_[ii].noun2)[first_sw_token]
    what = torch.log(torch.abs(Ps[ii][:, ind])) - torch.log(
        torch.abs(1 - Ps[ii][:, ind])
    )
    what[torch.isinf(what)] = 35  # truncate infs
    return what


def makepreds_disc(lgo, threshold=0):
    pred = ["Yes" if x > threshold else "No" for x in [i[-1].tolist() for i in lgo][:]]
    return pred


def makepreds_gen(ranks, threshold=40):
    return ["Yes" if r <= threshold else "No" for r in ranks]


def compute_logodds(
    Psgen_, Psgenfs_, Ps_, Psfs_, L_, tokenizer, first_sw_token, yestoks, notoks
):

    gold = [i.taxonomic.capitalize() for i in L_[:]]
    ranks = [
        get_rank(
            Psgen_[ii][-1, :], tokenizer.encode("a " + L_[ii].noun2)[first_sw_token]
        )
        for ii in tqdm(range(len(Psgen_)))
    ]
    lr = -np.log(ranks[:])

    # log-odds, generator:
    lgo = [
        logodds(Psgen_, L_, ii, tokenizer, first_sw_token) for ii in range(len(Psgen_))
    ]
    lgo_yfs = [logodds_yn(Psfs_, ii, yestoks, notoks) for ii in range(len(Psfs_))]

    print("correlation: zs gen, fs disc (more usual)")
    p = pearsonr(
        [i[-1].tolist() for i in lgo], [i[-1].tolist() for i in lgo_yfs]
    ).statistic
    print(p)

    print("\n\n")

    print("\naccuracy of discriminator fs: ")
    a = sklearn.metrics.accuracy_score(gold, makepreds_disc(lgo_yfs))
    print(a)

    print("\naccuracy of generator zs: (th=5)")
    a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=5))
    print(a)

    print("\naccuracy of generator zs: (th=10)")
    a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=10))
    print(a)

    print("\naccuracy of generator zs: (th=40)")
    a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=40))
    print(a)

    print("\naccuracy of generator zs: (th=100)")
    a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=100))
    print(a)

    print("\naccuracy of generator zs: (th=1000)")
    a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=1000))
    print(a)

    return ranks, lgo, lgo_yfs
