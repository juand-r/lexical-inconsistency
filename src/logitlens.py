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


def get_logitlens_output(prompt, model, modelname_short):
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


def get_logodds_disc(Ps, ii, yestoks, notoks):
    lgo = torch.log(torch.sum(Ps[ii][:, yestoks], dim=-1)) - torch.log(
        torch.sum(Ps[ii][:, notoks], dim=-1)
    )
    lgo[torch.isinf(lgo)] = 35  # truncate infs
    return lgo


def get_logodds_gen(Ps, L, ii, tokenizer, first_sw_token, task):
    #TODO should clean up this code so it takes in the completion
    if task=='hypernym':
        ind = tokenizer.encode("a " + L[ii].noun2)[first_sw_token]
    elif task=='trivia-qa':
        ind = tokenizer.encode("a " + L[ii]['answers'][0])[first_sw_token]
    else:
        raise ValueError("!")

    lgo = torch.log(torch.abs(Ps[ii][:, ind])) - torch.log(
        torch.abs(1 - Ps[ii][:, ind])
    )
    lgo[torch.isinf(lgo)] = 35  # truncate infs
    return lgo


def makepreds_disc(logodds, threshold=0, layer_disc=-1):
    pred = ["Yes" if x > threshold else "No" for x in [i[layer_disc].tolist() for i in logodds][:]]
    return pred


def makepreds_gen(ranks, threshold=40):
    return ["Yes" if r <= threshold else "No" for r in ranks]


def compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=-1, layer_disc=-1):

    if task=='hypernym':
        gold = [i.taxonomic.capitalize() for i in L]
    elif task=="trivia-qa":
        #TODO will need to do this differently if include negative examples
        gold = ['Yes' for i in L]
    else:
        raise ValueError("!")

    print("correlation: zs gen, fs disc (more usual)")
    corr = pearsonr(
        [i[layer_gen].tolist() for i in logodds_gen],
        [i[layer_disc].tolist() for i in logodds_disc]
    ).statistic
    print(corr)

    disc_accuracy = sklearn.metrics.accuracy_score(gold, makepreds_disc(logodds_disc, threshold=0, layer_disc=layer_disc))
    print("\naccuracy of discriminator fs: {}".format(disc_accuracy))

    gen_accuracies = {} #map threshold to accuracy
    for threshold in [5, 10, 40, 100, 1000]:
        a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=threshold))
        gen_accuracies[threshold]= a
        print("accuracy of generator zs: (th={}: {})".format(threshold, a))
    return disc_accuracy, gen_accuracies, corr


def compute_logodds(
    task, P_gen, P_disc, L, tokenizer, first_sw_token, yestoks, notoks, layer_gen=-1, layer_disc=-1
):

    if task=='hypernym':
        ranks = [
            get_rank(
                P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii].noun2)[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task=='trivia-qa':
        ranks = [
            get_rank(
                P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii]['answers'][0])[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    else:
        raise ValueError("!!")

    logodds_gen = [get_logodds_gen(P_gen, L, ii, tokenizer, first_sw_token, task) for ii in range(len(P_gen))]
    logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]

    disc_accuracy, gen_accuracies, corr = compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=layer_gen, layer_disc=layer_disc)

    return ranks, logodds_gen, logodds_disc, corr


