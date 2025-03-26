"""
Implementation of logit lens using nnsight.
For more information, see https://nnsight.net/notebooks/tutorials/logit_lens/

"""

from nnsight import LanguageModel
import torch
import seaborn as sns
import numpy as np
import sklearn.metrics
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
    if modelname_short in ["gemma-2-2b", "Meta-Llama-3-8B-Instruct"] or 'llama' in modelname_short.lower():
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
                    "Llama-3.2-3B-Instruct"
                ]:
                    layer_output = model.lm_head(model.model.norm(layer.output[0]))
                else:
                    raise NotImplementedError("Model not implemented.")

                probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                probs_layers.append(probs)

    probs = torch.cat([probs.value for probs in probs_layers])

    # Find the maximum probability and corresponding tokens for each position
    max_probs, tokens = probs.max(dim=-1)
    #print(f"logitlens_output--max_ind:{tokens[-1, -1]}, {model.tokenizer.decode([tokens[-1, -1]])}, {max_probs[-1,-1]}")
    # Decode token IDs to words for each layer
    words = [
        [
            model.tokenizer.decode(t).encode("unicode_escape").decode()
            for t in layer_tokens
        ]
        for layer_tokens in tokens
    ]

    # Access the 'input_ids' attribute of the invoker object to get the input words
    # input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0]["input_ids"][0]]
    input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
    # print(len(input_words))
    # print(len(words), len(words[0]))
    # print(f"logitlens_output: {probs.shape}, {max_probs.shape}, {tokens.shape}, {words}, {input_words}")
    # torch.Size([26, 10, 256000]), torch.Size([26, 10]), torch.Size([26, 10])
    return probs, max_probs, tokens, words, input_words


def get_rank(L, ind):
    value = L[ind]
    sorted_L, sorted_indices = torch.sort(L, descending=True)
    rank = (sorted_L == value).nonzero(as_tuple=True)[0][0].item() + 1
    return rank


def get_logodds_disc(Ps, ii, yestoks, notoks):
    if notoks is None:
        lgo = torch.log(torch.sum(Ps[ii][..., yestoks], dim=-1)) 
    else:
        lgo = torch.log(torch.sum(Ps[ii][..., yestoks], dim=-1)) - torch.log(
            torch.sum(Ps[ii][..., notoks], dim=-1)
        )
    lgo[torch.isinf(lgo)] = 35  # truncate infs
    return lgo


def get_logodds_gen(Ps, L, ii, tokenizer, first_sw_token, task, is_chat = False, use_lgo=True):
    #TODO should clean up this code so it takes in the completion
    if is_chat:
        prefix = ""
    else:
        prefix = "a "
    if task=='hypernym':
        ind = tokenizer.encode(prefix + L[ii].noun2)[first_sw_token]
    elif task=='trivia-qa':
        ind = tokenizer.encode(prefix + L[ii]['answers'][0].capitalize())[first_sw_token]
    elif task=='swords':
        ind = tokenizer.encode(prefix + L[ii].replacement)[first_sw_token]
    elif task=='lambada':
        ind = tokenizer.encode(prefix + L[ii]['final_word'])[first_sw_token]
    else:
        raise ValueError("!")
    if use_lgo:
        lgo = torch.log(torch.abs(Ps[ii][..., ind])) - torch.log(
            torch.abs(1 - Ps[ii][..., ind])
        )
    else:
        lgo = torch.log(torch.abs(Ps[ii][..., ind])) 
    lgo[torch.isinf(lgo)] = 35  # truncate infs
    return lgo


def makepreds_disc(logodds, threshold=0, layer_disc=-1):
    if logodds[0].dim() == 0:
        pred = ["Yes" if x > threshold else "No" for x in [i.tolist() for i in logodds][:]]
    else:
        pred = ["Yes" if x > threshold else "No" for x in [i[layer_disc].tolist() for i in logodds][:]]
    return pred


def makepreds_gen(ranks, threshold=40):
    return ["Yes" if r <= threshold else "No" for r in ranks]


def compute_disc_accuracy(gold, logodds_disc):
    preds = [1 if i>0 else 0 for i in logodds_disc]
    disc_accuracy = sklearn.metrics.accuracy_score(gold, preds)
    # print("\naccuracy of discriminator fs: {}".format(disc_accuracy))

    # fpr, tpr, thresholds = roc_curve(gold, preds)
    if len(set(gold)) == 1:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(gold, logodds_disc)
    # print(f'roc_auc: {roc_auc}')
    return disc_accuracy, roc_auc

def compute_gen_accuracy(golds, ranks, thresholds = [5, 10, 40, 100, 1000]):
    gen_accuracies = {} #map threshold to accuracy
    for t in thresholds:
        preds = [1 if r <= t else 0 for r in ranks]
        a = sklearn.metrics.accuracy_score(golds, preds)
        gen_accuracies[t]= a
        print("accuracy of generator zs: (th={}: {})".format(t, a))
    return gen_accuracies

def compute_gen_mrr(golds, ranks):
    ranks_pos = [r for i, r in enumerate(ranks) if golds[i] == 1]
    ranks_neg = [r for i, r in enumerate(ranks) if golds[i] == 0]
    mrr_pos = np.mean([1 / r for r in ranks_pos])
    mrr_neg = np.mean([1 / r for r in ranks_neg])
    return mrr_pos, mrr_neg

def compute_metrics(task, L, logodds_gen, logodds_disc, ranks):
    if task=='hypernym':
        golds = [1 if i.taxonomic.strip().capitalize() == 'Yes' else 0 for i in L]
    elif task=="trivia-qa":
        if 'correct' in L[0]:
            golds = [1 if i['correct'] == 'Yes' else 0 for i in L]
        else:
            golds = [1  for i in L]
    elif task=='swords':
        golds = [1 if i.synonym.capitalize() == 'Yes' else 0 for i in L]
    elif task=='lambada':
        if 'correct' in L[0]:
            golds = [1 if i['correct'] == 'Yes' else 0 for i in L]
        else:
            golds = [1  for i in L]
    else:
        raise ValueError("!")

    print("correlation: zs gen, fs disc (more usual)")
    corr_all = pearsonr(logodds_gen, logodds_disc).statistic
    spear_all = spearmanr(logodds_gen, logodds_disc).statistic
    logodds_gen_pos = [logodds_gen[i] for i in range(len(logodds_gen)) if golds[i] == 1]
    logodds_gen_neg = [logodds_gen[i] for i in range(len(logodds_gen)) if golds[i] == 0]
    logodds_disc_pos = [logodds_disc[i] for i in range(len(logodds_disc)) if golds[i] == 1]
    logodds_disc_neg = [logodds_disc[i] for i in range(len(logodds_disc)) if golds[i] == 0]
    corr_pos = pearsonr(logodds_gen_pos, logodds_disc_pos).statistic
    spear_pos = spearmanr(logodds_gen_pos, logodds_disc_pos).statistic
    if len(logodds_gen_neg) == 0:
        corr_neg = np.nan
        spear_neg = np.nan
    else:
        corr_neg = pearsonr(logodds_gen_neg, logodds_disc_neg).statistic
        spear_neg = spearmanr(logodds_gen_neg, logodds_disc_neg).statistic
    print(f"correlation: all = {corr_all}, pos = {corr_pos}, neg = {corr_neg}")
    print(f"spearman: all = {spear_all}, pos = {spear_pos}, neg = {spear_neg}")
    disc_acc, disc_roc = compute_disc_accuracy(golds, logodds_disc)
    print(f"disc_acc: {disc_acc}, disc_roc: {disc_roc}")

    gen_acc_dict = compute_gen_accuracy(golds, ranks, thresholds = [5, 10, 40, 100, 1000])

    gen_mrr_pos, gen_mrr_neg = compute_gen_mrr(golds, ranks)
    print(f"gen_mrr_pos: {gen_mrr_pos}, gen_mrr_neg: {gen_mrr_neg}")

    return {
        'corr_all': corr_all,
        'corr_pos': corr_pos,
        'corr_neg': corr_neg,
        'disc_acc': disc_acc,
        'disc_roc': disc_roc,
        'gen_acc_dict': gen_acc_dict,
        'gen_mrr_pos': gen_mrr_pos,
        'gen_mrr_neg': gen_mrr_neg,
        'spear_all': spear_all,
        'spear_pos': spear_pos,
        'spear_neg': spear_neg
    }




def compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=-1, layer_disc=-1):

    if task=='hypernym':
        gold = [i.taxonomic.capitalize() for i in L]
    elif task=="trivia-qa":
        if 'correct' in L[0]:
            gold = [i['correct'] for i in L]
        else:
            gold = ['Yes' for i in L]
    elif task=='swords':
        gold = [i.synonym.capitalize() for i in L]
    elif task=='lambada':
        if 'correct' in L[0]:
            gold = [i['correct'] for i in L]
        else:
            gold = ['Yes' for i in L]
    else:
        raise ValueError("!")

    print("correlation: zs gen, fs disc (more usual)")
    # print(logodds_gen[0].size)
    # print(type(logodds_gen[0]) == torch.Tensor)
    if logodds_gen[0].dim() == 0:
        corr = pearsonr(
            [i.tolist() for i in logodds_gen],
            [i.tolist() for i in logodds_disc]
        ).statistic
    else:
        corr = pearsonr(
            [i[layer_gen].tolist() for i in logodds_gen],
            [i[layer_disc].tolist() for i in logodds_disc]
        ).statistic
    print(corr)

    disc_accuracy = sklearn.metrics.accuracy_score(gold, makepreds_disc(logodds_disc, threshold=0, layer_disc=layer_disc))
    print("\naccuracy of discriminator fs: {}".format(disc_accuracy))
    # disc_accuracy_new, roc = compute_disc_accuracy(task, L, logodds_disc)
    # print("\naccuracy of discriminator fs (new): {}, roc{}".format(disc_accuracy_new, roc))

    gen_accuracies = {} #map threshold to accuracy
    for threshold in [5, 10, 40, 100, 1000]:
        a = sklearn.metrics.accuracy_score(gold, makepreds_gen(ranks, threshold=threshold))
        gen_accuracies[threshold]= a
        print("accuracy of generator zs: (th={}: {})".format(threshold, a))
    return disc_accuracy, gen_accuracies, corr


# def compute_logodds(
#     task, P_gen, P_disc, L, tokenizer, first_sw_token, yestoks, notoks, layer_gen=-1, layer_disc=-1
# ):

#     if task=='hypernym':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii].noun2)[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='trivia-qa':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii]['answers'][0])[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='swords':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii].replacement)[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     elif task=='lambada':
#         ranks = [
#             get_rank(
#                 P_gen[ii][layer_gen, :], tokenizer.encode("a " + L[ii]['final_word'])[first_sw_token]
#             )
#             for ii in tqdm(range(len(P_gen)))
#         ]
#     else:
#         raise ValueError("!!")

#     logodds_gen = [get_logodds_gen(P_gen, L, ii, tokenizer, first_sw_token, task) for ii in range(len(P_gen))]
#     logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]

#     disc_accuracy, gen_accuracies, corr = compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks, layer_gen=layer_gen, layer_disc=layer_disc)
#     res_dict = compute_metrics(task, L, logodds_gen, logodds_disc, ranks)
#     # return res_dict
#     return ranks, logodds_gen, logodds_disc, corr


def compute_logodds_final_layer(
    task, P_gen, P_disc, L, tokenizer, first_sw_token, yestoks, notoks, is_chat = False):
    prefix = "a " if not is_chat else ""
    if task=='hypernym':
        # for ii in range(len(P_gen)):
        #     print(f'--compute-logodds, i=0:P:{P_gen[ii].shape}')
        #     print(f'--compute-logodds, i=0:P:{P_gen[ii].shape}')
        #     print(f'--compute-logodds, i=0:L:{L[ii]}')
        #     print(f'--compute-logodds, i=0:L:{L[ii].noun2}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.encode("a " + L[ii].noun2)}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.tokenize("a " + L[ii].noun2)}')
        #     print(f'--compute-logodds, i=0:encode:{tokenizer.encode("a " + L[ii].noun2)[first_sw_token]}')

        ranks = [
            get_rank(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii].noun2)[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task=='trivia-qa':
        ranks = [
            min(
                get_rank(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0])[first_sw_token]
                ),
                get_rank(
                    P_gen[ii][:], tokenizer.encode(prefix + L[ii]['answers'][0].capitalize())[first_sw_token]
                )
            )
            
            for ii in tqdm(range(len(P_gen)))
        ]
    elif task=='swords':
        if is_chat:
            ranks = [
                get_rank(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token]
                )
                for ii in tqdm(range(len(P_gen)))
            ]
        else:
            ranks = [
                get_rank(
                    P_gen[ii][:], tokenizer.encode(L[ii].replacement)[first_sw_token-1]
                )
                for ii in tqdm(range(len(P_gen)))
            ]
    
    elif task == 'lambada':
        ranks = [
            get_rank(
                P_gen[ii][:], tokenizer.encode(prefix + L[ii]['final_word'])[first_sw_token]
            )
            for ii in tqdm(range(len(P_gen)))
        ]
    else:
        raise ValueError("!!")

    logodds_gen = [get_logodds_gen(P_gen, L, ii, tokenizer, first_sw_token, task, is_chat=is_chat) for ii in range(len(P_gen))]
    logodds_disc = [get_logodds_disc(P_disc, ii, yestoks, notoks) for ii in range(len(P_disc))]

    # disc_accuracy, gen_accuracies, corr = compute_accuracy_and_correlations(task, L, logodds_gen, logodds_disc, ranks)
    res_dict = compute_metrics(task, L, logodds_gen, logodds_disc, ranks)
    print(res_dict)
    return res_dict

