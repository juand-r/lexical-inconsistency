"""
Implementation of logit lens using nnsight.
For more information, see https://nnsight.net/notebooks/tutorials/logit_lens/

"""

from nnsight import LanguageModel
import torch
import seaborn as sns
import numpy as np

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
    if modelname_short == "gpt":
        layers = model.transformer.h
    if modelname_short in ["gemma-2", "meta-llama/Meta-Llama-3-8B-Instruct"]:
        layers = model.model.layers

    probs_layers = []

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                if modelname_short == "gpt":
                    layer_output = model.lm_head(
                        model.transformer.ln_f(layer.output[0])
                    )
                elif modelname_short in [
                    "gemma-2",
                    "meta-llama/Meta-Llama-3-8B-Instruct",
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
