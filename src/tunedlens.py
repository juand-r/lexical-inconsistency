"""
Can use `tunedlens.obtain_prob_tensor` instead of `logitlens.get_logitlens_output`
if I wish to use tuned lens. But we need a tuned model first. See:

https://tuned-lens.readthedocs.io/en/latest/

"""
import torch
import numpy as np
from tuned_lens.plotting import PredictionTrajectory
from tuned_lens import TunedLens

def init_lens(model_name, device):
    device = torch.device(device)
    global model
    global tuned_lens
    global tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tuned_lens = TunedLens.from_model_and_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def obtain_prob_tensor(input_ids, token_pos=-1):
    p = PredictionTrajectory.from_lens_and_model(
        tuned_lens,
        model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        targets=None,
    )
    X = torch.tensor(np.exp(p.log_probs[:, token_pos, :]))
    return X
