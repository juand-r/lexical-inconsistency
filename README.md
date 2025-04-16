RankAlign
==========

This repository contains code for experiments to analyze the Generator-Validator gap and methods to fix from our paper [RankAlign: A Ranking View of the Generator-Validator Gap in Large Language Models](https://arxiv.org/abs/2504.11381)

Citation:
```
@article{rodriguez2025rankalign,
  title={RankAlign: A Ranking View of the Generator-Validator Gap in Large Language Models},
  author={Rodriguez, Juan Diego and Ding, Wenxuan and Erk, Katrin and Durrett, Greg},
  journal={arXiv preprint arXiv:2504.11381},
  year={2025}
}
```


The Generator-Validator Gap
---------------------------

Language models can be inconsistent when prompted with generator and validator forms of the same question. For example, to probe knowledge of hypernymy, we can prompt:

- (Generator)  "A bee is a kind of"  [look at next token predictions]

- (Discriminator)   "Do you think a bee is a kind of furniture? Answer:" [Yes/No]

We can look at the degree of model confidence/certainty of the answer by looking at log-odds, for both generator and discriminator, and quantify the Generator-Validator gap through the correlation of these log-odds. This repo contains code to evaluate the Generator-Validator gap, and to run our method, **RankAlign**, and other baselines from our paper.


Repo organization
-----------------

- `data/` contains the hypernymy dataset from [Rodriguez et al., 2025](https://arxiv.org/abs/2410.22590), and versions of the SWORDS lexical substitution dataset, LAMBADA, and TriviaQA.
- `scripts/`:
    - RankAlign is contained in `ranking_loss_ref.py`
    - Other baselines:
       * Use `fine_tune_lora.py` to SFT models with LoRA on variations of our prompts.
       * Use `dpo.py` to run the DPO baselines.
       * Use `consistency_ft.py` to run the consistency fine-tuning baseline.
- `models/` and `outputs/` is where model checkpoint and eval results are saved, respectively.

