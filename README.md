Discriminator-Generator Gap in LM knowledge of lexical semantics
================================================================

This repository contains code for experiments to analyze the generator/discriminator gap, attempt to fix it, and (ideally) analyze this mechanistically.

**Work in progress.**

Details
-------

Language models have slight inconsistencies between generator and discriminator versions of questions. This is a more specific problem than inconsistency between prompt variations (refs:...). For example, to probe knowledge of hypernymy, we can prompt:

- (Generator)  "A bee is a kind of"  [look at next token predictions]

- (Discriminator)   "Do you think a bee is a kind of furniture? Answer:" [Yes/No]

We can look at the degree of confidence/certainty of the answer by looking at log-odds, for both generator and discriminator. (Sanity check: for the generator, these correlate with -log(rank), but should re-test this with every new model).

Goals
-----

- Characterize and describe the gap
-  Devise method(s) to close the gap
    - Fine-tuning
    - Model-surgery
- Evaluation
    - (important) Does the language model still function well, or have we specialized it so much that it’s damaged in some way? ("relation specificity" in Knowledge Editing literature)

    - (important) Evaluate whether gap is closed (graded notion)
    - (good to have) Mechanistically, is the LM using a different computational pathway after the modification?

Repo organization
-----------------

- `src/` contains visualization code, an implementation of logitlens, and utilities for loading and formatting the text
- `data/` contains the hypernymy dataset.
- `scripts/`:
    - Use `fine_tune_lora.py` to SFT models with LoRA on variations of our prompts. Models will be saved in `models/`.
    - Use `logodds.py` to run logitlens on the test set, save the log-odds at the last position across layers, and compute accuracy and correlations. Log-odds will be saved in `outputs`.
- `notebooks`: Jupyter notebooks to look plot some of the results.

