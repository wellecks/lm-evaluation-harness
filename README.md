# `Llemma` evaluation harness

Fork of the Eleuther LM Evaluation Harness used in [Azerbayev et al 2023]().

## Running the evaluation 

See `eval_scripts/generic_run.sh` for an entrypoint to running evaluation on a model from the HuggingFace Hub.

The script shows the set of non-theorem-proving tasks:
```bash
SYMBOLIC=minerva_math*,gsm8k,ocw_courses
MUL_CHOICE=minerva-hendrycksTest*,math_sat_cot
TOOLS=sympy_math*,python_gsm8k
```

Refer to `lm_eval/tasks` directory for their associated implementations.

#### Theorem proving task

The informal-to-formal theorem proving task is kept in the `minif2f-isabelle` branch. Please see the README in this branch for further instructions.


## Additions

This `Llemma` evaluation harness implemented several extensions of the Eleuther LM Evaluation Harness at the time of development.
Note that these may have been implemented in the Harness subsequently. An incomplete list includes:
- Support for `vLLM`
- Saving generated sequences and metadata
- Majority voting (see `configs/majk.json` for example usage)
- Temperature and top-p sampling
- Domain-specific evaluation (e.g. Sympy equivalence)


## Citation
Please cite the Llemma paper if you find code from this fork useful to your work:
```
@article{
}
```




Please cite the Eleuther LM Evaluation Harness using:
```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
