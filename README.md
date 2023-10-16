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

The task consists of generation and proof checking.\
To run generation, see `eval_scripts/minif2f_isabelle.sh` for an entrypoint.

Please see [docs/isabelle_setup.md](docs/isabelle_setup.md) for instructions on setting up and running proof checking.

## Additions

This `Llemma` evaluation harness implemented several extensions of the Eleuther LM Evaluation Harness at the time of development.
Note that these may have been implemented in the Harness subsequently. An incomplete list includes:
- Support for `vLLM`
- Saving generated sequences and metadata
- Majority voting (see `configs/majk.json` for example usage)
- Temperature and top-p sampling
- Domain-specific evaluation (e.g. Sympy equivalence)


## Citation
```
@article{azerbayev2023llemma,
    title={Llemma: an open language model for mathematics},
    author={Zhangir Azerbayev and Hailey Schoelkopf and Keiran Paster and Marco Dos Santos and Stephen McAleer and Albert Q. Jiang and Jia Deng and Stella Biderman and Sean Welleck},
    eprint={xyz.xyz},
    archivePrefix={arXiv}
    year={2023}
}

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
