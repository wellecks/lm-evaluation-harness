# `Llemma` evaluation harness

Fork of the Eleuther LM Evaluation Harness used in [Azerbayev et al 2023]().

This branch contains code for the informal-to-formal theorem proving task.

## Running the evaluation 

The task consists of generation and proof checking.\
To run generation, see `eval_scripts/minif2f_isabelle.sh` for an entrypoint.

Please see [docs/isabelle_setup.md](docs/isabelle_setup.md) for instructions on setting up and running proof checking.


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
