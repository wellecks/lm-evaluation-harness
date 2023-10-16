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


## Tasks Supported

Below, we detail all evaluations implemented and reported in our paper. 

* `math_sat_cot`: A small test set of SAT questions from the May 2023 College Board SAT examination, which occurred after the knowledge cutoff for Llemma's training set. Evaluated via chain-of-thought in natural language. 
* `hendrycks_math_ppl`: Perplexity evaluation on reference answers of sub-tasks of the [MATH dataset](https://arxiv.org/abs/2103.03874).
* `minif2f_isabelle`: Proof autoformalization in Isabelle on the miniF2F benchmark based on [Draft-Sketch-Prove](https://arxiv.org/abs/2210.12283), with a [Portal-to-Isabelle](https://github.com/albertqjiang/Portal-to-ISAbelle/tree/main) proof checker.
* `minerva_math*`: The MATH benchmark with the prompt and Sympy evaluation from [Minerva](https://arxiv.org/abs/2206.14858).
* `minerva-hendrycksTest*`: MMLU-STEM tasks with prompting and chain-of-thought, following [Minerva](https://arxiv.org/abs/2206.14858).
* `ocw_courses`: The OCW Courses task from [Minerva](https://arxiv.org/abs/2206.14858).
* `python_gsm8k`: GSM8k solved by writing Python programs that return the numeric answer, based on [PAL](https://arxiv.org/abs/2211.10435).
* `sympy_math`: MATH evaluation, with Sympy or Python `math` modules used to write a programmatic solution.

We additionally implement the following tasks in this fork, though we do not report them in our paper due to time+space limitations:

* `lila_*` - Evaluation on the [Lila dataset](https://arxiv.org/abs/2210.17517). Note that **this requires executing model-generated code**.
* `proofnet*` - Evaluation on the [ProofNet dataset](https://arxiv.org/abs/2302.12433) for both auto- and in- formalization. Informalization requires GPT-3.5 evaluation and an OpenAI API key.



# Quick Replication Instructions


## Maj@1 

To run the model on desired tasks with 1 attempt, run the following sample command with your model.

```
MODEL=EleutherAI/llemma_7b # your HF Hub model path here
TASK=minerva_math* # select tasks as desired. This codebase supports wildcard task names.
OUT=</path/to/save/outputs>

python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASK --output_path ${OUT} --tp_degree ${TP_DEGREE}
```

## Maj@K

To replicate Maj@K task results, additionally pass `--description_dict_path configs/majk.json` to run majority voting with K attempts. 

```
MODEL=EleutherAI/llemma_7b # your HF Hub model path here
TASK=minerva_math* # select tasks as desired. This codebase supports wildcard task names.
OUT=</path/to/save/outputs>

python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASK --output_path ${OUT} --tp_degree ${TP_DEGREE} --description_dict_path ${HARNESS_DIR}/configs/majk.json
```


`TP_DEGREE` can be set as needed to determine how many GPUs will be used by vLLM. 

Be sure to set $OUT to the desired save location for scores and model output text.


## Answer Checking + Scoring

Due to heavy CPU burden, we do not calculate metrics for tasks like `minerva_math` that rely on checking correctness via SymPy equivalence, or tasks like `sympy_math` or `python_gsm` that require execution of model-generated Python code. 

After running the model on one of these tasks, we provide utilities to perform answer checking.


Note that SymPy answer checking can be quite heavy on CPU resources and time-consuming for Maj@K at high K. 


:rotating_light: **WARNING: `unsafe_score_sympy_math.py` and `unsafe_score_python_gsm.py` will execute model-written Python code! Please use in a sandbox and at your own risk**.

:rotating_light: **WARNING: scoring scripts modify eval-harness output JSONs in-place. Back up your results files and use with caution!**

To score `sympy_math` outputs, run:

```
python unsafe_score_sympy_math.py --output <path-to-results-with-sympy_math>
```

To score `python_gsm8k` outputs, run

```
python unsafe_score_python_gsm.py --output <path-to-results-with-python_gsm8k>
```

These scripts will take in a results file, read in the LM's generated programs, and execute them to check for correctness. It will then incorporate the per-sample and full-task accuracies into the results file and rewrite the entire file with these values added.

All scripts allow for a `--limit X` flag to be passed to only score the first X documents.

**Due to the high resource cost of scoring MATH with SymPy, `unsafe_score_minerva_math.py` has additional requirements**.

To run Sympy scoring using multiprocessing, run

```
python unsafe_score_minerva_math.py --output <path-to-math-result.json>
```

To run in a single process, run 
```
python unsafe_score_minerva_math.py --output <path-to-math-result.json> --no_multiprocessing
```

Additionally, **MATH scoring with SymPy is resumable**--results and pass rates / accuracies for each document are saved to the results file in-place. *by default, the script will not rescore already-scored documents*.

## Aggregation

Finally, we provide utilities for aggregating MMLU and MATH scores across subtasks, aggregating at the sample level rather than the subset level. 

To aggregate MMLU-STEM scores, run:

```
python score_mmlu.py <path-to-mmlu-results.json> 
```

To aggregate MATH scores, run:

```
python score_math.py <path-to-math-subtask-1.json>,<path-to-math-subtask-2-and-3.json>,...
```



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
