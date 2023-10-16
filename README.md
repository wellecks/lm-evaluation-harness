# [Math] Language Model Evaluation Harness

This section contains documentation related to changes/additions made to the lm-evaluation-harness.
The original readme for the evaluation harness is in the subsequent section.
 
This fork contains code to replicate the evaluations as performed in the Llemma paper. If you use the evaluation implementations in this paper, including the newly introduced SAT evaluation, please cite 


```bibtex

```

As well as the corresponding papers for the benchmarks used, which can be found in the implementation files.


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



## Further Documentation

See `eval_scripts` for example scripts, including slurm scripts we used (only included for referencing). We will walk through a simple case: 

### LILA with Huggingface accelerate
Example command to evaluate Pythia-1.4b-deduped on the `lila_addsub` task:
```bash
MODEL="EleutherAI/pythia-1.4b-deduped"
NAME="pythia-1.4b-deduped"

BASE_DIR="./"
OUTPUT_DIR="./output/lila"
mkdir -p ${OUTPUT_DIR}

FEWSHOT=5
BATCH_SIZE=1

TASKS="lila_addsub"

python ${BASE_DIR}/main.py --model_args pretrained=${MODEL} \
	--description_dict_path ${BASE_DIR}/configs/config_lila.json \
	--num_fewshot ${FEWSHOT} \
	--model hf-causal \
	--use_accelerate \
	--accelerate_dtype float32 \
	--tasks ${TASKS} \
	--output_path ${OUTPUT_DIR}/${NAME}.json \
	--batch_size ${BATCH_SIZE} 
```

**NOTE:** the `--accelerate_dtype float32` flag is needed to match the performance of the non-`accelerate` code.

**NOTE:** the `--description_dict_path` flag provides a config file (`configs/config_lila.json`) file. \
See `configs/config_math.json` for a non-trivial config file that enables a custom prompt and majority voting.




# New features

## Majority voting, generation hyperparameters

Additional generation options can be specified through a configuration file and the `--description_dict_path` argument.
For example, to enable majority voting with temperature 0.3 on the `math_algebra` task, we create a `config.json` file containing a `params` field:
```json
{
    "math_algebra": {
        "params": {"majority_voting": 16, "sampling_temperature":0.5, "eval_batch_size":4},
    }
}
```
then pass the file through the `--description_dict_path` argument:
```bash
python main.py --model gpt2 \
    --tasks math_algebra 
    --description_dict_path config.json 
    --device cuda 
    --num_fewshot 3 
```
**Warning:** Currently only the tasks defined in `hendrycks_math.py` support these options. If you are interested in adding this functionality to other tasks, see [this guide](./docs/task_guide.md).

## Prepending a task description in the prompt
In the `config` file, you can add a `description` field containing a string. The string will be prepended to each prompt during evaluation.
Continuing the example from above, we have a `config.json` file containing:
```json
{
    "math_algebra": {
        "params": {"majority_voting": 16, "sampling_temperature":0.5, "eval_batch_size":4},
        "description": "You will solve a mathematical problem. Here are some examples:", 
    }
}
```

## Accelerate
You can use the HuggingFace `accelerate` library. To do so, use the `--use_accelerate` flag along with a `hf-causal` model. 
Here is an example command:
```bash
python main.py \
    --model hf-causal \
    --use_accelerate \
    --model_args pretrained=EleutherAI/pythia-2.8b-deduped \
    --num_fewshot 5 \
    --tasks lila_addsub
```
NOTE: With default settings, `--model hf-causal` may have different performance than `--model gpt2`. One known discrepancy is that `hf-causal` may use `float16` by default, while `--model gpt2` uses `float32`. Add the command-line argument `--accelerate_dtype float32` to prevent this discrepancy.

NOTE: we do not yet support `hf-seq2seq`.

## vLLM support

This fork supports vLLM for fast and optionally tensor-parallel inference. (Which can be controlled by passing `--model vllm` and `--tp_degree ${TP_DEGREE}`). For more information, including a list of supported models, see https://github.com/vllm-project/vllm . 

## Experimental gpt-based evaluation for ProofNet informalization
The ProofNet informalization task (`proofnet_informalize_statements`) consists of mapping a formal theorem statement to an informal statement.
We provide an experiment GPT-based evaluation as a proxy of correctness.

Given a `(formal theorem statement, gold informal statement, generated informal statement)` triple, a `gpt-3.5-turbo` or `gpt-4` model is prompted to decide whether the generated informal statement is correct, and to provide a reason for the decision.

To enable:
- Set your openai api key as an environment variable:
```bash
export OPENAI_API_KEY="..."
```
- Enable it in the config file (e.g. `configs/config_proofnet.json`):
```json
{
  "proofnet_autoformalize_statements" : {
    "description": "",
    "params": {}
  },
  "proofnet_informalize_statements" : {
    "description": "",
    "params": {
      "gpt_eval" : {
        "enabled": true,
        "settings": {
          "engine": "gpt-3.5-turbo",
          "max_tokens": 512
        }
      }
    }
  }
}
```


A full example script is in `eval_scripts/eval_proofnet_accelerate.sh`. 

### Important considerations about experimental gpt-based evaluation:
- This evaluation costs money, since it calls the openai api.
- This evaluation is not fully reproducible since it depends on the openai api.
- This evaluation has not been extensively validated as an evaluation methodology. We hypothesize that it may serve as a proxy of correctness that is useful for relative comparison of models.

# Language Model Evaluation Harness

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)

## Overview

This project provides a unified framework to test autoregressive language models (GPT-2, GPT-3, GPTNeo, etc) on a large number of different evaluation tasks.

Features:

- 200+ tasks implemented. See the [task-table](./docs/task_table.md) for a complete list.
- Support for GPT-2, GPT-3, GPT-Neo, GPT-NeoX, and GPT-J, with flexible tokenization-agnostic interface.
- Task versioning to ensure reproducibility.

## Install

```bash
pip install lm-eval
```

To install additional multlingual tokenization and text segmenation packages, you must install the package with the `multilingual` extra:

```bash
pip install "lm-eval[multilingual]"
```

## Basic Usage

> **Note**: When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility. This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible. See the [Task Versioning](#task-versioning) section for more info.

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) you can use the following command:


```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks lambada_openai,hellaswag \
    --device 0
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partialy trained checkpoints:

```bash
python main.py \
    --model hf-causal \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks lambada_openai,hellaswag \
    --device 0
```

To evaluate models that are called via `AutoSeq2SeqLM`, you instead use `hf-seq2seq`.

> **Warning**: Choosing the wrong model may result in erroneous outputs despite not erroring.

Our library also supports the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially mantained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model gpt3 \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

To evaluate mesh-transformer-jax models that are not available on HF, please invoke eval harness through [this script](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.



## Advanced features

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/task_guide.md).

## Task Versioning

To help improve reproducibility, all tasks have a `VERSION` field. When run from the command line, this is reported in a column in the table, or in the "version" field in the evaluator return dict. The purpose of the version is so that if the task definition changes (i.e to fix a bug), then we can know exactly which metrics were computed using the old buggy implementation to avoid unfair comparisons. To enforce this, there are unit tests that make sure the behavior of all tests remains the same as when they were first implemented. Task versions start at 0, and each time a breaking change is made, the version is incremented by one.

When reporting eval harness results, please also report the version of each task. This can be done either with a separate column in the table, or by reporting the task name with the version appended as such: taskname-v0.

## Test Set Decontamination

For details on text decontamination, see the [decontamination guide](./docs/decontamination.md).

Note that the directory provided to the `--decontamination_ngrams_path` argument should contain the ngram files and info.json. See the above guide for ngram generation for the pile, this could be adapted for other training sets.

```bash
python main.py \
    --model gpt2 \
    --tasks sciq \
    --decontamination_ngrams_path path/containing/training/set/ngrams \
    --device 0
```

## Cite as

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
