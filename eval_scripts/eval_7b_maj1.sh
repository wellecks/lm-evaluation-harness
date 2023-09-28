#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --array=0-2
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:1                 # Number of gpus
#SBATCH --output=7b_maj1_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --error=7b_maj1_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue

HARNESS_DIR=/fsx/proj-mathlm/lm-evaluation-harness-dev

### begin configure eval parameters
profiles=("meta-llama" "codellama" "open-web-math") 
endpoints=("Llama-2-7b-hf" "CodeLlama-7b-hf" "llemma_7b")

PROFILE=${profiles[$SLURM_ARRAY_TASK_ID]}
ENDPOINT=${endpoints[$SLURM_ARRAY_TASK_ID]}

MODEL=${PROFILE}/${ENDPOINT}
OUT=${HARNESS_DIR}/output/${ENDPOINT}_maj1.json

TASKS=minerva_math*,gsm8k,ocw_courses,minerva-hendrycksTest*,math_sat_cot,sympy_math*,python_gsm8k

# uncomment line below to run a subset of tasks, useful for testing.
# TASKS=minerva_math_prealgebra,gsm8k,ocw_courses,minerva-hendrycksTest-abstract_algebra,math_sat_cot,sympy_math_prealgebra,python_gsm8k

cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output
### end configure eval parameters

### begin configure environment
TP_DEGREE=1

source ${HARNESS_DIR}/eval_scripts/env.sh
### end configure environment

# if testing, uncomment --limit for testing
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASKS --output_path ${OUT} --tp_degree ${TP_DEGREE} # --limit 10
