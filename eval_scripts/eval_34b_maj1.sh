#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --array=2
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=34b_maj1_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --error=34b_maj1_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --account=neox
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --requeue

HARNESS_DIR=/fsx/proj-mathlm/lm-evaluation-harness-dev

### begin configure eval parameters
models=("" "/fsx/proj-mathlm/downloaded-weights/CodeLlama-34b-hf" "/fsx/proj-mathlm/downloaded-weights/llemma_34b") 
names=("" "codellama_34b" "llemma_34b")

MODEL=${models[$SLURM_ARRAY_TASK_ID]}
NAME=${names[$SLURM_ARRAY_TASK_ID]}

OUT=${HARNESS_DIR}/output/${NAME}_maj1.json

TASKS=minerva_math*,gsm8k,ocw_courses,minerva-hendrycksTest*,math_sat_cot,sympy_math*,python_gsm8k

# uncomment line below to run a subset of tasks, useful for testing.
# TASKS=minerva_math_prealgebra,gsm8k,ocw_courses,minerva-hendrycksTest-abstract_algebra,math_sat_cot,sympy_math_prealgebra,python_gsm8k

cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output
### end configure eval parameters

### begin configure environment
TP_DEGREE=8

source ${HARNESS_DIR}/eval_scripts/env.sh
### end configure environment

# if testing, uncomment --limit for testing
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASKS --output_path ${OUT} --tp_degree ${TP_DEGREE} # --limit 10
