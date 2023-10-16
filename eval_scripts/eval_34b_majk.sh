#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --array=0-5
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=slurmouts/34b_majk/34b_majk_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --error=slurmouts/34b_majk/34b_majk_%A_%a.out      # Set this dir where you want slurm outs to go
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue

HARNESS_DIR=/fsx/proj-mathlm/lm-evaluation-harness-dev
cd $HARNESS_DIR

tasks=("minerva_math*"
       "gsm8k"
       "ocw_courses"
       "minerva-hendrycksTest*"
       "math_sat_cot"
       "sympy_math*"
       )
task_names=("minerva"
    "gsm"
    "ocw"
    "mmlu"
    "sat"
    "sympy"
    )


MODEL="/fsx/proj-mathlm/downloaded-weights/llemma_34b"

TASK=${tasks[$SLURM_ARRAY_TASK_ID]}
TASK_NAME=${task_names[$SLURM_ARRAY_TASK_ID]}

OUT=${HARNESS_DIR}/output/llema_34b_${TASK_NAME}_majk.json


cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output

### begin configure environment
source ${HARNESS_DIR}/eval_scripts/env.sh
### end configure environment

# if testing, uncomment --limit for testing
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASK --output_path ${OUT} --tp_degree 1 --description_dict_path ${HARNESS_DIR}/configs/majk.json --limit 10
