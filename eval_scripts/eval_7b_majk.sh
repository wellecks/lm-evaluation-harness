#!/bin/bash

HARNESS_DIR=/fsx/proj-mathlm/lm-evaluation-harness-dev
cd $HARNESS_DIR

models=("/fsx/proj-mathlm/downloaded-weights/Llama-2-7b-hf" "/fsx/proj-mathlm/downloaded-weights/CodeLlama-7b-hf" "/fsx/proj-mathlm/downloaded-weights/llemma_7b") 
model_names=("llama-2_7b" "codellama_7b" "llemma_7b")

tasks=("minerva_math_prealgebra"
       "minerva_math_algebra"
       "minerva_math_intermediate_algebra"
       "minerva_math_num_theory"
       "minerva_math_counting_and_prob"
       "minerva_math_geometry"
       "minerva_math_precalc"
       "gsm8k,ocw_courses"
       "minerva-hendrycksTest*,math_sat_cot"
       "sympy_math*,python_gsm8k")
task_names=("prealgebra"
       "algebra"
       "intermediate_algebra"
       "num_theory"
       "counting_and_prob"
       "geometry"
       "precalc"
       "rest_nl"
       "mul_choice"
       "tools")

# Bash metaprogramming is not as nice as Lean...

for i in {0..2}; do
    MODEL=${models[$i]}
    MODEL_NAME=${model_names[$i]}

    for j in {0..9}; do
        TASK=${tasks[$j]}
        TASK_NAME=${task_names[$j]}

        OUT=${HARNESS_DIR}/output/${MODEL_NAME}_${TASK_NAME}_majk.json

        cat > temp.sh << EOL
#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1                 # Number of gpus
#SBATCH --output=slurmouts/7b_majk/${MODEL_NAME}_${TASK_NAME}_majk_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=slurmouts/7b_majk/${MODEL_NAME}_${TASK_NAME}_majk_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue


HARNESS_DIR=${HARNESS_DIR}

cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output
### end configure eval parameters

### begin configure environment

source ${HARNESS_DIR}/eval_scripts/env.sh
### end configure environment

# if testing, uncomment --limit for testing
python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASK --output_path ${OUT} --tp_degree 1 --description_dict_path ${HARNESS_DIR}/configs/majk.json
EOL
        
        sbatch temp.sh

        rm temp.sh

    done 
done
