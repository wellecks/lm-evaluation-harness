export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

BASE_DIR="./"
OUTPUT_DIR="./output/minerva_math"
mkdir -p ${OUTPUT_DIR}

MODEL="EleutherAI/pythia-1.4b-deduped"
NAME="pythia-1.4b-deduped"

FEWSHOT=0
BATCH_SIZE=1

TASKS="minerva_math_algebra,minerva_math_prealgebra,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_precalc"

python ${BASE_DIR}main.py --description_dict_path ${BASE_DIR}configs/config_math.json \
	--model_args pretrained=${MODEL} \
	--num_fewshot ${FEWSHOT} \
	--model vllm \
	--tasks ${TASKS} \
	--batch_size ${BATCH_SIZE} \
	--output_path ${OUTPUT_DIR}/${NAME}.json
