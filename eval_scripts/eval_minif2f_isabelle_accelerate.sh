export HF_DATASETS_CACHE="/net/nfs/mosaic/seanw/.cache"
export TRANSFORMERS_CACHE="/net/nfs/mosaic/seanw/.cache"

BASE_DIR="./"
OUTPUT_DIR="./output/minif2f_isabelle"
mkdir -p ${OUTPUT_DIR}

MODEL="EleutherAI/pythia-1.4b-deduped"
NAME="pythia-1.4b-deduped"

FEWSHOT=0
BATCH_SIZE=1

TASKS="minif2f_isabelle"

python ${BASE_DIR}/main.py --description_dict_path ${BASE_DIR}/configs/config_minif2f_isabelle.json \
	--model_args pretrained=${MODEL} \
	--num_fewshot ${FEWSHOT} \
	--model hf-causal \
	--use_accelerate \
	--accelerate_dtype float32 \
	--tasks ${TASKS} \
	--batch_size ${BATCH_SIZE} \
	--output_path ${OUTPUT_DIR}/${NAME}.json
