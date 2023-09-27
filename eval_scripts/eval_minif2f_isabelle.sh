BASE_DIR="."
OUTPUT_DIR="./output/minif2f_isabelle"
mkdir -p ${OUTPUT_DIR}

MODEL="codellama/CodeLlama-7b-hf"
NAME="codellama_CodeLlama-7b-hf"

FEWSHOT=0
BATCH_SIZE=1

TASKS="minif2f_isabelle_informal2formal"

python ${BASE_DIR}/main.py --description_dict_path ${BASE_DIR}/configs/config_minif2f_isabelle.json \
	--model_args pretrained=${MODEL} \
	--num_fewshot 0 \
	--model vllm \
	--tasks ${TASKS} \
	--batch_size ${BATCH_SIZE} \
	--output_path ${OUTPUT_DIR}/${NAME}.json
