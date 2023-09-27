# Requires:
#   HARNESS_DIR: base directory of evaluation harness

PROFILE=open-web-math
TASKS=math_algebra_easy
CONFIG=${HARNESS_DIR}/configs/majk.json

cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output

# owm models
for ENDPOINT in "owm_1b_v1.3" "proof-pile-v1_1b_v1.3" "mix_1b_v1.3" "pile-sample_1b_v1.3"; do

    MODEL=${PROFILE}/${ENDPOINT}
 
    OUT=${HARNESS_DIR}/output/${ENDPOINT}_majk.json

    python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASKS \
        --output_path ${OUT} --tp_degree 1 --description_dict_path ${CONFIG} --num_fewshot 5
done

# pythia 1.4b
MODEL="EleutherAI/pythia-1.4b"
OUT=${HARNESS_DIR}/output/pythia-1.4b_majk.json

python main.py --no_cache --model vllm --model_args pretrained=${MODEL} --tasks $TASKS \
    --output_path ${OUT} --tp_degree 1 --description_dict_path ${CONFIG}
