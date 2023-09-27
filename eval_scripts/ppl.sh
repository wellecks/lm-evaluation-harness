# Requires:
#   HARNESS_DIR: base directory of evaluation harness

PROFILE=open-web-math
TASKS=math_algebra_ppl,math_counting_and_prob_ppl,math_geometry_ppl,math_intermediate_algebra_ppl,math_num_theory_ppl,math_prealgebra_ppl,math_precalc_ppl,gsm8k_ppl

cd ${HARNESS_DIR}
mkdir -p ${HARNESS_DIR}/output

# owm models
for ENDPOINT in "owm_1b_v1.3" "proof-pile-v1_1b_v1.3" "mix_1b_v1.3" "pile-sample_1b_v1.3"; do

    MODEL=${PROFILE}/${ENDPOINT}
 
    OUT=${HARNESS_DIR}/output/${ENDPOINT}_ppl.json

    python main.py --no_cache --model hf-causal --model_args pretrained=${MODEL} --tasks $TASKS \
        --output_path ${OUT}
done

# pythia 1.4b
MODEL="EleutherAI/pythia-1.4b"
OUT=${HARNESS_DIR}/output/pythia-1.4b_ppl.json

python main.py --no_cache --model hf-causal --model_args pretrained=${MODEL} --tasks $TASKS \
    --output_path ${OUT}
