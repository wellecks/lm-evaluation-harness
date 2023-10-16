module load openmpi cuda/11.8

source /admin/home-hailey/miniconda3/bin/activate eval-vllm-new

CONDA_HOME=/admin/home-hailey/miniconda3/envs/eval-vllm-new
CUDNN_HOME=/fsx/hailey/cudnn-linux-x86_64-8.6.0.163_cuda11-archive

export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDNN_HOME/include:$CPATH

export PATH=$CONDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_HOME/include:$CPATH

export LD_PRELOAD=/usr/local/cuda-11.7/lib/libnccl.so

export HF_DATASETS_CACHE=${HARNESS_DIR}/.cache
export TRANSFORMERS_CACHE=${HARNESS_DIR}/.cache
