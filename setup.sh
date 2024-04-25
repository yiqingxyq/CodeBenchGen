export HOME_DIR="~"
export STORAGE_DIR=$HOME_DIR

# code
export CODE_DIR=${HOME_DIR}"/CodeBenchGen"
export TS_LANG_PATH=${CODE_DIR}/resource/python.so

# cache
export CACHE_DIR=${HOME_DIR}/"tmp"

# data
export dataset_generation_DIR=${STORAGE_DIR}"/CodeBenchGen_example"
export final_dataset_DIR=${STORAGE_DIR}"/ExecCSN"

# docker
export docker_HOME_DIR="/home/user"
export docker_CACHE_DIR=${docker_HOME_DIR}/"tmp"
export docker_dataset_generation_DIR=${docker_HOME_DIR}"/CodeBenchGen_example"
export docker_final_dataset_DIR=${docker_HOME_DIR}"/ExecCSN"