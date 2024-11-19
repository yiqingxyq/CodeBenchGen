export HOME_DIR=$PWD
export STORAGE_DIR=$HOME_DIR"/data"

# code
export CODE_DIR=${HOME_DIR}
export TS_LANG_PATH=${CODE_DIR}/resource/python.so

# cache
export CACHE_DIR=${HOME_DIR}/"tmp"

# data
export dataset_generation_DIR=${STORAGE_DIR}"/random50"
export final_dataset_DIR=${CODE_DIR}"/ExecCSN_dataset"

# docker
export docker_HOME_DIR="/home/user"
export docker_CODE_DIR=${docker_HOME_DIR}"/CodeBenchGen"
export docker_CACHE_DIR=${docker_HOME_DIR}/"tmp"
export docker_dataset_generation_DIR=${docker_HOME_DIR}"/random50"
export docker_final_dataset_DIR=${docker_HOME_DIR}"/random50"