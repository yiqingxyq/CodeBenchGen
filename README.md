# CodeBenchGen 

<p align="left">
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-blue"></a>
  <a href="https://arxiv.org/abs/2404.00566"><img src="https://img.shields.io/badge/arXiv-2404.00566-b31b1b.svg"></a>
</p>

Code for "CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks" ([Arxiv](https://arxiv.org/abs/2404.00566))

If you find our paper or code useful, please cite the paper:
```
@misc{xie2024codebenchgen,
      title={CodeBenchGen: Creating Scalable Execution-based Code Generation Benchmarks}, 
      author={Yiqing Xie and Alex Xie and Divyanshu Sheth and Pengfei Liu and Daniel Fried and Carolyn Rose},
      year={2024},
      eprint={2404.00566},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```


## Table of content
 - [Setup](#environment)
   - [Environment Variables](#environment-var), [Libraries](#environment-lib), [Docker](#environment-docker)
 - [Use CodeBenchGen to Create Your Own Benchmark](#codebenchgen)
   - [Input](#codebenchgen-step0)
   - [Step 1: Sandboxing](#codebenchgen-step1)
   - [Step 2: Test Generation](#codebenchgen-step2)
   - [Step 3: Iterative Execution & Debugging](#codebenchgen-step3)
   - [Step 4: Post-processing](#codebenchgen-step4)
   - [(Optional) Test Augmentation ](#test-aug)
 - [Evaluation with Exec-CSN](#evaluation)


&nbsp;
<a id="environment"></a>
## Setup

<a id="environment-var"></a>
### Environment Variables
You'll need to set up the environment variables in `setup.sh` and run
```
source setup.sh
```

<a id="environment-lib"></a>
### External Libraries
If you need to build your own benchmark using CodeBenchGen, you'll need to install the following packages:
```
pip install transformers==4.21.0 tree_sitter==0.20.1 sacrebleu=="1.2.11"
```

<a id="environment-docker"></a>
### Setup a Docker for Code Execution
You're strongly recommended to execute the code inside a docker.
First, build the docker image using the `Dockerfile` we provide:
```
docker build --tag ${YOUR_IMAGE_NAME} .
```
You can check all your docker images using `docker image ls`

Then run a container using the docker image you just built:
```
docker run --name=${YOUR_CONTAINER NAME} -v ${dataset_generation_DIR}:${docker_dataset_generation_DIR} -v ${final_dataset_DIR}:${docker_final_dataset_DIR} -it ${YOUR_IMAGE_NAME}
```
The `-v` command allows you to access and revise the following folders inside the container: (1) `${dataset_generation_DIR}`: the folder containing all the datafiles when building your benchmark, and (2) ${final_dataset_DIR} the folder containing the final dataset you want to evaluate your code generation models on.

You can check all your containers using `docker ps`


&nbsp;
<a id="codebenchgen"></a>
## Use CodeBenchGen to Create Your Own Benchmark

<a id="codebenchgen-step0"></a>
### Input
To create a benchmark, you'll need to prepare a set of code snippets and select a segment of the code in each input snippet as the target code. In our scripts, we assume the target code is the body of a function (i.e., the "focal method"). You are also welcome to revise the scripts so that they can support target code in other formats.

You should set the directory of the input file in `setup.sh`. The input should be a .JSON file containing a list of dictionaries. Each dictionary is of the following format:
```
{
  "context":     the content of the code, containing the target code and its context,
  "func_name":   the name of the focal method (the target code),
  "idx":         the index of this example,
}
```

Optionally, you can also create the input data from the CodeSearchNet dataset. Assume you want to sample the input data from the test dataset named `python_test_0.jsonl`, you can run:
```
# Randomly select k examples
python get_CSN_input/random_sample.py --input_file "python_test_0.jsonl" --output_file "python_test_0.sampled.json" --sampled_size ${YOUR_SAMPLE_SIZE}

# Download the context based on the url
python get_CSN_input/download_context.py --input_file "python_test_0.sampled.json" --output_file "cleaned_python_test.json"

# Detect banned keywords in the list we provide
python detect_banned_keywords.py --data_file "cleaned_python_test.json"

```
The input data will be stored in `${dataset_generation_DIR}/cleaned_python_test.json`.

&nbsp;
<a id="codebenchgen-step1"></a>
### Step 1: Sandboxing
The first step is to sandbox the input code so that we can test it in an isolated environment:
```
python CodeBenchGen/sandboxing.py --input_file ${YOUR_INPUT_FILE}
python CodeBenchGen/extract_code.py --round 0 --output_key "simplified_output"
```

Note that you're recommended to run this step multiple times in case the model fails to generate code in the correct format for some examples.

The results will be stored to `${dataset_generation_DIR}/simplified_round0_python_test.json`

&nbsp;
<a id="codebenchgen-step2"></a>
### Step 2: Test Generation
The second step is to generate a testing function for each example:
```
python CodeBenchGen/generate_test_cases.py
python CodeBenchGen/extract_code.py --round 0 --output_key "simplified_output_w_test"
```

Similarly, you can run this step multiple times.
The results will be stored to `${dataset_generation_DIR}/simplified_round0_python_test.json`

&nbsp;
<a id="codebenchgen-step3"></a>
### Step 3: Iterative Execution & Debugging
The third step is to iteratively execute and debug the generated examples. You're strongly recommended to execute the code in a docker container:
```
# Execute the code in the docker
python CodeBenchGen/execution.py --round $K --mode "debug"
```

You don't need to run debugging inside the docker:
```
python CodeBenchGen/debug.py --round $K
python CodeBenchGen/extract_code.py --round $K --output_key "simplified_output_w_test"
```
You can run `debug.py` and the `extract_code.py` multiple times. The execution results will be saved in `${dataset_generation_DIR}/execution_round${K}_python_test.json` and the commands that are run will be saved in `${dataset_generation_DIR}/execution_commands_round${K}_python_test.json`. The debugging results will be saved in `${dataset_generation_DIR}/simplified_round${K}_python_test.json`.

&nbsp;
<a id="codebenchgen-step4"></a>
### Step 4: Post-processing
We have multiple sub-steps in post-processing.

#### (4-1) Environment Check
First, we aggregate the examples where the target output can pass all the test cases:
```
python CodeBenchGen/aggregate_examples.py --max_round ${MAX_ROUND}
```

In the docker, we run all examples in two passes to ensure they can share the same environment. Where `--skip_sh` prevents the execution of any shell commands:
```
# In the docker
python CodeBenchGen/execution.py --round ${MAX_ROUND} --mode "final"
python CodeBenchGen/execution.py --round ${MAX_ROUND} --mode "final" --skip_sh
```

We then collect the remaining examples where the target output can pass all the test cases in the shared environment:
```
python CodeBenchGen/process_final_test_set.py --max_round ${MAX_ROUND}
```


#### (4-2) Instruction Generation
Then we generate the instructions for each example:
```
python CodeBenchGen/generate_instructions.py --max_round ${MAX_ROUND}
```

The data file containing your final benchmark will be `${dataset_generation_DIR}/test_set_final_round${MAX_ROUND}.json`. The shell commands executed will be aggregated in `${dataset_generation_DIR}/test_set_final_commands_round${MAX_ROUND}.json`.

&nbsp;
 <a id="test-aug"></a>
### (Optional) Test Augmentation 
To improve the test coverage rate, you can generate additional tests for your dataset. 

#### Generate candidate tests
The first step is to generate candidate tests. An example command using gpt-3.5 to generate additional tests is as follows:
```
python test_augmentation/augment_tests.py \
  --data_file ${dataset_generation_DIR}/test_set_final_round${MAX_ROUND}.json \        # the dataset file
  --output_file ${dataset_generation_DIR}/test_set_aug.json \                          # the output file (containing the test candidates)
  --model_name gpt-3.5-turbo-0125 \  # the model used to generate additional tests
  --model_backend openai \           # openai, vllm, or hf
  --prompt oracle                    # the prompt template for test augmentation. check test_augmentation/test_aug_prompts.py for details.
```

We also support open-source models (both huggingface and vllm backend). For example:
```
python test_augmentation/augment_tests.py \
  --data_file ${dataset_generation_DIR}/test_set_final_round${MAX_ROUND}.json \
  --output_file ${dataset_generation_DIR}/test_set_aug.json \
  --model_name deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --model_backend vllm \
  --prompt oracle
```


#### Execute candidate tests
Since some candidate tests may not be correct, in the second step, we execute all the candidate tests and make sure that the ground truth reference can pass all tests.
```
# Execute the code in the docker
python test_augmentation/execute_aug_tests.py \
  --data_file ${docker_dataset_generation_DIR}/test_set_final_round${MAX_ROUND}.json \       # the dataset file
  --test_file ${docker_dataset_generation_DIR}/test_set_aug.json \                           # the output file (containing the test candidates)
  --output_file ${docker_dataset_generation_DIR}/test_set_aug_exec_results.json \            # the execution results for the candidate tests
  --prompt oracle
```

#### Add successful tests to the data file
Finally, we put the tests that can be passed by the ground truth reference in the data file by running:
```
python test_augmentation/make_aug_data.py \
  --data_file ${dataset_generation_DIR}/test_set_final_round${MAX_ROUND}.json \       # the dataset file
  --new_test_files ${dataset_generation_DIR}/test_set_aug_exec_results.json \         # the execution results for the candidate tests
  --output_file ${dataset_generation_DIR}/aug_test_set_final_round${MAX_ROUND}.json   # the output file, containing the additional tests
```
The additional tests will be saved to a key `new_test_code`.



&nbsp;
 <a id="evaluation"></a>
## Evaluation with Exec-CSN
You can evaluate code generation models on Exec-CSN or your new dataset. 

The datafile of Exec-CSN is `ExecCSN_dataset/test_set_final_round3.json` and the commands to setup the docker environment are stored in `ExecCSN_dataset/test_set_final_commands_round3.sh`

#### Inference 
We provide the code for inference with both open-source models and openAI APIs.
```
python evaluation/inference.py \
  --model_name deepseek-ai/deepseek-coder-7b-instruct-v1.5 \     # the model to evaluate. Can be an open-source model or an openAI API
  --model_backend vllm \                                         # can be vllm, hf, openai
  --data_path ${final_dataset_DIR}/aug_test_set_final_round${MAX_ROUND}.json \   # the path to the .json data file
  --output_path "results/generation_outputs.json"                                # the path to store the output file
```

Then we post-process the generation outputs:
```
python evaluation/postprocess_code.py \
  --result_file "results/generation_outputs.json" \     # the path to store the output file
  --out_key gen_masked_method_no_test                   # the processed results will be saved as this key
```


#### Pass@k evaluation 
Finally, evaluate with Pass@k:
```
# Execute the code in the docker
python evaluation/execution_passk.py \
  --data_file ${final_dataset_DIR}/aug_test_set_final_round${MAX_ROUND}.json \   # the path to the .json data file
  --hypo_file "results/generation_outputs.json" \   # the file with generation outputs
  --code_gen_model deepseek7B \                     # the save name of the evaluated model
  --output_file "results/execution_results.json"    # the evaluation results will be saved here
```


