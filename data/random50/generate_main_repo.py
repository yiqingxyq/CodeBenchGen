import json
from tqdm import tqdm 
import os
import re
import time
import argparse

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

docker_CACHE_DIR = os.environ.get("docker_CACHE_DIR")
repos_DIR = os.path.join(dataset_generation_DIR, "repos")

import sys 
sys.path.insert(0,CODE_DIR)
from CodeBenchGen.utils import extract_code, call_openai_completion

API_NAME = "gpt-4o"

init_prompt_template = f"""\
Instructions:
- You are given a PYTHON FUNCTION called __FUNCTION_NAME__ and its CONTEXT in the repository. 
- Your task is to write a __main__ function that calls the __FUNCTION_NAME__ function.
- You need to create your own inputs to the function. 
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.
- Provide your reasoning and the implementation of the __main__ function below SOLUTION. 

PYTHON CODE:
__CODE__

Your answer should follow the format below:

Reasoning: ...
```python
if __name__ == "__main__":
    # Your Code
```

Do NOT include other formatting. Only include the __main__ function in your output code block.

SOLUTION:
"""


debug_prompt_template = f"""\
Instructions:
- You are given a piece of PYTHON CODE and a __main__ function.
- Your task is to debug the __main__ function based on the ERROR MESSAGE. DO NOT modify other parts
- You need to create your own inputs to the function. 
- Make sure that your code can be directly executed without any modification. For example, statements like `token = "your_auth_token_here"  # You need to replace this with a real token` is NOT allowed.
- If you need to write files to the disk, use `{docker_CACHE_DIR}` as the directory.
- Provide your reasoning and the implementation of the __main__ function below SOLUTION. 

PYTHON CODE:
__CODE__

ERROR MESSAGE:
__ERROR_MSG__


Your answer should follow the format below:

Reasoning: ...
```python
if __name__ == "__main__":
    # Your Code
```

Do NOT include other formatting. Only include the debugged __main__ function in your output code block.

SOLUTION:
"""

def sanity_check(main_code):
    if MAIN_FUNC_STR not in main_code and MAIN_FUNC_STR2 not in main_code:
        return False
    
    if "try" in main_code and "except" in main_code:
        return False
    
    return True


MAIN_FUNC_STR = 'if __name__ == "__main__":'
MAIN_FUNC_STR2 = "if __name__ == '__main__':"
def extract_main_code(response_text):
    code_blocks = extract_code(response_text)
    for block_name, code in code_blocks:
        if sanity_check(code):
            code = code.replace(MAIN_FUNC_STR2, MAIN_FUNC_STR)
            code = MAIN_FUNC_STR + code.split(MAIN_FUNC_STR)[-1]
            print('-'*30, f"func_name: {func_name}", '-'*30)
            print('-'*30, 'Output', '-'*30)
            print(response_text)
            print('='*60)
            
            return code 

    print('Wrong format: No __main__ code block found.')
    return None


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='cleaned_python_test.json')
    parser.add_argument("--output_file", type=str, default='cleaned_python_test_main_round0.json')
    parser.add_argument("--mode", type=str, choices=["init", "debug"])
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    output_data_file = os.path.join(dataset_generation_DIR, args.output_file)
    
    examples = json.load(open(input_data_file, 'r'))
    
    compute_count = 0
    successful_count = 0
    for idx,func_dict in enumerate(tqdm(examples)):
        if len(extract_code(func_dict["context"])) != 1:
            continue 
        
        if args.mode == "debug":
            if "exec_result" not in func_dict or func_dict["exec_result"][0] == "success":
                continue 
        
        context = extract_code(func_dict["context"])[0][1]
        wrapped_context = f"```python\n{context}\n```"
        func_name = func_dict["func_name"]
        
        if args.mode == "init":
            prompt = init_prompt_template.replace('__FUNCTION_NAME__', func_name).replace('__CODE__', wrapped_context)
        elif args.mode == "debug":
            prompt = debug_prompt_template.replace('__FUNCTION_NAME__', func_name).replace('__CODE__', wrapped_context)
            prompt = prompt.replace('__ERROR_MSG__', func_dict["exec_result"][2] if func_dict["exec_result"][2] else "error")
            
            del func_dict["main_func"]
            del func_dict["exec_result"]
        
        response_text = call_openai_completion(
            model=API_NAME, messages=[{"role": "user", "content": prompt}], max_tokens=8192,
        )
        
        if response_text is None:
            print("Generation Error")
            continue
        
        main_code = extract_main_code(response_text)
        if main_code:
            func_dict["main_func"] = main_code 
            successful_count += 1
            
        compute_count += 1
        
        if compute_count % 10 == 0 and compute_count > 0:
            print(f'Saving {successful_count}/{compute_count}/{(idx+1)} examples to file..')
            json.dump(examples, open(output_data_file, 'w'), indent=4)
            
    json.dump(examples, open(output_data_file, 'w'), indent=4)