import json
import numpy as np
from tqdm import tqdm 

import os
import re
import sys
import time
import argparse

import openai

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

openai.api_base = "https://api.openai.com/v1"
openai.api_key = os.environ.get("OPENAI_API_KEY")
API_NAME = "gpt-4-turbo-2024-04-09"
# API_NAME = "gpt-3.5-turbo-0125"

prompt_template = "The following python code cannot be successfully executed. Debug and output the complete piece code with no omission. You should modify the code as little as possible, especially the test___FUNCTION_NAME__ function and the __FUNCTION_NAME__ function. The code will be run in an isolated environment with no GPU and no access to external APIs. The error message is:\n\n__ERROR_MSG__\n\nFix the code. Output one complete piece of python code in the format of ```python ... ```.\n\nIf the python code does not contain any error, copy the input python code to the output.\n\nIf necessary, output the bash scripts for Ubuntu in another code block in the format of ```bash ... ```.\n\n__CODE__\n\n"


def call_openai_completion(**kwargs):
    is_ok = False
    retry_count = 0
    while not is_ok:
        retry_count += 1
        try:
            response = openai.ChatCompletion.create(**kwargs)
            is_ok = True
        except openai.error.RateLimitError as error:
            if retry_count <= 20:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(5)
                continue
            else:
                print('Retry limit reached.')
                return {}
        except Exception as error:
            print(error)
            return {}
    return response
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1) # 1, 2, 3, ...
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, f"execution_round{args.round-1}_python_test.json")
    output_data_file = os.path.join(dataset_generation_DIR, f"simplified_round{args.round}_python_test.json")
    
    if not os.path.exists(output_data_file):
        examples = json.load(open(input_data_file, 'r'))
    else:
        examples = json.load(open(output_data_file, 'r'))

    
    compute_count = 0
    for idx,p in enumerate(tqdm(examples)):
        
        if f'result_round{args.round-1}' not in p:
            continue 
        
        if f'simplified_code_w_test_round{args.round}' in p:
            continue
        
        success_flag = False
        for k in p:
            if "result_round" in k and p[k][0] == 'success':
                success_flag = True 
                
        if success_flag:
            continue 
        
        input_code = p[f'simplified_code_w_test_round{args.round-1}']

        if not p[f'result_round{args.round-1}'][2]:
            err_msg = p[f'result_round{args.round-1}'][0]
        else:
            err_msg = p[f'result_round{args.round-1}'][2]
            
        err_msg = ' '.join(err_msg.split(' ')[:500])
        func_name = p['func_name']
        
        prompt = [
            {
                "role": "user",
                "content": prompt_template.replace('__FUNCTION_NAME__', p['func_name']).replace('__ERROR_MSG__', err_msg).replace('__CODE__', input_code)
            }
        ]
        
        if f'simplified_sh_code_w_test_round{args.round}' in p:
            sh_code = p[f'simplified_sh_code_w_test_round{args.round}']
            prompt[0]["content"] = prompt[0]["content"] + f"The following bash scripts are executed:\n\n{sh_code}\n\n"
        
        response = call_openai_completion(
            model=API_NAME, messages=prompt, max_tokens=4096,
        )
        
        try:
            response_text = response['choices'][0]['message']['content']
            print('-'*30, 'Output', '-'*30)
            print(response_text)
            print('='*60)
            compute_count += 1
            p[f"simplified_output_w_test_round{args.round}"] = response_text
        except:
            print('ERROR')
            continue
        
        if compute_count % 5 == 0 and compute_count > 0:
            print(f'Saving {compute_count}/{(idx+1)} examples to file..')
            json.dump(examples, open(output_data_file, 'w'), indent=4)
            
    print(f"Computed {compute_count} examples")
    json.dump(examples, open(output_data_file, 'w'), indent=4)