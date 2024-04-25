import json
import numpy as np
from tqdm import tqdm 

import os
import re
import sys
import time
import argparse
from copy import deepcopy

import openai


CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

openai.api_base = "https://api.openai.com/v1"
openai.api_key = os.environ.get("OPENAI_API_KEY_OPENAI")
API_NAME = "gpt-4-turbo-2024-04-09"
# API_NAME = "gpt-3.5-turbo-0125"

prompt_template = [
  {
    "role": "system",
    "content": "Instruction: you are a programmar. Revise the following Python code in this way: copy the input code with no omission, generate or revise the test code for the foo function. Wrap the test code in a function named 'test_foo' and generate at least 3 assert statements. Output one complete piece of code, including the original input code with no omission, the test code, and code under 'if __name__=='__main__':'."
  },
  {
    "role": "system",
    "name": "example_user",
    "content": "import os\n\ndef foo(a):\n    if a > 0:\n        return a \n    else:\n        return -a"
  },
  {
    "role": "system",
    "name": "example_assistant",
    "content": "The functionality of foo() is to ... \nTo test the code, we should:\n(1) test the first if branch\n(2) check the second if branch \n(3) ...\n\n```\nimport os\n\ndef foo(a):\n    if a > 0:\n        return a \n    else:\n        return -a\n\ndef test_foo():\n  assert foo(1) == 1\n  assert foo[-1] == 1\n  assert foo(0) == 0\n\nif __name__=='__main__':\n  test_foo()\n```"
  },
  {
    "role": "system",
    "content": "Instruction: you are a programmar. Revise the following Python code in this way: copy the input code with no omission, generate or revise the test code for the __FULL_FUNCTION_NAME__ function. Wrap the test code in a function named 'test___FUNCTION_NAME__' and generate at least 3 assert statements. Output one complete piece of code, including the original input code with no omission, the test code, and code under 'if __name__=='__main__':'."
  },
  {
    "role": "user",
    "content": ""
  }
]


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
    
    input_data_file = output_data_file = os.path.join(dataset_generation_DIR, "simplified_round0_python_test.json")
    examples = json.load(open(input_data_file, 'r'))
        
    compute_count = 0
    for idx,p in enumerate(tqdm(examples)):
        if 'simplified_code_round0' not in p:
            continue
        
        if 'simplified_code_w_test_round0' in p:
            continue
        
        full_func_name = p['func_name']
        func_name = p['func_name'].split('.')[-1]
        
        prompt = deepcopy(prompt_template)
        prompt[-2]['content'] = prompt[-2]['content'].replace('__FULL_FUNCTION_NAME__', full_func_name).replace('__FUNCTION_NAME__', func_name)
        prompt[-1]['content'] = p['simplified_code_round0']
    
        response = call_openai_completion(
            model=API_NAME, messages=prompt, max_tokens=4096,
        )
        
        try:
            response_text = response['choices'][0]['message']['content']
            print(f'Testing {full_func_name} ...')
            print('-'*30, 'Output', '-'*30)
            print(response_text)
            print('='*60)
            compute_count += 1
            
            p['simplified_output_w_test_round0'] = response_text
        except:
            continue
        
        if compute_count % 5 == 0 and compute_count > 0:
            print(f'Saving {compute_count}/{(idx+1)} examples to file..')
            json.dump(examples, open(output_data_file, 'w'), indent=4)
            
    json.dump(examples, open(output_data_file, 'w'), indent=4)
            
