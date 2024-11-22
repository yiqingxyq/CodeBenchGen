import os 
import json
import numpy as np
from tqdm import tqdm 
import re 
import argparse

from transformers import GPT2TokenizerFast
from src.ts_utils import ts_parser, find_func_node, find_main_node, check_function_calls_by_name, remove_comments

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
SH_CODE_BLOCK_PATTERN = r"```bash\n(.*?)\n```"

def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN):
    match = re.findall(pattern, text, flags=re.DOTALL)
    return match if match else [("code_not_found", text)]

def extract_python_block(text):
    code_blocks = extract_code(text)
    python_code_blocks = [x[1] for x in code_blocks if x[0] != 'bash']
    sh_code_blocks = [x[1] for x in code_blocks if x[0] == 'bash']
    if len(python_code_blocks)==1:
        return python_code_blocks[0], sh_code_blocks
    else:
        return None, sh_code_blocks

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output_key", type=str, default='simplified_output_w_test', choices=['simplified_output', 'simplified_output_w_test'])
    
    parser.add_argument("--skip_length_check", default=False, action='store_true')
    parser.add_argument("--skip_omission_check", default=False, action='store_true')
    parser.add_argument("--skip_keyword_check", default=False, action='store_true')
    
    args = parser.parse_args()
    
    data_file = os.path.join(dataset_generation_DIR, f"simplified_round{args.round}_python_test.json")
    
    if args.round >= 1:
        input_code_key = f"simplified_code_w_test_round{args.round-1}"
    else:
        if args.output_key == "simplified_output":
            input_code_key = "context"
        elif args.output_key == "simplified_output_w_test":
            input_code_key = "simplified_code_round0"
    
    output_key = f"{args.output_key}_round{args.round}"
    output_code_key = output_key.replace('output', 'code')
    output_sh_code_key = output_key.replace('output', 'sh_code')
    
    simplified_python = json.load(open(data_file))
    
    # compute length
    if not args.skip_length_check:
        input_code_list = []
        output_code_list = []
        for i,p in enumerate(simplified_python):
            if output_key not in p:
                input_code_list.append("")
                output_code_list.append("")
                continue
            input_code = remove_comments(p[input_code_key])
            input_code_list.append(input_code)
            
            output_text = p[output_key]
            output_code, _ = extract_python_block(output_text)
            output_code_list.append(remove_comments(output_code) if output_code else "")
            
        tokenizer = GPT2TokenizerFast.from_pretrained('RaymondLi/gpt-4-tokenizer')
        tok_input_code_list = tokenizer(input_code_list)
        tok_output_code_list = tokenizer(output_code_list)
        input_code_lens = [len(x) for x in tok_input_code_list['input_ids']]
        output_code_lens = [len(x) for x in tok_output_code_list['input_ids']]
    
        assert len(input_code_lens) == len(output_code_lens) == len(simplified_python)
    
    output_count = 0
    code_count = 0
    sh_code_count = 0
    total_count = len(simplified_python)
    for i,p in enumerate(simplified_python):
        
        if output_code_key in p:
            del p[output_code_key]
        if output_sh_code_key in p:
            del p[output_sh_code_key]
            
        if output_key not in p:
            continue
        
        output = p[output_key]
        full_func_name = p['func_name']
        func_name = p['func_name'].split('.')[-1]
        output_count += 1
        
        # basic check: check whether the python block exist && the focal method exist
        output_code, sh_code_blocks = extract_python_block(output)
        if output_code is None:
            print([i], 'CODE FORMAT NOT CORRECT')
            continue
        
        tree = ts_parser.parse(bytes(output_code, 'utf-8'))
        func_node = find_func_node(tree.root_node, full_func_name, duplicate_ok=False)
        if func_node is None:
            # if f"def{func_name}(" not in ''.join(output_code.split()):
            print([i], 'FOCAL METHOD NOT EXIST')
            continue
        focal_method = func_node.text.decode("utf-8")
        
        
        # keywork check: check whether the code has banned keywords
        if not args.skip_keyword_check:
            banned_keywords = []
            with open(os.path.join(CODE_DIR, 'resource/banned_keywords.txt'), 'r') as fin:
                for line in fin:
                    if line.strip():
                        banned_keywords.append(line.strip())
                        
            banned_flag = False 
            for k in banned_keywords:
                if k in "".join(output_code.split()):
                    print([i], f'BANNED KEYWORD EXISTS: {k}')
                    banned_flag = True 
                    break 
                
            if banned_flag:
                continue  
        
        
        # omission check: it should not contain '...', except for in the code
        if not args.skip_omission_check:
            code_str = output_code.replace('[...','').replace('...]','')
            if '...' in code_str or "remains unchanged" in code_str:
                print([i], 'CODE NOT COMPLETE')
                continue
            
            
        # length check: different rules for different steps 
        if not args.skip_length_check:
            input_len, output_len = input_code_lens[i], output_code_lens[i]
            length_fail_flag = False
            
            if args.round == 0:
                if args.output_key == "simplified_output": # input: 'context'
                    if input_len > 600:
                        if output_len < 200:
                            length_fail_flag = True
                    else:
                        if output_len < 50:
                            length_fail_flag = True
                        
                elif args.output_key == "simplified_output_w_test":
                    if output_len + 30 < input_len:
                        length_fail_flag = True 
            else:
                if output_len + 50 < input_len:
                    length_fail_flag = True 
                    
            if length_fail_flag:
                print([i], 'OUTPUT TOO SHORT')
                continue
        
        # format check for the test functions:
        # 1. The test function name should be test_XXX
        # 2. Num of asserts > certain numbers 
        # 3. Call the focal method at least once 
        # 4. Have a __main__ function to call the tests
        if 'w_test' in output_key:
            test_func_name = f"test_{func_name}"
            
            # 1. The test function name should be test_XXX
            test_node = find_func_node(tree.root_node, test_func_name, duplicate_ok=True)
            if test_node is None:
                # if f"def{test_func_name}(" not in ''.join(output_code.split()):
                print([i], 'TEST FUNCTION NOT EXIST')
                continue
            test_code = test_node.text.decode("utf-8")
            
            
            # 2. Num of asserts > certain numbers 
            assert_count = test_code.lower().count('assert') # raise AssertionException also counts
            if assert_count < 3:
                print([i], f'NOT ENOUGH ASSERT STATEMENTS (FOUND {assert_count})')
                continue 
            
            # 3. Call the focal method at least once 
            func_call_flag = check_function_calls_by_name(test_node, func_name)
            if not func_call_flag:
                print([i], 'DOES NOT CALL THE FOCAL METHOD')
                continue 
            
            # 4. Have a __main__ function to call the tests
            main_node = find_main_node(tree.root_node)
            if main_node is None:
                print([i], "if__name__=='__main__' NOT EXIST")
                continue 
            
            
        code_count += 1
        p[output_code_key] = output_code
            
        sh_code = '\n'.join(sh_code_blocks)
        if 'simplified_sh_code_round0' in p:
            sh_code = p['simplified_sh_code_round0'] + '\n' + sh_code
            
        sh_code = sh_code.replace('sudo ', '') # we don't need root access in docker
        if len(sh_code) > 0:
            sh_code_count += 1
            p[output_sh_code_key] = sh_code
            
    print(f"Num of examples in {output_key}: {output_count}")
    print(f"Among which, num of examples with a valid format: {code_count}")
    print(f'Num of examples with shell scripts: {sh_code_count}')
    
    json.dump(simplified_python, open(data_file, 'w'), indent=4)
    
    
   