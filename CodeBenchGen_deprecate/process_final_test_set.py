import json
import numpy as np
from tqdm import tqdm 
import re 
import os 
import argparse

from src.ts_utils import ts_parser, find_func_node, find_main_node, remove_docstring, remove_comments, get_text

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_round", type=int, default=3)
    args = parser.parse_args()
    
    execution_result_data_file = os.path.join(dataset_generation_DIR, f"test_set_execution_round{args.max_round}.json")
    output_data_file = os.path.join(dataset_generation_DIR, f"test_set_final_round{args.max_round}.json")
    
    command_file = os.path.join(dataset_generation_DIR, f"test_set_commands_round{args.max_round}.sh")
    execution_command_file = os.path.join(dataset_generation_DIR, f"test_set_execution_commands_round{args.max_round}.json")
    output_command_file = os.path.join(dataset_generation_DIR, f"test_set_final_commands_round{args.max_round}.sh")
    
    test_data = json.load(open(execution_result_data_file, 'r'))
    test_data = [x for x in test_data if x['result'][0] == 'success']
    
    success_count = 0
    count = 0
    final_test_data = []
    for idx,example in enumerate(tqdm(test_data)):
        
        if example['result'][0] == 'success':
            success_count += 1
        else:
            continue
        
        code, full_func_name = example['code'], example['func_name']
        test_func_name = "test_" + example['func_name'].split('.')[-1]
        
        # 0. sanity check
        tree = ts_parser.parse(bytes(code, 'utf-8'))
        func_node = find_func_node(tree.root_node, full_func_name)
        if func_node is None:
            continue
        
        
        # 1. remove comments and the docstring in the focal method
        code_lines = code.split('\n')
        func_text = '\n'.join(code_lines[func_node.start_point[0]:func_node.end_point[0]+1])
        
        clean_func_text = remove_docstring(func_text)
        clean_code_lines = code_lines[:func_node.start_point[0]] + [clean_func_text] + code_lines[func_node.end_point[0]+1:]
        clean_code = '\n'.join(clean_code_lines)
        
        clean_code = remove_comments(clean_code)
        
        
        # 2. extract the start/end of the body of the focal method 
        # 3. replace the body by "...", save the content of the body as "reference_output"
        code_lines = clean_code.split('\n')
        tree = ts_parser.parse(bytes(clean_code, 'utf-8'))
        func_node = find_func_node(tree.root_node, full_func_name)
        if func_node is None:
            continue 
        func_text = '\n'.join(code_lines[func_node.start_point[0]:func_node.end_point[0]+1])
        
        first_tok = func_text.split()[0]
        indent = func_text.split(first_tok)[0]
        new_content = f"{indent}__MASK__"
        masked_code_lines = code_lines[:func_node.start_point[0]] + [new_content] + code_lines[func_node.end_point[0]+1:]
        masked_code = '\n'.join(masked_code_lines)
        example['masked_code'] = masked_code
        
        
        func_body_node = func_node.children[-1]
        if func_body_node.type != 'block' or len(get_text(func_body_node)) == 0:
            continue 
        func_body_text = '\n'.join(code_lines[func_body_node.start_point[0]:func_body_node.end_point[0]+1])
        
        first_tok = func_body_text.split()[0]
        indent = func_body_text.split(first_tok)[0]
        # new_content = f"{indent}..."
        new_content = f'{indent}"""insert the instruction here"""\n{indent}...'
        
        masked_method_code_lines = code_lines[:func_body_node.start_point[0]] + [new_content] + code_lines[func_body_node.end_point[0]+1:]
        masked_method_code = '\n'.join(masked_method_code_lines)
        example['gt_answer'] = func_body_text
        example['dummy_docstring'] = "insert the instruction here"
        
        
        # 4. extract the test function 
        # 5. replace the test function by "# test function here", save the content of the test function as "test_func"
        code_lines = masked_method_code.split('\n')
        tree = ts_parser.parse(bytes(masked_method_code, 'utf-8'))
        test_node = find_func_node(tree.root_node, test_func_name, duplicate_ok=True)
        if test_node is None:
            continue 
        test_content = '\n'.join(code_lines[test_node.start_point[0]:test_node.end_point[0]+1])
        
        example['test_function'] = test_content
        example['test_code'] = '\n'.join(code_lines[test_node.start_point[0]:]) # everything after the start of the test function
        example['masked_method_no_test'] = '\n'.join(code_lines[:test_node.start_point[0]])
        
        
        # if the test example satisfy all format requirements, save it to file.
        final_test_data.append(example)
        
            
    print(f"Num of examples that are successfully executed: {success_count}")
    print(f'\nSaving {len(final_test_data)}/{success_count} examples to file..')
    json.dump(final_test_data, open(output_data_file, 'w'), indent=4)
    
    
    # Aggregate commands
    commands = []
    with open(command_file, 'r') as fin:
        for line in fin:
            commands.append(line.strip())
    
    commands.extend(json.load(open(execution_command_file, 'r')))
    
    # skip pip uninstall commands 
    commands = [c for c in commands if 'uninstall' not in c]
    
    # delete unused commands (and keep the order the same as before)
    new_commands = []
    for c in commands:
        if c not in new_commands:
            new_commands.append(c)
    commands = new_commands
    
    print(f"Num of commands: {len(commands)}")
    with open(output_command_file, 'w') as fout:
        for cmd in commands:
            fout.write(cmd + '\n')
