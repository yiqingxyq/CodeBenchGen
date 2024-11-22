import json
import re 
import os 
import argparse

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_round", type=int, default=3)
    args = parser.parse_args()
    
    input_data_file = os.path.join(dataset_generation_DIR, f"execution_round{args.max_round}_python_test.json")
    output_data_file = os.path.join(dataset_generation_DIR, f"test_set_round{args.max_round}.json")
    output_command_file = os.path.join(dataset_generation_DIR, f"test_set_commands_round{args.max_round}.sh")
    
    examples = []
    final_results = json.load(open(input_data_file, 'r'))
    
    for example in final_results:
        already_success_flag = False
        for i in range(args.max_round+1):
            exec_result_key = f'result_round{i}'
            code_key = f'simplified_code_w_test_round{i}'
            sh_code_key = f'simplified_sh_code_w_test_round{i}'
            
            if exec_result_key in example and example[exec_result_key][0] == 'success' and code_key in example:
                assert already_success_flag == False 
                
                new_example = {
                    'code': example[code_key],
                    'idx': example['idx'],
                    'func_name': example['func_name'],
                }
                if 'docstring' in example:
                    new_example['docstring'] = example['docstring']
                
                if sh_code_key in example:
                    new_example['sh_code'] = example[sh_code_key]
                examples.append(new_example)
                
                already_success_flag = True 
                
    # write commands into file 
    commands = []
    for rid in range(args.max_round+1):
        command_file = os.path.join(dataset_generation_DIR, f"execution_commands_round{rid}_python_test.json")
        cur_commands = json.load(open(command_file))
        for c in cur_commands:
            if c not in commands:
                commands.append(c)
                
    # skip pip uninstall commands 
    commands = [c for c in commands if 'uninstall' not in c]
    
    # delete unused commands (and keep the order the same as before)
    new_commands = []
    for c in commands:
        if c not in new_commands:
            new_commands.append(c)
    commands = new_commands
             
    print(f'Num of Final examples: {len(examples)}')   
    json.dump(examples, open(output_data_file, 'w'), indent=4)
    
    print(f"Num of commands: {len(commands)}")
    with open(output_command_file, 'w') as fout:
        for c in commands:
            fout.write(c+'\n')