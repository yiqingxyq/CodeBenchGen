import subprocess
import json
from tqdm import tqdm 
import os
import argparse

import random

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='python_test_0.sampled.json')
    parser.add_argument("--output_file", type=str, default='cleaned_python_test.json')
    args = parser.parse_args()
    
    output_DIR = CACHE_DIR
    if not os.path.exists(output_DIR):
        os.mkdir(output_DIR)
    
    input_data_file = os.path.join(dataset_generation_DIR, args.input_file)
    output_data_file = os.path.join(dataset_generation_DIR, args.output_file)
    
    if os.path.exists(output_data_file):
        python_data = json.load(open(output_data_file, 'r'))
    else:
        # Read data 
        python_data = []
        python_data_input = json.load(open(input_data_file, 'r'))
        python_data = []
        for d in python_data_input:
            python_data.append({
                'url': d['url'],
                'docstring': d['docstring'],
                'code': d['code'],
                'func_name': d['func_name'],
                'idx': d['idx'],
            })
        
    # run wget command to extract .json files 
    wrong_format_count = 0
    total_count = 0
    for d in tqdm(python_data):
        if 'context' in d:
            continue
        file_name = f"{output_DIR}/py{d['idx']}.py"
        if os.path.exists(file_name):
            continue
        
        download_url = d['url'].replace('/blob/', '/raw/').split('#')[0]
        command = f"wget -O {file_name} {download_url}"
        
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)
    
    # put the context to orig dict 
    wrong_format_count = 0
    total_count = 0
    for i,d in enumerate(python_data):
        if 'context' in d:
            continue
        
        file_name = f"{output_DIR}/py{d['idx']}.py"
        if os.path.exists(file_name):
            with open(file_name, 'r') as fin:
                file_content = fin.read()
            # if file_content:
            if d['func_name'].split('.')[-1] in file_content:
                python_data[i]['context'] = file_content
            else:
                print(f"CANNOT READ FILE: {d['idx']}")
                wrong_format_count += 1
        else:
            print(f"FILE NOT EXIST: {d['idx']}")
            wrong_format_count += 1
        total_count += 1
            
    print(f"Wrong format count: {wrong_format_count}/{total_count}")
    
    json.dump(python_data, open(output_data_file, 'w'), indent=4)
    
    subprocess.run(f'rm {output_DIR}/*', shell=True)