import json
import os
import argparse

CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default='cleaned_python_test.json')
    args = parser.parse_args()

    data_file = os.path.join(dataset_generation_DIR, args.data_file)
    examples = json.load(open(data_file, 'r'))
    
    banned_keywords = []
    with open(os.path.join(CODE_DIR, 'resource/banned_keywords.txt'), 'r') as fin:
        for line in fin:
            if line.strip():
                banned_keywords.append(line.strip())
    
    total_count = 0
    valid_count = 0
    for idx,p in enumerate(examples):
        banned_flag = False 
        for k in banned_keywords:
            if k in "".join(p['code'].split()):
                banned_flag = True 
                break 
            
        examples[idx]['has_banned_keywords'] = banned_flag
        
        valid_count += not banned_flag
        total_count += 1
    
    json.dump(examples, open(data_file, 'w'), indent=4)
    
    print(f"{valid_count}/{total_count} examples where the focal methods do not have banned keywords")
