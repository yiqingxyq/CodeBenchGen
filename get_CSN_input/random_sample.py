import json
import os
import argparse
import random


CODE_DIR = os.environ.get("CODE_DIR")
CACHE_DIR = os.environ.get("CACHE_DIR")
dataset_generation_DIR = os.environ.get("dataset_generation_DIR")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='python_test_0.jsonl')
    parser.add_argument("--output_file", type=str, default='python_test_0.sampled.json')
    parser.add_argument("--sampled_size", type=int, default=50)
    args = parser.parse_args()
    
    data_file = os.path.join(dataset_generation_DIR, args.input_file)
    sampled_data_file = os.path.join(dataset_generation_DIR, args.output_file)
    
    python_data = []
    with open(data_file, 'r') as fin:
        for line_id,line in enumerate(fin):
            d = json.loads(line)
            python_data.append(d)
            d['idx'] = line_id
    
    random.Random(1).shuffle(python_data)
    python_data = python_data[:args.sampled_size]
    
    json.dump(python_data, open(sampled_data_file, 'w'), indent=4)