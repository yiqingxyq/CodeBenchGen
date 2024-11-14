import os
import json
import argparse
from tqdm import tqdm

from r2e.generators.testgen import TestGenArgs, R2ETestGenerator
from r2e.utils.data import load_functions
from r2e.paths import EXTRACTED_DATA_DIR
from r2e.utils.data import write_functions

def write_chat_hist(chat_hist_list, chat_hist_file):
    json.dump(chat_hist_list, open(chat_hist_file, 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default='random50_sampled20')
    parser.add_argument("--context_type", type=str, default='sliced')
    parser.add_argument("--max_context_size", type=int, default=6000)
    args = parser.parse_args()
    
    input_func_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_extracted.json"
    output_func_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_context.json"
    chat_hist_file = EXTRACTED_DATA_DIR / f"{args.exp_id}_chat_hist.json"
    
    functions = load_functions(input_func_file)
    test_gen_args = TestGenArgs(exp_id=args.exp_id, context_type="sliced", max_context_size=6000)
    tasks = R2ETestGenerator.prepare_tasks(test_gen_args, functions)
    
    functions_w_context = [x.func_meth for x in tasks]
    chat_hist_list = [x.chat_messages for x in tasks]
    write_functions(functions_w_context, output_func_file)
    write_chat_hist(chat_hist_list, chat_hist_file)
    