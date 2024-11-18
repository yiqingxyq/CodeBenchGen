import os
import json
import argparse
from tqdm import tqdm

from r2e.generators.testgen import TestGenArgs, R2ETestGenerator
from r2e.utils.data import load_functions
from r2e.paths import EXTRACTED_DATA_DIR
from r2e.utils.data import write_functions

dataset_generation_DIR = os.environ.get("dataset_generation_DIR")

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
    
    # obtain context
    functions = load_functions(input_func_file)
    test_gen_args = TestGenArgs(exp_id=args.exp_id, context_type="sliced", max_context_size=6000)
    tasks = R2ETestGenerator.prepare_tasks(test_gen_args, functions)
    
    # save files
    functions_w_context = [x.func_meth for x in tasks]
    chat_hist_list = [x.chat_messages for x in tasks]
    write_functions(functions_w_context, output_func_file)
    write_chat_hist(chat_hist_list, chat_hist_file)
    
    # stats
    # functions_w_context = json.load(open(output_func_file))
    multi_file_context_count = len([x for x in functions_w_context if x["context"]["context"].count("```python") > 1 ])
    print(f"Examples with multi-file context: {multi_file_context_count}/{len(functions_w_context)}")
    
    # process into CodeBenchGen's format
    os.makedirs(dataset_generation_DIR, exist_ok=True)
    codebenchgen_input_file = os.path.join(dataset_generation_DIR, "cleaned_python_test.json")
    
    codebenchgen_inputs = []
    for idx,func_dict in enumerate(functions_w_context):
        if "function_id" in func_dict:
            func_name = func_dict["function_id"]["identifier"].split(".")[-1]
            func_code = func_dict["function_code"]
            
        elif "method_id" in func_dict:
            func_name = ".".join( func_dict["method_id"]["identifier"].split(".")[-2:] )
            func_code = func_dict["method_code"]
            
        codebenchgen_inputs.append({
            "func_name":   func_name,
            "idx":         str(idx),
            "func_code":   func_code,
            "context":     func_dict["context"]["context"],
            "repo_name":   func_dict["file"]["file_module"]["repo"]["repo_id"],
            "func_path":   func_dict["file"]["file_module"]["module_id"]["identifier"].replace(".","/") + ".py",
        })
    json.dump(codebenchgen_inputs, open(codebenchgen_input_file, "w"), indent=4)