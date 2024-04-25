"""
Additional postprocessing to extract well-formed code outputs.

This step is optional for most models; it is only important for GPT models as
they have a tendency to omit function headers and respond only with function
bodies.
"""

import json
import re
import argparse
import sys

from tqdm import tqdm

from src.ts_utils import find_func_node, make_tree

CODE_BLOCK_PATTERN = r"(```|\[PYTHON\])(([ ]|\w)*)\n((.|\n)*?)\n(```|\[/PYTHON\])"


def extract_code(completion):
    pattern = CODE_BLOCK_PATTERN
    match = re.findall(pattern, completion)
    if match:
        return max((m[3] for m in match), key=lambda x: len(x))
    return ""


def check_func_syntax(func_code):
    """
    Very simple check that a function is syntactically valid.

    This should not be run on arbitrary code; we only use it on function code.
    In these cases, executing the code only has the effect of defining the function
    in scope (or triggering a syntax error).
    """
    try:
        exec(func_code)
    except SyntaxError:
        return False
    except:
        pass
    return True


def add_indent(code, indent_amt):
    spaces = indent_amt * " "
    return "\n".join([spaces + line for line in code.splitlines()])


def get_func_header(code, func_name):
    """Gets function header of function named `func_name` in code"""
    func_node = find_func_node(make_tree(code), func_name)
    header_start = func_node.start_byte
    header_end = func_node.named_children[-1].start_byte
    func_header = code[header_start:header_end]
    return func_header


def main(args):
    
    print(args.result_file)
    with open(args.result_file) as f:
        data = json.load(f)
    
    for example in tqdm(data):
        n_failed = 0
        if args.in_key not in example:
            continue
        completions = example[args.in_key]
        if isinstance(completions, str):
            completions = [completions]

        if args.out_key not in example:
            example[args.out_key] = ["" for _ in range(len(completions))]

        func_header = get_func_header(example["code"], example["func_name"])
        method_name = example["func_name"].split(".")[-1]

        for idx, completion in enumerate(completions):
            code = extract_code(completion)
            if f"def {method_name}" not in code:
                # assume we only have the function body in these cases
                orig_code = code
                code = func_header.strip() + "\n" + add_indent(orig_code, 4)

                syn_error = not check_func_syntax(code)
                if syn_error:
                    code = func_header + orig_code

            if code != "":
                example[args.out_key][idx] = code
            else:
                n_failed += 1

        if n_failed > 0:
            print(
                f"Failed to get {n_failed}/{len(completions)} generations for {example['func_name']}"
            )

    with open(args.result_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument(
        "--in_key",
        type=str,
        default="completions",
        help="Key mapping to raw model outputs",
    )
    parser.add_argument(
        "--out_key",
        type=str,
        default="gen_masked_method_no_test",
        help="Key mapping to model's code outputs",
    )

    args = parser.parse_args()

    main(args)
