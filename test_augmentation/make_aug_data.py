"""Constructs new data json file with augmented test suites"""

import argparse
import json
import os

from src.ts_utils import find_func_node, ts_parser

CODE_DIR = os.environ.get("CODE_DIR")

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default="data/test_set_final_round3.json")
parser.add_argument(
    "--new_test_files",
    type=str,
    default="exec_new_test_output_extend.json",
    help="Comma-separated list of json files containing *consistent* new tests.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="data/aug_test_set_final_round3.json",
    help="New data json file with all valid tests appended to current tests",
)
parser.add_argument(
    "--new_tests_only",
    action='store_true',
    help="Only include new tests and remove original tests",
)

args = parser.parse_args()

with open("kept_indices.txt") as f:
    keep_indices = [int(n) for n in f.read().splitlines()]

with open(args.data_file) as f:
    data = json.load(f)

new_test_files = args.new_test_files.split(",")

# Do not add test cases if they contain potentially dangerous commands from the list below
DANGEROUS_COMMANDS = []
with open(os.path.join(CODE_DIR, 'resource/banned_keywords.txt'), 'r') as fin:
    for line in fin:
        if line.strip():
            DANGEROUS_COMMANDS.append(line.strip())

new_tests = []
for new_test_file in new_test_files:
    with open(new_test_file) as f:
        new_tests.append(json.load(f))


no_aug = 0
changed = []

for idx, example in enumerate(data):

    func_name = example["func_name"]
    test_name = f"\ndef test_{func_name.split('.')[-1]}"
    masked_code = example["masked_code"]

    if args.new_tests_only:
        test_func_start = masked_code.find(test_name)
        if test_func_start == -1:
            print("Unable to find test function, skipping.")
            no_aug += 1
            continue

        masked_code = masked_code[:test_func_start].rstrip()

    root_node = ts_parser.parse(bytes(example["code"], "utf-8")).root_node
    func_node = find_func_node(root_node, func_name, duplicate_ok=True)
    func_code = func_node.text.decode("utf-8")
    example["ref_func"] = [func_code]

    if idx not in keep_indices:
        continue

    new_test_codes = [
        tests[idx]["new_test_code"]
        for tests in new_tests
        if tests[idx].get("new_test_code")
    ]

    if len(new_test_codes) == 0:
        no_aug += 1
        continue

    new_test = "\n\n".join(new_test_codes)

    orig_code = example["code"]
    if any((cmd in new_test or cmd in orig_code) for cmd in DANGEROUS_COMMANDS):
        no_aug += 1
        continue

    masked_code += "\n\n\n" + new_test

    changed.append(str(idx))
    example["masked_code"] = masked_code

with open(args.output_file, "w+") as f:
    json.dump(data, f)

print(
    f"Successfully augmented {len(keep_indices) - no_aug} / {len(keep_indices)} = {(1 - no_aug/len(keep_indices)) * 100:.2f}% examples"
)
print(
    f"     Failed to augment {no_aug} / {len(keep_indices)} = {no_aug / len(keep_indices) * 100:.2f}% examples"
)

print(len(changed))
output_dir_path = os.path.dirname(args.output_file)
with open( os.path.join(output_dir_path, "changed.txt"), "w+") as f:
    f.write("\n".join(changed))