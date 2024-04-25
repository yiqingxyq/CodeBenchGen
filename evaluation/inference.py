"""
Main inference script for code gen models
"""

import copy
import json
import logging
import os
import re
import argparse
from typing import Any

import torch
from tqdm import tqdm

from src.decoders import ModelBackend, build_decoder
from src.utils import (
    extract_code,
    get_instruction_template,
    get_stop,
    get_system_prompt,
    replace_docstring,
    set_chat_template,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(args) -> list[dict]:
    """
    Reads in test data for inference.

    If args.use_originial_docstring is set, the original docstrings of the target methods
    are replaced with the revised instructions.
    """
    with open(args.data_path) as f:
        data = json.load(f)

    if args.start_index > 0 or args.end_index != -1:
        data = data[args.start_index : args.end_index]

    logger.info(f"Loaded data, {len(data)} examples.")

    if args.use_originial_docstring:
        instruction_key = "docstring"
    else:
        instruction_key = "revised_instruction"
        
    for i, example in enumerate(data):
        original_docstring = example["dummy_docstring"].strip()
        revised_docstring = example[instruction_key].strip()
        original_code = example[args.data_key]

        revised_code = replace_docstring(
            original_code, original_docstring, revised_docstring
        )
        if revised_code == "":
            revised_code = original_code
        example[args.data_key] = revised_code
    return data


def load_model(args) -> ModelBackend:
    """
    Loads LLM decoder
    """
    decoder_args = {}

    if args.model_backend == "vllm":
        if args.model_cache_dir is not None:
            decoder_args["download_dir"] = args.model_cache_dir

        n_gpus = torch.cuda.device_count()
        decoder_args["tensor_parallel_size"] = n_gpus

        # This may be necessary when running on GPUs that do not support bfloat16
        # decoder_args['dtype'] = torch.float16

    return build_decoder(args.model_name, args.model_backend, **decoder_args)


def inference_batch(
    args,
    model: ModelBackend,
    batch_inputs: list[dict],
    batch_outputs: list[dict],
    generation_config: dict[str, Any],
) -> tuple[list[dict], bool]:
    """
    Args:
        model: model for inference
        batch_inputs: list of input data dicts in batch
        batch_outputs: list of output dicts in batch, to be populated in-place with code outputs
        generation_config: model sampling configurations (i.e. temperature, top-p, etc)

    Returns:
        batch_outputs
        is_modified
    """
    batch_prompts = []
    tokenizer = model.get_tokenizer()

    # add chat templates for tokenizers that don't originally have them
    set_chat_template(args.model_name, tokenizer)

    gen_indices = []  # indices of batch members that were generated

    for idx, (input_data, output_data) in enumerate(zip(batch_inputs, batch_outputs)):
        if args.output_key in output_data:
            continue

        gen_indices.append(idx)
        func_name = input_data["func_name"]
        code = input_data[args.data_key]
        instruction = args.instruction_template.format(func_name=func_name, code=code)

        if args.model_backend != "openai":
            chat_history = []
            if args.system_prompt is not None:
                chat_history.append({"role": "system", "content": args.system_prompt})
            chat_history.append({"role": "user", "content": instruction})
            prompt = tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = instruction

        batch_prompts.append(prompt)

    if len(batch_prompts) == 0:
        return batch_outputs, False

    is_first = not os.path.isfile(args.output_path)
    if is_first:
        # for the first example, print input as sanity check
        logger.info("Prompt:\n" + batch_prompts[0])

    if len(batch_prompts) == 1:
        batch_prompts = batch_prompts[0]

    completions, _ = model(batch_prompts, **generation_config)

    if is_first:
        logger.info("Completion:\n" + completions[0])

    for idx, gen_idx in enumerate(gen_indices):
        output_data = batch_outputs[gen_idx]
        curr_completions = completions[
            idx * args.n_samples : (idx + 1) * args.n_samples
        ]
        curr_codes = [extract_code(c, args.model_name) for c in curr_completions]
        n_failed = sum(c == "" for c in curr_codes)
        if n_failed > 0:
            func_name = batch_inputs[gen_idx]["func_name"]
            logger.warning(
                f"Failed to get {n_failed}/{args.n_samples} generations for {func_name}"
            )
        output_data["completions"] = curr_completions
        output_data[args.output_key] = curr_codes

    return batch_outputs, True


def main(args):
    data = load_data(args)

    model = load_model(args)

    logger.info(f"Writing outputs to file {args.output_path}")
    if os.path.isfile(args.output_path):
        with open(args.output_path, "r") as f:
            outputs = json.load(f)
    else:
        outputs = copy.deepcopy(data)

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": args.n_samples,
        "max_tokens": args.max_tokens,
        "stop": get_stop(args.model_name),
    }

    model.set_generation_config(generation_config)

    args.system_prompt = get_system_prompt(args.model_name)
    args.instruction_template = get_instruction_template(args.model_name)

    for n_batch, start_idx in enumerate(
        tqdm(range(0, len(data), args.batch_size), disable=args.disable_tqdm)
    ):
        end_idx = min(start_idx + args.batch_size, len(data))
        batch_inputs = data[start_idx:end_idx]
        batch_outputs = outputs[start_idx:end_idx]

        batch_outputs, is_modified = inference_batch(
            args, model, batch_inputs, batch_outputs, generation_config
        )

        if is_modified:
            outputs[start_idx:end_idx] = batch_outputs

            # save partial results
            if n_batch % args.save_freq == 0:
                logger.info(f"Saving partial output at batch #{n_batch+1}")
                with open(args.output_path, "w+") as out_fp:
                    json.dump(outputs, out_fp)

    with open(args.output_path, "w+") as out_fp:
        json.dump(outputs, out_fp)

    logger.info("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of model used for code generation. Can be either an HF or OpenAI model",
    )
    parser.add_argument(
        "--model_backend",
        type=str,
        choices=[
            "hf",
            "vllm",
            "openai",
        ],
        help="Backend library to use for model inference",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory in which HF should download/look for models",
    )

    # Data args
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Exec-CSN json data file"
    )
    parser.add_argument(
        "--data_key",
        type=str,
        default="masked_method_no_test",
        help="Specific key in data to use as code generation prompt. Different keys correspond to different levels of context for the model.",
    )
    parser.add_argument(
        "--use_originial_docstring",
        action="store_true",
        help="Use the original docstring as the instruction",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting point in dataset"
    )
    parser.add_argument(
        "--end_index", type=int, default=-1, help="Ending point in dataset"
    )

    # Output args
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Key of code generations in output json",
    )

    # Sampling args
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of examples to do inference on at once",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="Number of samples to draw per example",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max number of tokens to generate per sample",
    )
    parser.add_argument(
        "--save_freq", type=int, default=100, help="How often to save generations"
    )

    # Misc
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Do not display inference progress bar",
    )

    args = parser.parse_args()

    if args.output_key is None:
        args.output_key = "gen_" + args.data_key

    main(args)
