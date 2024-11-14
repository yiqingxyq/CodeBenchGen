EXP_ID="random50"

# call repo clone and function extraction
PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_docstring" --disable_no_docstring
PYTHONPATH=./ python get_functions.py --exp_id="${EXP_ID}_no_docstring"

# sample functions
python sample_functions.py --exp_id="${EXP_ID}" --func_num 10

python get_context.py --exp_id="${EXP_ID}" --context_type sliced --max_context_size 6000