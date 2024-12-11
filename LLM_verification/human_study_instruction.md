### Human Study v3

We use LLM to construct evaluation examples based on real github code. You are given the ORIGINAL FUNCTION and the evaluation script containing the REVISED FUNCTION. Your task is to judge whether the REVISED FUNCTION has the same functionality as the original one.

The data is stored in `human_study_v3_data.json`, which is a list of dicts. Each dict is in the following format:
```
{
    "function_name": the function name,
    "orig_function": the ORIGINAL FUNCTION,
    "new_script":    the evaluation script we generate, which contains the REVISED FUNCTION,
}
```

You'll need to add an additional field into each dict called `answer`, which could have the value: `same`, `yes`, `minor`, and `no`.
- If the REVISED FUNCTION is exactly the same as the ORINIGAL FUNCTION, answer `same`.
- If the functionality of the REVISED FUNCTION is the same as the ORIGINAL FUNCTION, answer `yes`.
- If the two functions' functionalities only have minor differences, answer `minor`. Minor differences may include one or more of the following:
    (1) sanity checks of the input arguments, 
    (2) default arguments of function calls, such as revising `run(command, capture_output=True)` to `run(command, shell=True, text=True)`,
    (3) difference in print statements or logging information.
- Otherwise, if the functionality of the REVISED FUNCTION has any major differences that do not belong to any of the above 3 categories, answer `no`.
