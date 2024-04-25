"""Prompts for test augmentation"""

###############################################################################
# Structured prompts using reference solution as oracle
###############################################################################

METHOD_ORACLE_TEST_TEMPLATE = '''\
Consider the reference implementation of the `{func_name}` method in the following code:
```python
{code}
```
Here is an example test function for the `{func_name}` method.
```python
{test_func}
```
Assume we have access to an oracle method `{func_name}_oracle`. Your task is to write a test function of the following form that uses the oracle to test the `{func_name}` method:
```python
import copy
def test_{func_name_last}_with_oracle():
    inputs = get_inputs()
    for (test_object, test_input) in inputs: # input_data is a tuple of test inputs (may be a singleton tuple)
        oracle_object = copy.deepcopy(test_object)
        oracle_input = copy.deepcopy(test_input)
        
        try:
            test_result = test_object.{func_name_last}(*test_input)
        except Exception as test_exception:
            test_result = test_exception

        try:
            oracle_result = oracle_object.{func_name_last}_oracle(*oracle_input)
        except Exception as oracle_exception:
            oracle_result = oracle_exception


        if isinstance(test_result, Exception) and not isinstance(oracle_result, Exception):
            # `{func_name_last}` should not have raised an exception
            raise test_result
        elif not isinstance(test_result, Exception) and isinstance(oracle_result, Exception):
            # `{func_name_last}` should have raised an exception
            raise Exception("{func_name_last} should have raised an exception.")
        elif isinstance(test_result, Exception) and isinstance(oracle_result, Exception):
            # make sure both raised the same type of exception
            assert type(test_result) == type(oracle_result)
        else: 
            # make sure both yielded the same results
            assert validate_output(test_object, oracle_object, test_input, oracle_input, test_result, oracle_result)

def get_inputs():
    """
    Creates 7-8 challenging and comprehensive test inputs for the `{func_name}` function.
    `get_inputs` should return a list of 2-tuples, where the first element of each tuple is an instance of the `{func_name_first}` class and the second element of each tuple is a test input to the `{func_name}` method. The test input to the `{func_name}` method should be a (possibly singleton) tuple of method arguments.
    """
    <fill this in>

def validate_output(test_object, oracle_object, test_input, oracle_input, test_output, oracle_output):
    """
    Returns True if `{func_name}` and `{func_name}_oracle` exhibit the same behavior on input `test_input`, otherwise returns False.
    Note that not all arguments to the function may be need to be used to validate correctness.
    The validation logic may take inspiration from the example `test_{func_name_last}` function above.
    If `{func_name}` modifies the input and/or the object in-place, this function should also validate those changes in the input and/or object.
    """
    <fill this in>
```
Please fill in the `get_inputs` and `validate_output` functions in the test code above. Adhere to the following guidelines:
- `get_inputs` should return a list of 7-8 complex inputs to the `{func_name}` method. These inputs should cover both main functionality and edge cases.
- `get_inputs` should create executable test inputs and not return placeholder values.
- `validate_output` should only use the built-in equality operator (`==`) on built-in Python objects (int, float, bool, str, list, dict, set). It should compare non-built-in objects by checking their attributes, or calling custom equality functions, if they exist. Never use the built-in equality operator on anything that is not a built-in Python object.
- Before writing the `get_inputs` and `validate_output` functions, first consider the following questions: What are the input and output types of the `{func_name}` method? Are these built-in types (i.e. int, float, bool, str, list, dict, set)? If not, what is the best way to check equality of these objects?'''

#########################################################################

FUNCTION_ORACLE_TEST_TEMPLATE = '''\
Consider the reference implementation of the `{func_name}` function in the following code:
```python
{code}
```
Here is an example test function for the `{func_name}` function.
```python
{test_func}
```
Assume we have access to an oracle function `{func_name}_oracle`. Your task is to write a test function of the following form that uses the oracle to test the `{func_name}` function:
```python
import copy
def test_{func_name_last}_with_oracle():
    inputs = get_inputs()
    for test_input in inputs: # input_data is a tuple of test inputs (may be a singleton tuple)
        oracle_input = copy.deepcopy(test_input)
        
        try:
            test_result = {func_name_last}(*test_input)
        except Exception as test_exception:
            test_result = test_exception

        try:
            oracle_result = {func_name_last}_oracle(*oracle_input)
        except Exception as oracle_exception:
            oracle_result = oracle_exception

        if isinstance(test_result, Exception) and not isinstance(oracle_result, Exception):
            # `{func_name_last}` should not have raised an exception
            raise test_result
        elif not isinstance(test_result, Exception) and isinstance(oracle_result, Exception):
            # `{func_name_last}` should have raised an exception
            raise Exception("{func_name_last} should have raised an exception.")
        elif isinstance(test_result, Exception) and isinstance(oracle_result, Exception):
            # if both raised an exception, make sure they raised the same type of exception
            assert type(test_result) == type(oracle_result)
        else: 
            # if neither raised an exception, make sure both yielded the same results
            assert validate_output(test_input, oracle_input, test_result, oracle_result)

def get_inputs():
    """
    Creates 7-8 challenging and comprehensive test inputs for the `{func_name}` function.
    `get_inputs` should return a list of tuples, where each tuple is a test input to the `{func_name}` method. The test input to the `{func_name}` method should be a (possibly singleton) tuple of arguments.
    """
    <fill this in>

def validate_output(test_input, oracle_input, test_output, oracle_output):
    """
    Returns True if `{func_name}` and `{func_name}_oracle` exhibit the same behavior on input `test_input`, otherwise returns False.
    Note that not all arguments to the function may be need to be used to validate correctness.
    The validation logic should take inspiration from the example `test_{func_name_last}` function above.
    If `{func_name}` modifies the input in-place, this function should also validate those changes in the input.
    """
    <fill this in>
```
Please fill in the `get_inputs` and `validate_output` functions in the test code above. Adhere to the following guidelines:
- `get_inputs` should return a list of 7-8 complex inputs to the `{func_name}` method. These inputs should cover both main functionality and edge cases.
- `get_inputs` should create executable test inputs and not return placeholder values.
- `validate_output` should only use the built-in equality operator (`==`) on built-in Python objects (int, float, bool, str, list, dict, set). It should compare non-built-in objects by checking their attributes, or calling custom equality functions, if they exist. Never use the built-in equality operator on anything that is not a built-in Python object.
- Before writing the `get_inputs` and `validate_output` functions, first consider the following questions: What are the input and output types of the `{func_name}` method? Are these built-in types (i.e. int, float, bool, str, list, dict, set)? If not, what is the best way to check equality of these objects?'''


###############################################################################
# Simple prompt
###############################################################################

EXTEND_TEST_TEMPLATE = """\
Consider the reference implementation of the `{func_name}` function in the following code:
```python
{code}
```
Here is a simple test function for the `{func_name}` function.
```python
{test_func}
```
Write another test function `test_hard_{func_name_last}` that uses similar test logic to the `test_{func_name_last}` function above, but tests the `{func_name}` function on 4-5 difficult or corner-case inputs. The goal of this function is to unearth bugs that might not have been revealed by the original `test_{func_name_last}` function. IMPORTANT: the reference implementation of the `{func_name}` function above should be able to pass all these tests!"""
