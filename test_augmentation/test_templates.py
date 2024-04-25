METHOD_ORACLE_TEST_CODE = """\
{oracle_code}

{get_inputs}

{validate_output}

import types
import copy
import random
def test_{func_name_last}_with_oracle():
    random.seed(0)
    test_inputs = get_inputs()
    random.seed(0)
    oracle_inputs = get_inputs()
    for test_idx, ((test_object, test_input), (oracle_object, oracle_input)) in enumerate(zip(test_inputs, oracle_inputs)):
        oracle_object.{func_name_last}_oracle = types.MethodType({func_name_last}_oracle, oracle_object)
        
        try:
            random.seed(test_idx)
            test_result = test_object.{func_name_last}(*test_input)
        except Exception as test_exception:
            test_result = test_exception

        try:
            random.seed(test_idx)
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

if __name__ == '__main__':
    print("Running test_{func_name_last}_with_oracle...", end='', flush=True)
    test_{func_name_last}_with_oracle()
    print("Passed!")
"""


################################################################################


FUNCTION_ORACLE_TEST_CODE = """\
{oracle_code}

{get_inputs}

{validate_output}

import random
import copy
def test_{func_name_last}_with_oracle():
    random.seed(0)
    test_inputs = get_inputs()
    random.seed(0)
    oracle_inputs = get_inputs()
    for test_idx, (test_input, oracle_input) in enumerate(zip(test_inputs, oracle_inputs)): # input_data is a tuple of test inputs (may be a singleton tuple)
        try:
            random.seed(test_idx)
            test_result = {func_name_last}(*test_input)
        except Exception as test_exception:
            test_result = test_exception

        try:
            random.seed(test_idx)
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

if __name__ == '__main__':
    print("Running test_{func_name_last}_with_oracle...", end='', flush=True)
    test_{func_name_last}_with_oracle()
    print("Passed!")
"""


################################################################################


EXTEND_TEST_CODE = """\
{test_code}

if __name__ == '__main__':
    print("Running test_hard_{func_name_last}", end='', flush=True)
    test_hard_{func_name_last}()
    print("Passed!")
"""


FULL_CODE = """\
{nontest_code}

{new_test_code}
"""
