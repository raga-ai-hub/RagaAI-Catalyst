from unique_decorator import mydecorator
from unique_decorator import generate_unique_hash
import inspect

def print_test_case(case_num, description, expected_behavior, hash1, hash2=None):
    print(f"\n{'='*100}")
    print(f"Test Case #{case_num}: {description}")
    print(f"Expected Behavior: {expected_behavior}")
    print(f"{'='*100}")
    if hash2 is not None:
        print(f"Hash ID 1: {hash1}")
        print(f"Hash ID 2: {hash2}")
        print(f"Hash IDs are {'EQUAL' if hash1 == hash2 else 'DIFFERENT'} (Expected: {expected_behavior})")
    else:
        print(f"Hash ID: {hash1}")
    print(f"{'='*100}\n")

# Test Case 1: Same function with different formatting
# Expected: Same hash_id
@mydecorator
def example_function():
    x = 1
    return x

hash1 = example_function.hash_id

@mydecorator
def example_function():
    # This is a comment
    x     =      1  # Another comment
    return     x    # More spacing

hash2 = example_function.hash_id

print_test_case(1, 
                "Same function with different formatting and comments", 
                "Hash IDs should be EQUAL",
                hash1, hash2)

# Test Case 2: Function with parameters - different argument orders
# Expected: Same hash_id for same arguments in different order
@mydecorator
def function_with_params(a: int, b: int = 10):
    return a + b

result1 = function_with_params(a=2, b=3)
hash1 = function_with_params.hash_id

result2 = function_with_params(b=3, a=2)
hash2 = function_with_params.hash_id

print_test_case(2, 
                "Same function call with different argument order (a=2, b=3 vs b=3, a=2)", 
                "Hash IDs should be EQUAL",
                hash1, hash2)

# Test Case 3: Function with different default value
# Expected: Different hash_id
@mydecorator
def function_with_params(a: int, b: int = 5):  # Different default value
    return a + b

hash3 = function_with_params.hash_id

print_test_case(3, 
                "Same function name but different default parameter value", 
                "Hash IDs should be DIFFERENT",
                hash2, hash3)

# Test Case 4: Class methods with different formatting
# Expected: Same hash_id
@mydecorator
class ExampleClass:
    @mydecorator
    def method1(self):
        x = 1
        return x

hash1 = ExampleClass().method1.hash_id

@mydecorator
class ExampleClass:
    @mydecorator
    def method1(self):
        # Comment here
        x    =    1
        return    x

hash2 = ExampleClass().method1.hash_id

print_test_case(4, 
                "Class method with different formatting", 
                "Hash IDs should be EQUAL",
                hash1, hash2)

# Test Case 5: Functions with different argument types but same content
# Expected: Same hash_id
@mydecorator
def complex_function(a: dict, b: list = [1, 2]):
    return a, b

test_dict1 = {"a": 1, "b": 2}
test_dict2 = {"b": 2, "a": 1}  # Same content, different order
test_list1 = [1, 2, 3]
test_list2 = [1, 2, 3]  # Identical list

result1 = complex_function(test_dict1, test_list1)
hash1 = complex_function.hash_id

result2 = complex_function(test_dict2, test_list2)
hash2 = complex_function.hash_id

print_test_case(5, 
                "Complex function with same content in different order", 
                "Hash IDs should be EQUAL",
                hash1, hash2)

# Test Case 6: Function with docstring - different formatting
# Expected: Same hash_id
@mydecorator
def documented_function(x: int):
    """
    This is a docstring.
    It should be preserved in the hash.
    """
    # This is a comment that should be ignored
    return x * 2  # This comment should also be ignored

hash1 = documented_function.hash_id

@mydecorator
def documented_function(x:int):
    """
    This is a docstring.
    It should be preserved in the hash.
    """
    return x*2

hash2 = documented_function.hash_id

print_test_case(6, 
                "Function with docstring - different formatting", 
                "Hash IDs should be EQUAL",
                hash1, hash2)

# Test Case 7: Different functions with same structure
# Expected: Different hash_id
@mydecorator
def function_a(x):
    return x + 1

@mydecorator
def function_b(x):
    return x + 1

print_test_case(7, 
                "Different function names with same implementation", 
                "Hash IDs should be DIFFERENT",
                function_a.hash_id, function_b.hash_id)

# Test Case 8: Same function with different argument values
# Expected: Different hash_id
result1 = function_with_params(a=1, b=2)
hash1 = function_with_params.hash_id

result2 = function_with_params(a=3, b=4)
hash2 = function_with_params.hash_id

print_test_case(8, 
                "Same function with different argument values", 
                "Hash IDs should be DIFFERENT",
                hash1, hash2)