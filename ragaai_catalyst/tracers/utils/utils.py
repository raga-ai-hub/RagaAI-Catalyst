import hashlib
import json
import unicodedata


def normalize_string(input_str):
    # Normalize Unicode string and make it case-insensitive
    return unicodedata.normalize("NFKC", input_str).lower()


def get_unique_key(input_data):
    """
    Generate a unique key based on the input data.

    Args:
        input_data (Union[dict, str]): The input data to generate the unique key from. It can be either a dictionary or a string.

    Returns:
        str: The unique key generated from the input data.

    Raises:
        ValueError: If the input data is neither a dictionary nor a string.

    Processing Steps:
        1. If the input data is a dictionary, process it to ensure that the keys are case-sensitive but the values are case-insensitive.
        2. Convert the processed dictionary to a canonical JSON representation.
        3. If the input data is a string, normalize and make it case-insensitive.
        4. Calculate the SHA-256 hash of the canonical JSON representation.
        5. Return the unique key generated from the hash.

    Note:
        - The keys in the input dictionary are case-sensitive.
        - The values in the input dictionary are case-insensitive if they are strings.
        - The input string is normalized and made case-insensitive.


        # Example usage:
        data1 = {'a': "Hello", 'b': "World"}
        data2 = {'a': "hello", 'b': "world"}
        string1 = "Hello World"
        string2 = "hello world"
    """
    if isinstance(input_data, dict):
        # Process dictionary to ensure keys are case-sensitive but values are case-insensitive
        processed_dict = {
            k: normalize_string(v) if isinstance(v, str) else v
            for k, v in input_data.items()
        }
        # Convert the dictionary to a canonical JSON representation
        canonical_json = json.dumps(processed_dict, sort_keys=True)
    elif isinstance(input_data, str):
        # Normalize and make the string case-insensitive
        canonical_json = normalize_string(input_data)
    else:
        # If input is neither a dictionary nor a string, raise an error
        raise ValueError("Input must be a dictionary or a string")

    # Calculate the SHA-256 hash of the canonical JSON representation
    hash_object = hashlib.sha256(canonical_json.encode())
    unique_key = hash_object.hexdigest()

    return unique_key
