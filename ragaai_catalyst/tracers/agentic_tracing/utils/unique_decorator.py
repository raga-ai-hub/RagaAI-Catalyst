import hashlib
import inspect
import functools
import re
import tokenize
import io
import types

def normalize_source_code(source):
    """
    Advanced normalization of source code that:
    1. Preserves docstrings
    2. Removes comments
    3. Removes extra whitespace
    
    Args:
        source (str): Original source code
    
    Returns:
        str: Normalized source code
    """
    normalized_tokens = []
    
    try:
        token_source = io.StringIO(source).readline
        
        for token_type, token_string, _, _, _ in tokenize.generate_tokens(token_source):
            if token_type == tokenize.STRING:
                normalized_tokens.append(token_string.strip())
            elif token_type in [tokenize.NAME, tokenize.NUMBER, tokenize.OP]:
                normalized_tokens.append(token_string.strip())
    
    except tokenize.TokenError:
        normalized_tokens = re.findall(r'\w+|[^\w\s]', source)
    
    return ''.join(normalized_tokens)

def generate_unique_hash(func, *args, **kwargs):
    """Generate a unique hash based on the original function and its arguments"""
    if inspect.ismethod(func) or inspect.isfunction(func):
        # Get function name and source code
        func_name = func.__name__
        try:
            func_source = inspect.getsource(func)
            normalized_source = normalize_source_code(func_source)
        except (IOError, TypeError):
            normalized_source = ""
        
        # Normalize argument values
        def normalize_arg(arg):
            if isinstance(arg, (str, int, float, bool)):
                return str(arg)
            elif isinstance(arg, (list, tuple, set)):
                return '_'.join(normalize_arg(x) for x in arg)
            elif isinstance(arg, dict):
                return '_'.join(f"{normalize_arg(k)}:{normalize_arg(v)}" 
                              for k, v in sorted(arg.items()))
            elif callable(arg):
                if hasattr(arg, "__name__"):
                    return arg.__name__
                return str(type(arg).__name__)
            else:
                return str(type(arg).__name__)

        # Create normalized strings of arguments
        args_str = '_'.join(normalize_arg(arg) for arg in args)
        kwargs_str = '_'.join(f"{k}:{normalize_arg(v)}" 
                            for k, v in sorted(kwargs.items()))
        
        # Combine all components
        hash_input = f"{func_name}_{normalized_source}_{args_str}_{kwargs_str}"
    
    elif inspect.isclass(func):
        try:
            class_source = inspect.getsource(func)
            normalized_source = normalize_source_code(class_source)
            hash_input = f"{func.__name__}_{normalized_source}"
        except (IOError, TypeError):
            hash_input = f"{func.__name__}_{str(func)}"
    
    else:
        hash_input = str(func)

    hash_obj = hashlib.md5(hash_input.encode('utf-8'))
    return hash_obj.hexdigest()

def generate_unique_hash_simple(func):
    """Generate a unique hash based on the function name and normalized source code.
    Works for both standalone functions and class methods (where self would be passed)."""
    import hashlib
    import inspect
    
    # Handle bound methods (instance methods of classes)
    if hasattr(func, '__self__'):
        # Get the underlying function from the bound method
        func = func.__func__
    

    # Get function name
    func_name = func.__name__
    
    # Get and normalize source code based on type
    try:
        if isinstance(func, (types.FunctionType, types.MethodType)):
            source = inspect.getsource(func)
            # Remove whitespace and normalize line endings
            normalized_source = "\n".join(line.strip() for line in source.splitlines())
        elif inspect.isclass(func):
            source = inspect.getsource(func)
            normalized_source = "\n".join(line.strip() for line in source.splitlines())
        else:
            normalized_source = str(func)
    except (IOError, TypeError):
        normalized_source = str(func)
    
    # Use fixed timestamp for reproducibility
    timestamp = "2025-01-03T18:15:16+05:30"
    
    # Combine components
    hash_input = f"{func_name}_{normalized_source}_{timestamp}"
    
    # Generate MD5 hash
    hash_obj = hashlib.md5(hash_input.encode('utf-8'))
    return hash_obj.hexdigest()

class UniqueIdentifier:
    _instance = None
    _hash_cache = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, salt=None):
        if not hasattr(self, 'salt'):
            self.salt = salt

    def __call__(self, obj):
        if inspect.isclass(obj):
            hash_id = generate_unique_hash(obj)
            setattr(obj, 'hash_id', hash_id)
            return obj
        
        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            # Generate hash based on the original function and its arguments
            if hasattr(args[0], 'original_func'):  # Check if it's a wrapped LLM call
                original_func = args[0].original_func
                func_args = args[1:]  # Skip the original_func argument
                hash_id = generate_unique_hash(original_func, *func_args, **kwargs)
            else:
                hash_id = generate_unique_hash(obj, *args, **kwargs)
            
            # Store hash_id on the wrapper function
            wrapper.hash_id = hash_id
            
            return obj(*args, **kwargs)
        
        # Initialize hash_id
        initial_hash = generate_unique_hash(obj)
        wrapper.hash_id = initial_hash
        
        return wrapper

# Create a single instance to be used across all mixins
mydecorator = UniqueIdentifier()