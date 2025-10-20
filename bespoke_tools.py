from pathlib import Path 
import json
import time 
from datetime import datetime 
from functools import wraps

def safe_strip(string: str) -> str:
    if string == None:
        return 
    return string.strip()

def write_dict_to_json(dictionary: dict, write_path: Path,print_message=True):
    with open(write_path, 'w') as file:
        json.dump(dictionary, file,indent=4, separators=(", ", ": "))
    
    if print_message:
        print(f"Successfully wrote json to file {write_path}.")

def load_json(file_path):
    with open(str(file_path)) as file:
        dict = json.load(file)
        return dict

def get_first_key_from_value(dictionary, value_to_find):
    for key, value in dictionary.items():
        if value == value_to_find:
            return key
    return None  

def time_a_function(return_time=False,should_print_message=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            if should_print_message:
                print(f"Function '{func.__name__}' took {elapsed:.4f} seconds")
            if return_time:
                return result, elapsed
            return result
        return wrapper
    return decorator

def clear_directory_shallow(directory: Path):
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()

def print_divider():
    print("=========================================================")

def get_current_time() -> str:
    rn = datetime.now().isoformat()

    return rn

def coloured_print(obj, R: int, G: int, B: int):

    print(f"\033[38;2;{R};{G};{B}m{obj}\033[0m")


    