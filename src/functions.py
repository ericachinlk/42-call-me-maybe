import json

def load_functions(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_function_map(functions):
    return {fn["name"]: fn for fn in functions}


