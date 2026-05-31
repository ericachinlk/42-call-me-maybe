import json
from src.models import FunctionDefinition


def load_functions(path: str) -> list[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [FunctionDefinition(**item) for item in raw]