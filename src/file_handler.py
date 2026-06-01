import json
from src.models import FunctionDefinition


def load_functions(path: str) -> list[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [FunctionDefinition(**item) for item in raw]


def load_input(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [item["prompt"] for item in raw]


def save_output(output_file: str, results: list[dict]) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )
