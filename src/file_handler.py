import json
from enum import Enum
from typing import Any
from pydantic import BaseModel


class ParameterType(str, Enum):
    string = "string"
    number = "number"
    boolean = "boolean"


class ParameterDefinition(BaseModel):
    type: ParameterType


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class TestPrompt(BaseModel):
    prompt: str


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


def load_functions(path: str) -> list[FunctionDefinition]:
    result: list[FunctionDefinition] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The functions definition file was not found at: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"The functions definition file is not valid JSON: {e}")

    for item in raw:
        fn_def = FunctionDefinition.model_validate(item)
        result.append(fn_def)

    return result


def load_input(path: str) -> list[str]:
    result: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The input file was not found at: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"The input file is not valid JSON: {e}")

    for item in raw:
        test = TestPrompt.model_validate(item)
        result.append(test.prompt)

    return result


def save_output(output_file: str, results: list[dict[str, Any]]) -> None:
    validated_res: list[FunctionCallOutput] = []
    for item in results:
        output = FunctionCallOutput.model_validate(item)
        validated_res.append(output)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [item.model_dump() for item in validated_res],
                f,
                indent=2,
                ensure_ascii=False
            )
    except OSError:
        raise OSError(f"Unable to write to output file at: {output_file}")
