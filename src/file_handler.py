import json
from typing import Any
from src.models import FunctionDefinition, TestPrompt, FunctionCallOutput
from pydantic import ValidationError


class PipelineError(Exception):
    pass


def format_validation_errors(e: ValidationError) -> str:
    return "\n".join(
        f"  - {'.'.join(map(str, err['loc']))}: {err['msg']}"
        for err in e.errors())


def load_functions(path: str) -> list[FunctionDefinition]:
    result: list[FunctionDefinition] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise PipelineError(
            f"The functions definition file was not found at: {path}")
    except json.JSONDecodeError as e:
        raise PipelineError(
            f"The functions definition file is not valid JSON: {e}")

    if not raw:
        raise PipelineError("Function definition file is empty.")
    if not isinstance(raw, list):
        raise PipelineError(
            "Function definition file must contain a JSON array.")

    try:
        for item in raw:
            fn_def = FunctionDefinition.model_validate(item)
            result.append(fn_def)
    except ValidationError as e:
        raise PipelineError(
            "Invalid function definition file:\n"
            + format_validation_errors(e))

    return result


def load_input(path: str) -> list[str]:
    result: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise PipelineError(f"The input file was not found at: {path}")
    except json.JSONDecodeError as e:
        raise PipelineError(f"The input file is not valid JSON: {e}")

    if not raw:
        raise PipelineError("Input file is empty.")
    if not isinstance(raw, list):
        raise PipelineError("Input file must contain a JSON array.")

    try:
        for item in raw:
            test = TestPrompt.model_validate(item)
            result.append(test.prompt)
    except ValidationError as e:
        raise PipelineError(
            "Invalid input file:\n" + format_validation_errors(e))

    return result


def save_output(output_file: str, results: list[dict[str, Any]]) -> None:
    validated_res: list[FunctionCallOutput] = []
    try:
        for item in results:
            output = FunctionCallOutput.model_validate(item)
            validated_res.append(output)
    except ValidationError as e:
        raise PipelineError("Invalid output:\n" + format_validation_errors(e))

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [item.model_dump() for item in validated_res],
                f,
                indent=2,
                ensure_ascii=False
            )
    except OSError as e:
        raise PipelineError(
            f"Unable to write to output file at {output_file}: {e}")
