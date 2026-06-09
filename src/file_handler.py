"""
Handles I/O operations including loading inputs,
definitions, and saving pipeline outputs.
"""

import json
from typing import Any
from src.models import FunctionDefinition, TestPrompt, FunctionCallOutput
from src.validator import PipelineError
from pydantic import ValidationError


def format_validation_errors(e: ValidationError) -> str:
    """Format Pydantic ValidationError instances into a readable string list.

    Args:
        e: The Pydantic ValidationError exception to parse.

    Returns:
        str: A newline-separated string showing each
            error location and message.
    """
    return "\n".join(
        f"  - {'.'.join(map(str, err['loc']))}: {err['msg']}"
        for err in e.errors())


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definition configurations from a JSON file.

    Args:
        path: The file path to the function definition JSON file.

    Returns:
        list[FunctionDefinition]: A list of validated
            FunctionDefinition models.

    Raises:
        PipelineError: If the file is missing, contains corrupted JSON,
            is empty, is not structured as an array,
            or fails Pydantic schema validation.
    """
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


def load_input(path: str) -> list[TestPrompt]:
    """Load and validate test prompt strings from a JSON target payload file.

    Args:
        path: The file path to the JSON evaluation file.

    Returns:
        list[TestPrompt]: A list of parsed and validated TestPrompt models.

    Raises:
        PipelineError: If the file is missing, contains corrupted JSON,
            is empty, is not an array, or fails internal
            Pydantic schema verification.
    """
    result: list[TestPrompt] = []
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
            result.append(test)
    except ValidationError as e:
        raise PipelineError(
            "Invalid input file:\n" + format_validation_errors(e))

    return result


def save_output(output_file: str, results: list[dict[str, Any]]) -> None:
    """Validate pipeline extraction results and save them out to a JSON file.

    Args:
        output_file: Target file path destination for the write transaction.
        results: Raw dictionary arrays capturing extracted prompt mappings.

    Raises:
        PipelineError: If output elements break schema requirements or if an
            operating system issue blocks file writing.
    """
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
