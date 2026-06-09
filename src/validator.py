"""
Validates execution parameter types against
function definition specifications.
"""

from src.models import ParameterType, FunctionDefinition
from typing import Any


class PipelineError(Exception):
    """
    Custom exception raised for operational and
    logical failures in the pipeline.
    """
    pass


def validate_parameters(
        fn_def: FunctionDefinition,
        params: dict[str, Any]
) -> None:
    """
    Validate that parameters extracted by the
    LLM match expected type definitions.

    Args:
        fn_def: Function definition configuration model
            containing type definitions.
        params: Map containing extracted keys mapped
            against generated values.

    Raises:
        PipelineError: If an argument data type does not
            conform to its configured ParameterType restriction.
    """
    for name, value in params.items():
        expected = fn_def.parameters[name].type

        if expected == ParameterType.number:
            if not isinstance(value, (int, float)):
                raise PipelineError(
                    f"Parameter '{name}' must be number, "
                    f"got {type(value).__name__}"
                )

        elif expected == ParameterType.string:
            if not isinstance(value, str):
                raise PipelineError(
                    f"Parameter '{name}' must be string, "
                    f"got {type(value).__name__}"
                )

        elif expected == ParameterType.boolean:
            if not isinstance(value, bool):
                raise PipelineError(
                    f"Parameter '{name}' must be boolean, "
                    f"got {type(value).__name__}"
                )
