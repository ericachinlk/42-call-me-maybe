from src.models import ParameterType, FunctionDefinition
from typing import Any


class PipelineError(Exception):
    pass


def validate_parameters(
        fn_def: FunctionDefinition,
        params: dict[str, Any]
) -> None:
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
