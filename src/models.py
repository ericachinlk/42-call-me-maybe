"""
Data schemas and verification models utilized across
the pipeline processing layer.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, model_validator


class ParameterType(str, Enum):
    """Supported fundamental JSON schema parameter types."""
    string = "string"
    number = "number"
    boolean = "boolean"
    integer = "integer"


class ParameterDefinition(BaseModel):
    """Defines structural criteria tracking expected data configurations.

    Attributes:
        type: The schema-validated primitive type.
    """
    type: ParameterType


class FunctionDefinition(BaseModel):
    """Tracks complete structured functional documentation schemas.

    Attributes:
        name: Name identifier for the function.
        description: Informative prompt text description of behavior.
        parameters: A mapping dictionary matching explicit
            fields with definitions.
        returns: Result output type configuration object.
        full_definition: Cached raw text description capture.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition
    full_definition: str

    @model_validator(mode="before")
    @classmethod
    def capture_raw_string(cls, data: Any) -> Any:
        """
        Capture the raw payload mapping as a string
        before validation transforms occur.

        Args:
            data: Raw dictionary initialization data.

        Returns:
            Any: Modified data dictionary containing
                a populated 'full_definition' field.
        """
        if isinstance(data, dict):
            data["full_definition"] = str(data)
        return data


class TestPrompt(BaseModel):
    """Simple wrapper encapsulation managing text evaluation input queries.

    Attributes:
        prompt: Raw string data source representing user queries.
    """
    prompt: str


class FunctionCallOutput(BaseModel):
    """Final verified model tracking targeted tool execution structures.

    Attributes:
        prompt: Initial origin context user prompt statement.
        name: Validated target function identifier choice.
        parameters: Collected keyword map containing
            extracted parameter arguments.
    """
    prompt: str
    name: str
    parameters: dict[str, Any]
