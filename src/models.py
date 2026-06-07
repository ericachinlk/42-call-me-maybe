from enum import Enum
from typing import Any
from pydantic import BaseModel, model_validator


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
    full_definition: str

    @model_validator(mode="before")
    @classmethod
    def capture_raw_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["full_definition"] = str(data) 
        return data

class TestPrompt(BaseModel):
    prompt: str


class FunctionCallOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
