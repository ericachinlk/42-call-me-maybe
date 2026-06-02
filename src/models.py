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
