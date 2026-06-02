from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from pydantic import BaseModel, Field


class LLMEngine(BaseModel):
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    model_config = {"arbitrary_types_allowed": True}

    def encode(self, text: str) -> Any:
        return self.model.encode(text)[0].tolist()

    def decode(self, tokens: list[int]) -> Any:
        return self.model.decode(tokens)

    def logits(self, tokens: list[int]) -> Any:
        return self.model.get_logits_from_input_ids(tokens)
