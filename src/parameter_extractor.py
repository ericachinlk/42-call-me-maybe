from pydantic import BaseModel
from typing import Any
from src.models import FunctionDefinition
from src.llm_engine import LLMEngine
from src.json_decoder import JSONConstrainedDecoder


class ParameterExtractor(BaseModel):
    llm: LLMEngine
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._decoder = JSONConstrainedDecoder(llm=self.llm)

    def extract(self, fn_def: FunctionDefinition, user_prompt: str) -> dict[str, Any]:
        return self._decoder.decode(fn_def, user_prompt)
