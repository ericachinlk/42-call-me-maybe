from llm_sdk import Small_LLM_Model
from typing import Any, Generator
from pydantic import BaseModel, Field


class LLMEngine(BaseModel):
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    model_config = {"arbitrary_types_allowed": True}

    def predict_token(
            self,
            prompt_message: str,
            previous_tokens: str = '',
            skip: int = 0
    ) -> Any:
        prompt = f"<|im_start|>user\n{prompt_message}<|im_end|>\n" + \
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        tensors = self.model.encode(prompt)
        probabilities = self.model.get_logits_from_input_ids(tensors.tolist()[0])
        sorted_tokens = sorted(probabilities, reverse=True)
        token = probabilities.index(sorted_tokens[skip])
        return self.model.decode(token)

    def predict_multiple_tokens(
            self,
            prompt_message: str,
            previous_tokens: str = '',
            skip: int = 0
    ) -> Generator[str]:
        prompt = f"<|im_start|>user\n{prompt_message}<|im_end|>\n" + \
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        tensors = self.model.encode(prompt)
        probabilities = self.model.get_logits_from_input_ids(tensors.tolist()[0])
        sorted_tokens = sorted(probabilities, reverse=True)

        while True:
            yield self.model.decode(probabilities.index(sorted_tokens[skip]))
            skip += 1


# class LLMEngine(BaseModel):
#     model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
#     model_config = {"arbitrary_types_allowed": True}

#     def encode(self, text: str) -> Any:
#         return self.model.encode(text)[0].tolist()

#     def decode(self, tokens: list[int]) -> Any:
#         return self.model.decode(tokens)

#     def logits(self, tokens: list[int]) -> Any:
#         return self.model.get_logits_from_input_ids(tokens)
