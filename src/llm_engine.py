from llm_sdk import Small_LLM_Model
from typing import Generator
from pydantic import BaseModel, Field


class LLMEngine(BaseModel):
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    model_config = {"arbitrary_types_allowed": True}

    def next_multiple_tokens(
            self,
            prompt_message: str,
            previous_tokens: str = '',
            skip: int = 0
    ) -> Generator[str]:
        prompt = f"<|im_start|>user\n{prompt_message}<|im_end|>\n" + \
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        tensors = self.model.encode(prompt)
        probabilities = self.model.get_logits_from_input_ids(
            tensors.tolist()[0])
        sorted_indices = sorted(
            range(len(probabilities)),
            key=probabilities.__getitem__, reverse=True)

        for idx in sorted_indices[skip:]:
            yield self.model.decode(idx)
