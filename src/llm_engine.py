from llm_sdk import Small_LLM_Model
from typing import Generator, Any
from pydantic import BaseModel, Field


class LLMEngine(BaseModel):
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    model_config = {"arbitrary_types_allowed": True}

    def get_token_ids(self, text: str) -> Any:
        """Convert text to token IDs."""
        tensor = self.model.encode(text)
        return tensor.tolist()[0]

    def next_multiple_tokens(
            self,
            prompt_message: str,
            previous_tokens: str = '',
            skip: int = 0,
            valid_token_ids: set[int] | list[int] | None = None
    ) -> Generator[str]:
        """Generate tokens with optional logit-level masking."""
        prompt = f"<|im_start|>user\n{prompt_message}<|im_end|>\n" + \
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        tensors = self.model.encode(prompt)
        probabilities = self.model.get_logits_from_input_ids(
            tensors.tolist()[0])

        # Convert to set for O(1) lookup if needed
        if valid_token_ids is not None:
            if not isinstance(valid_token_ids, set):
                valid_set = set(valid_token_ids)
            else:
                valid_set = valid_token_ids
            sorted_indices = sorted(
                (i for i in range(len(probabilities)) if i in valid_set),
                key=probabilities.__getitem__,
                reverse=True
            )
        else:
            sorted_indices = sorted(
                range(len(probabilities)),
                key=probabilities.__getitem__,
                reverse=True
            )

        for idx in sorted_indices[skip:]:
            yield self.model.decode(idx)
