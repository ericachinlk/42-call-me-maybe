"""
Implements interaction layers and logic tuning
for the core language model.
"""

from llm_sdk import Small_LLM_Model
from typing import Generator, Any
from pydantic import BaseModel, Field


class LLMEngine(BaseModel):
    """
    Execution engine interface interacting with
    underlying language model architectures.

    Attributes:
        model: Tokenizer and logic tensor evaluation instance.
    """
    model: Small_LLM_Model = Field(default_factory=Small_LLM_Model)
    model_config = {"arbitrary_types_allowed": True}

    def get_token_ids(self, text: str) -> Any:
        """
        Convert a target text segment or character into
        its corresponding token IDs.

        Args:
            text: String target to be encoded.

        Returns:
            list[int]: A list containing the generated integer token IDs.
        """
        tensor = self.model.encode(text)
        return tensor.tolist()[0]

    def next_multiple_tokens(
            self,
            prompt_message: str,
            previous_tokens: str = '',
            skip: int = 0,
            valid_token_ids: set[int] | list[int] | None = None
    ) -> Generator[str]:
        """Generate tokens with true logit masking.

        Invalid tokens have logits set to -inf before sampling,
        preventing them from being selected at the
        probability distribution level.

        Args:
            prompt_message: Base context instructions for the LLM.
            previous_tokens: Accumulated assistant history tokens.
            skip: Integer offset mapping to shift decoding indexes.
            valid_token_ids:
                Collection of integers representing allowed token IDs.

        Yields:
            str: Individual decoded character or token fragments.
        """
        prompt = f"<|im_start|>user\n{prompt_message}<|im_end|>\n" + \
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{previous_tokens}"
        tensors = self.model.encode(prompt)
        logits = self.model.get_logits_from_input_ids(tensors.tolist()[0])

        # Logit masking: Set invalid logits to -inf
        if valid_token_ids is not None:
            if not isinstance(valid_token_ids, set):
                valid_set = set(valid_token_ids)
            else:
                valid_set = valid_token_ids

            # Mask invalid tokens
            for i in range(len(logits)):
                if i not in valid_set:
                    logits[i] = float('-inf')

        # Sort by logits (highest first)
        sorted_indices = sorted(
            range(len(logits)),
            key=lambda i: logits[i],
            reverse=True
        )

        # Filter out -inf values (invalid tokens)
        valid_indices = [
            i for i in sorted_indices if logits[i] != float('-inf')]

        for idx in valid_indices[skip:]:
            yield self.model.decode(idx)
