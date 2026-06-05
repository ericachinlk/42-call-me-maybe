from typing import Any
import json
import re
from pydantic import BaseModel, PrivateAttr
from src.models import FunctionDefinition, ParameterType
from src.llm_engine import LLMEngine


class ParameterExtractorConstrained(BaseModel):
    llm: LLMEngine
    _token_to_id: dict[str, int] = PrivateAttr()
    _id_to_token: dict[int, str] = PrivateAttr()
    _number_tokens: set[int] = PrivateAttr()
    _boolean_tokens: set[int] = PrivateAttr()
    _letter_tokens: set[int] = PrivateAttr()
    _digit_tokens: set[int] = PrivateAttr()
    _whitespace_tokens: set[int] = PrivateAttr()
    _all_tokens: set[int] = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        vocab_path = self.llm.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)

        self._token_to_id = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        self._number_tokens = set()
        self._boolean_tokens = set()
        self._letter_tokens = set()
        self._digit_tokens = set()
        self._whitespace_tokens = set()
        self._all_tokens = set(vocab.values())

        for token_str, token_id in vocab.items():
            clean_token = token_str.replace("Ġ", "").strip().lower()

            # Skip empty or special tokens
            if not clean_token or "eos" in clean_token or "end" in clean_token:
                continue

            # Categorize by token type
            if clean_token in ["true", "false"]:
                self._boolean_tokens.add(token_id)
            elif self._is_pure_number(clean_token):
                self._number_tokens.add(token_id)
            elif clean_token in [" ", "\n", "\r", "\t"] or (len(clean_token) <= 2 and all(c in " \n\r\t" for c in token_str)):
                self._whitespace_tokens.add(token_id)
            elif any(c.isdigit() for c in clean_token) and all(c in "0123456789.-" for c in clean_token):
                self._digit_tokens.add(token_id)
            elif any(c.isalpha() for c in clean_token):
                self._letter_tokens.add(token_id)

    def _is_pure_number(self, t: str) -> bool:
        """Check if token is a pure number (no decimal, just digits)"""
        if not t:
            return False
        return all(c in "0123456789.-" for c in t) and any(c.isdigit() for c in t)

    def extract(
        self,
        fn_def: FunctionDefinition,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Extract parameters using constrained decoding"""
        result: dict[str, Any] = {}
        for param_name, param_def in fn_def.parameters.items():
            extracted_value = self._extract_parameter(
                user_prompt=user_prompt,
                param_name=param_name,
                param_type=param_def.type,
            )
            result[param_name] = extracted_value
        return result

    def _extract_parameter(
        self,
        user_prompt: str,
        param_name: str,
        param_type: ParameterType,
    ) -> Any:
        """Extract a single parameter with schema-aware constrained generation"""
        
        # Build the extraction prompt
        extraction_prompt = (
            "Extract the value exactly from the user request. Do not calculate or invent.\n"
            f"User request: {user_prompt}\n"
            f"Parameter name: {param_name}\n"
            f"Parameter type: {param_type.value}\n"
            "Value: "
        )

        input_ids = self.llm.encode(extraction_prompt)
        generated_tokens: list[int] = []

        # Determine generation limits based on type
        max_tokens = self._get_max_tokens(param_type)
        allowed_tokens = self._get_allowed_tokens_for_type(param_type)

        for step in range(max_tokens):
            # Get logits from the model
            logits = self.llm.logits(input_ids + generated_tokens)

            # Apply schema-aware masking
            masked_logits = self._mask_logits(logits, allowed_tokens, param_type, generated_tokens)

            # Select the highest-scoring valid token
            next_token = max(range(len(masked_logits)), key=lambda x: masked_logits[x])

            # Check if we hit a stopping condition
            if masked_logits[next_token] == float("-inf"):
                break

            # Decode and check stopping conditions
            token_str = self.llm.decode([next_token])
            
            if self._should_stop_before_adding(token_str, param_type):
                break

            generated_tokens.append(next_token)
            
            if self._should_stop_after_adding(generated_tokens, param_type):
                break

        # Decode and parse the result
        extracted_text = self.llm.decode(generated_tokens).strip()
        return self._parse_value(extracted_text, param_type)

    def _get_max_tokens(self, param_type: ParameterType) -> int:
        """Determine maximum tokens to generate based on type"""
        if param_type == ParameterType.number:
            return 4  # e.g., "-123.5" ~ 2-4 tokens
        elif param_type == ParameterType.boolean:
            return 2  # "true" or "false" ~ 1-2 tokens
        else:  # string
            return 8  # reasonable for names and short strings

    def _get_allowed_tokens_for_type(self, param_type: ParameterType) -> set[int]:
        """Get the set of allowed tokens based on parameter type"""
        if param_type == ParameterType.number:
            # Allow both number tokens (multi-digit) and digit tokens (single)
            return self._digit_tokens.union(self._number_tokens)

        elif param_type == ParameterType.boolean:
            # Allow only true/false tokens
            return self._boolean_tokens

        else:  # string
            # Allow letters and whitespace, but NOT numbers
            return self._letter_tokens.union(self._whitespace_tokens)

    def _mask_logits(
        self,
        logits: list[float],
        allowed_tokens: set[int],
        param_type: ParameterType,
        generated_tokens: list[int],
    ) -> list[float]:
        """Apply schema-aware masking to logits"""
        masked = [float("-inf")] * len(logits)

        if not allowed_tokens:
            return masked

        # Apply base type constraints
        for token_id in allowed_tokens:
            if token_id < len(masked):
                masked[token_id] = logits[token_id]

        return masked

    def _should_stop_before_adding(self, token_str: str, param_type: ParameterType) -> bool:
        """Check if we should stop before adding this token (structural barriers)"""
        # Stop on structural markers
        if any(c in token_str for c in ["\n", "\r", "}", "]", ","]):
            return True
        return False

    def _should_stop_after_adding(self, generated_tokens: list[int], param_type: ParameterType) -> bool:
        """Check if we should stop after adding the token"""
        current_text = self.llm.decode(generated_tokens).strip()
        
        if not current_text:
            return False

        # For numbers, stop when we have a complete number
        if param_type == ParameterType.number:
            if re.match(r"^-?\d+(\.\d+)?$", current_text) and not current_text.endswith('.'):
                return True

        # For booleans, stop after getting true or false
        if param_type == ParameterType.boolean:
            if current_text.lower() in ["true", "false"]:
                return True

        # For strings, stop on whitespace after having content
        if param_type == ParameterType.string:
            if current_text and current_text[-1].isspace():
                return True

        return False

    def _should_stop_generation(
        self,
        token_str: str,
        param_type: ParameterType,
        generated_tokens: list[int],
    ) -> bool:
        """Determine if generation should stop (DEPRECATED - use the two separate functions)"""
        
        # Stop on structural markers
        if any(c in token_str for c in ["\n", "\r", "}", "]", ","]):
            return True

        # For numbers, stop when we have a complete number
        if param_type == ParameterType.number:
            current_text = self.llm.decode(generated_tokens).strip()
            # Check if it's a valid complete number
            if re.match(r"^-?\d+(\.\d+)?$", current_text) and not current_text.endswith('.'):
                return True

        # For booleans, stop after getting true or false
        if param_type == ParameterType.boolean:
            current_text = self.llm.decode(generated_tokens).strip().lower()
            if current_text in ["true", "false"]:
                return True

        # For strings, stop on common delimiters
        if param_type == ParameterType.string:
            if token_str.strip() == "":  # whitespace
                return len(generated_tokens) > 0  # stop if we already have content

        return False

    def _parse_value(self, text: str, param_type: ParameterType) -> Any:
        """Parse extracted text into the correct Python type"""
        text = text.strip()

        if param_type == ParameterType.number:
            # Extract the first valid number
            match = re.search(r"-?\d+(\.\d+)?", text)
            return float(match.group()) if match else 0.0

        elif param_type == ParameterType.boolean:
            # Check if text contains "true"
            return "true" in text.lower()

        else:  # string
            # Return the text as-is, removing quotes if present
            return text.strip().strip('"').strip("'")
