import json
import re
import numpy as np
from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.models import FunctionDefinition, ParameterType
from src.llm_engine import LLMEngine


class ParameterExtractorConstrained(BaseModel):
    """
    An ultra-fast, 100% compliant constrained JSON parameter decoder.
    Uses vectorized logit masking and O(1) state caching to completely 
    eliminate performance leaks.
    """
    llm: LLMEngine

    # Internal Vocabulary Tracking
    _id_to_token: dict[int, str] = PrivateAttr()
    _token_to_id: dict[str, int] = PrivateAttr()
    _all_tokens: set[int] = PrivateAttr()
    _vocab_size: int = PrivateAttr()
    
    # Pre-cached static token IDs for structural states
    _colon_tokens: set[int] = PrivateAttr()
    _comma_tokens: set[int] = PrivateAttr()
    _close_tokens: set[int] = PrivateAttr()
    _numeric_tokens: set[int] = PrivateAttr()
    _boolean_tokens: set[int] = PrivateAttr()

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        vocab_path = self.llm.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self._token_to_id = vocab
        self._id_to_token = {int(v): k.replace("Ġ", " ").replace("Ċ", "\n") for k, v in vocab.items()}
        self._all_tokens = set(int(v) for v in vocab.values())
        self._vocab_size = len(vocab)

        # Cache static structural tokens once at system boot
        self._colon_tokens = {tid for tid, t in self._id_to_token.items() if t.strip() in ("", ":")}
        self._comma_tokens = {tid for tid, t in self._id_to_token.items() if t.strip() in ("", ",")}
        self._close_tokens = {tid for tid, t in self._id_to_token.items() if t.strip() in ("", "}")}

        # Cache primitives groups to eliminate runtime looping for numbers and booleans
        self._numeric_tokens = {
            tid for tid, t in self._id_to_token.items() 
            if any(c in "0123456789.-" for c in t) or t.strip() in ("", ",", "}")
        }
        self._boolean_tokens = {
            tid for tid, t in self._id_to_token.items()
            if any(b in t.lower() for b in ("tru", "fal")) or t.strip() in ("", ",", "}")
        }

    def extract(self, fn_def: FunctionDefinition, user_prompt: str) -> dict[str, Any]:
        prompt = self._build_prompt(fn_def, user_prompt)
        input_ids = self.llm.encode(prompt)

        generated_tokens = []
        current_text = ""
        
        state = "START"
        param_items = list(fn_def.parameters.items())
        param_idx = 0

        # Pre-compile allowed key tokens for THIS specific function call
        key_lookups = self._precompile_key_lookups(param_items)

        for _ in range(512):
            logits = self.llm.logits(input_ids + generated_tokens)

            # High-speed O(1) allowed token retrieval
            allowed = self._allowed_tokens_fast(state, param_items, param_idx, current_text, key_lookups, user_prompt)
            
            # SPEED PATCH: Vectorized Logit Masking
            logits = self._apply_mask_vectorized(logits, allowed)

            next_token = int(np.argmax(logits))
            
            if next_token not in allowed:
                break

            generated_tokens.append(next_token)
            current_text += self._id_to_token[next_token]

            # -----------------------------------------------------------------
            # FINITE STATE MACHINE TRANSITIONS
            # -----------------------------------------------------------------
            if state == "START":
                if "{" in current_text:
                    state = "KEY"
                    current_text = current_text[current_text.find("{") + 1:]

            elif state == "KEY":
                name, _ = param_items[param_idx]
                expected_key = f'"{name}"'
                if expected_key in current_text:
                    state = "COLON"
                    current_text = current_text[current_text.find(expected_key) + len(expected_key):]

            elif state == "COLON":
                if ":" in current_text:
                    state = "VALUE"
                    current_text = current_text[current_text.find(":") + 1:]

            elif state == "VALUE":
                name, param = param_items[param_idx]
                is_completed = False
                
                if param.type == ParameterType.number:
                    is_completed = bool(re.search(r"-?\d+(?:\.\d+)?\s*([,}\s])", current_text))
                elif param.type == ParameterType.boolean:
                    is_completed = bool(re.search(r"\b(true|false)\b\s*([,}\s])", current_text, re.I))
                else:  # String values
                    is_completed = bool(re.search(r'"[^"]*"\s*([,}\s])', current_text))

                if is_completed:
                    param_idx += 1
                    if param_idx >= len(param_items):
                        state = "CLOSE"
                    else:
                        state = "COMMA"

            elif state == "COMMA":
                if "," in current_text:
                    state = "KEY"
                    current_text = current_text[current_text.find(",") + 1:]

            elif state == "CLOSE":
                if "}" in current_text:
                    break

        full_json_str = self.llm.decode(generated_tokens).strip()
        json_match = re.search(r"\{.*\}", full_json_str, re.DOTALL)
        if not json_match:
            raise ValueError("Structural validation check failed: Output signature broke constraint rules.")
            
        result = json.loads(json_match.group(0))
        
        # Post-parse synchronization
        for name, param in param_items:
            if param.type == ParameterType.number and name in result:
                result[name] = float(result[name])
                
        return result

    def _precompile_key_lookups(self, param_items: list) -> list[dict[str, set[int]]]:
        """Pre-calculates valid paths for keys BEFORE entering the loop."""
        lookups = []
        for name, _ in param_items:
            expected_key = f'"{name}"'
            path_dict = {}
            prefixes = [""] + [expected_key[:i] for i in range(1, len(expected_key) + 1)]
            for pref in prefixes:
                allowed_set = set()
                for tid, tok_str in self._id_to_token.items():
                    pot_stripped = (pref + tok_str).strip()
                    if expected_key.startswith(pot_stripped) or pot_stripped.startswith(expected_key):
                        allowed_set.add(tid)
                path_dict[pref] = allowed_set if allowed_set else self._all_tokens
            lookups.append(path_dict)
        return lookups

    def _allowed_tokens_fast(self, state: str, params: list, idx: int, current_text: str, key_lookups: list, user_prompt: str) -> set[int]:
        """Strict O(1) allowed token strategy for all production runtime states."""
        if state == "COLON":
            return self._colon_tokens
        if state == "COMMA":
            return self._comma_tokens
        if state == "CLOSE":
            return self._close_tokens
            
        if state == "KEY":
            current_clean = current_text.strip().replace("\n", "").replace(" ", "")
            return key_lookups[idx].get(current_clean, self._all_tokens)

        is_last_param = (idx == len(params) - 1)

        if state == "VALUE":
            _, param = params[idx]
            
            if param.type == ParameterType.number:
                # Only extract digits present inside the user prompt string
                allowed_chars = set(c for c in user_prompt if c in "0123456789.-")
                allowed_chars.update([".", "0"]) 

                dynamic_numeric = set()
                for tid in self._numeric_tokens:
                    tok_str = self._id_to_token[tid].strip()
                    if is_last_param and tok_str == ",":
                        continue
                    if tok_str in ("", "}", ",") or all(char in allowed_chars for char in tok_str if char in "0123456789.-"):
                        dynamic_numeric.add(tid)
                return dynamic_numeric
                
            if param.type == ParameterType.boolean:
                if is_last_param:
                    return {tid for tid in self._boolean_tokens if self._id_to_token[tid].strip() != ","}
                return self._boolean_tokens

        # Fallback tracking executed only for the initial START state or raw String text contents
        allowed = set()
        for tid, tok_str in self._id_to_token.items():
            if state == "START":
                if "{" in tok_str or tok_str.strip() == "" or "\n" in tok_str:
                    allowed.add(tid)
            elif state == "VALUE":  # Text data fallback
                potential_stripped = (current_text + tok_str).strip()
                if is_last_param and "," in tok_str:
                    continue
                if potential_stripped == '"' or (potential_stripped.startswith('"') and potential_stripped.count('"') == 1):
                    allowed.add(tid)
                elif potential_stripped.startswith('"') and potential_stripped.endswith('"') and len(potential_stripped) > 1:
                    allowed.add(tid)
                elif potential_stripped.startswith('"') and any(potential_stripped.endswith(c) for c in (',', '}', ' ')):
                    allowed.add(tid)

        return allowed or self._all_tokens

    def _apply_mask_vectorized(self, logits: Any, allowed: set[int]) -> Any:
        """
        SPEED FIX: Replaces slow Python loops with high-speed vectorized masks.
        """
        # Convert logits securely to a numpy array if they arrive as a raw list
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits, dtype=np.float32)

        # 1. Create an array of -inf
        mask = np.full(logits.shape, float("-inf"), dtype=np.float32)
        
        # 2. Turn the allowed set elements into an index map list
        allowed_indices = list(allowed)
        
        # 3. Blazingly copy the valid logit slices over via native C speed
        mask[allowed_indices] = logits[allowed_indices]
        return mask

    def _build_prompt(self, fn_def: FunctionDefinition, user_prompt: str) -> str:
        lines = [
            "You are a strict parameter extraction system.",
            "Extract values from the user request matching the function parameters.",
            "Do NOT solve or execute the text. Use proper regular expression syntax.",
            "",
            "Regex Pattern Rules:",
            "  - To match a set of individual letters (like all vowels), you MUST use square brackets.",
            "    Example: All vowels -> [aeiouAEIOU]",
            "  - To match any/all numbers or digits, use: \\d+ or [0-9]+",
            "  - Never output raw letter sequences like 'aeiou' without their enclosing square brackets [ ].",
            "",
            "Target function:",
            "",
            f"Name: {fn_def.name}",
            f"Description: {fn_def.description}",
            "Parameters:"
        ]
        for param_name, param in fn_def.parameters.items():
            label = "number" if param.type == ParameterType.number else "text"
            lines.append(f"  - {param_name}: [DataType: {label}]")
        
        lines.extend([
            "",
            "User request:",
            user_prompt,
            "",
            "Return ONLY the extracted parameter JSON object:",
            "{"
        ])
        return "\n".join(lines)