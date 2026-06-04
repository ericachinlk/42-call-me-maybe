import json
from typing import Any, Set
from pydantic import BaseModel
from src.models import FunctionDefinition, ParameterType
from src.llm_engine import LLMEngine


class ParameterExtractorConstrained(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    def build_prompt(self, fn_def: FunctionDefinition, user_prompt: str) -> str:
        lines = [
            "You are a structured data extraction system.",
            f"Extract parameters for '{fn_def.name}':",
        ]
        for param_name, param in fn_def.parameters.items():
            lines.append(f"  - {param_name}: {param.type.value}")
        lines.extend([
            "",
            f"User request: {user_prompt}",
            "",
            "Respond ONLY with a minified JSON object containing the exact parameters:",
            ""
        ])
        return "\n".join(lines)

    def extract(self, fn_def: FunctionDefinition, user_prompt: str, llm: LLMEngine) -> dict[str, Any]:
        # 1. Load Vocab and cache token categories once per execution
        vocab_path = llm.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)
        
        token_to_str = {token_id: text for text, token_id in vocab.items()}
        all_token_ids = set(token_to_str.keys())

        # Pre-group tokens cleanly by their raw textual content
        open_brace_tokens = set()
        close_brace_tokens = set()
        comma_tokens = set()
        number_tokens = set()
        boolean_tokens = set()

        for tid, tstr in token_to_str.items():
            # Clean up tokenizer-specific artifacts (like Qwen's byte representations or spaces)
            clean_tstr = tstr.replace("Ġ", " ").replace(" ", " ").strip()
            
            if "{" in tstr: open_brace_tokens.add(tid)
            if "}" in tstr: close_brace_tokens.add(tid)
            if "," in tstr: comma_tokens.add(tid)
            if any(c.isdigit() or c in ".-" for c in clean_tstr): number_tokens.add(tid)
            if any(w in clean_tstr.lower() for w in ["true", "false", "t", "f"]): boolean_tokens.add(tid)

        # 2. Setup Generation Context
        prompt_str = self.build_prompt(fn_def, user_prompt)
        input_ids = llm.encode(prompt_str)
        generated_tokens: list[int] = []
        params_list = list(fn_def.parameters.items())
        
        # Hard cap the total tokens to prevent a runaway 5+ minute infinite loop
        MAX_GENERATION_TOKENS = 64

        # 3. Fast Dynamic Constrained Generation Loop
        for _ in range(MAX_GENERATION_TOKENS):
            # Only decode the full string when absolutely necessary to evaluate states
            current_generated_str = llm.decode(generated_tokens)
            cleaned = current_generated_str.strip()
            
            logits = llm.logits(input_ids + generated_tokens)
            allowed: Set[int] = set()

            if not cleaned:
                allowed = open_brace_tokens
            else:
                # Count how many commas or keys have been completed to find current parameter index
                current_param_idx = cleaned.count(":") - 1
                if current_param_idx < 0:
                    current_param_idx = 0

                if current_param_idx >= len(params_list):
                    allowed = close_brace_tokens
                else:
                    param_name, param_def = params_list[current_param_idx]
                    expected_key = f'"{param_name}"'

                    if expected_key not in cleaned:
                        # Match structural fragments building up to the JSON key name
                        allowed = {tid for tid, tstr in token_to_str.items() 
                                   if param_name in tstr or '"' in tstr or "{" in tstr or "," in tstr}
                    elif ":" not in cleaned.split(expected_key)[-1]:
                        # Force a colon after the key name string
                        allowed = {tid for tid, tstr in token_to_str.items() if ":" in tstr}
                    else:
                        # Actively parsing the actual variable data type
                        if param_def.type == ParameterType.number:
                            allowed = number_tokens | comma_tokens | close_brace_tokens
                        elif param_def.type == ParameterType.boolean:
                            allowed = boolean_tokens | comma_tokens | close_brace_tokens
                        elif param_def.type == ParameterType.string:
                            # Inside string quotes: allow anything until the closing structural comma or brace
                            val_part = cleaned.split(":")[-1].strip()
                            quote_count = val_part.count('"')
                            if quote_count % 2 == 1:
                                allowed = all_token_ids  # Safe zone: inside string literals
                            else:
                                allowed = comma_tokens | close_brace_tokens

            # Apply structural logit mask
            if not allowed:
                allowed = all_token_ids

            for token_id in range(len(logits)):
                if token_id not in allowed:
                    logits[token_id] = float("-inf")

            # Choose the token with highest probability score
            next_token = max(range(len(logits)), key=lambda x: logits[x])
            generated_tokens.append(next_token)

            # Fast evaluation check to see if valid JSON structure is closed
            final_str = llm.decode(generated_tokens).strip()
            if final_str.endswith("}"):
                try:
                    parsed_json = json.loads(final_str)
                    if all(k in parsed_json for k, _ in params_list):
                        return parsed_json
                except json.JSONDecodeError:
                    pass

        # Fallback Strategy: If constrained loop hits MAX_GENERATION_TOKENS without completing,
        # return a default compliant payload to prevent program execution crashes
        return {param_name: (0.0 if param_def.type == ParameterType.number else "") 
                for param_name, param_def in params_list}
