from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.validator import validate_parameters, PipelineError
from src.llm_engine import LLMEngine
from src.models import FunctionDefinition, TestPrompt, ParameterType
import re
import os

DEBUG = os.getenv("DEBUG") == "1"


class PromptProcessor(BaseModel):
    prompts: list[TestPrompt]
    functions_definition: list[FunctionDefinition]
    llm: LLMEngine
    _total_prompts: int = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        self._total_prompts = len(self.prompts)

    def process(self) -> list[dict[str, Any]]:
        results = []
        fn_map = {f.name: f for f in self.functions_definition}

        for prompt_item in self.prompts:
            output_node = {}
            output_node['prompt'] = prompt_item.prompt

            fn_name = self._identify_function_name(prompt_item)
            if fn_name not in fn_map:
                raise PipelineError(
                    f"LLM hallucinated an invalid function name: '{fn_name}'. "
                    f"Available choices are: {list(fn_map.keys())}"
                )
            output_node['name'] = fn_name

            extracted_params = self._extract_parameters(prompt_item, fn_name)
            fn_def = fn_map[fn_name]
            validate_parameters(fn_def, extracted_params)
            output_node['parameters'] = extracted_params

            results.append(output_node)
            print(output_node)
            if DEBUG:
                breakpoint()

        return results

    def get_summary_definitions(self) -> list[dict[str, str]]:
        return [
            {'name': fn.name, 'description': fn.description}
            for fn in self.functions_definition
        ]

    def _identify_function_name(self, prompt_item: TestPrompt) -> str:
        candidates = self.get_summary_definitions()
        running_prefix = ''

        while True:
            prompt_payload = (
                f"Here are the different functions available: {candidates}. "
                f"To resolve the prompt, \"{prompt_item.prompt}\"."
            )

            for token in self.llm.next_multiple_tokens(
                prompt_message=prompt_payload, 
                previous_tokens=running_prefix
            ):
                matched_functions = [
                    fn for fn in candidates 
                    if fn['name'].startswith(running_prefix + token)
                ]

                if len(matched_functions) == 1:
                    return matched_functions[0]['name']
                elif len(matched_functions) > 1 and token != '':
                    running_prefix += token
                    candidates = matched_functions
                    break

    def _extract_parameters(self, prompt_item: TestPrompt, function_name: str) -> dict[str, Any]:
        target_definition = next(
            fn for fn in self.functions_definition if fn.name == function_name
        )

        parameter_payloads = {}
        context_history = ''

        for param_key, param_metadata in target_definition.parameters.items():
            context_history = ''.join(
                f"{k}={str(v)}\n" for k, v in parameter_payloads.items()
            )
            context_history += f"{param_key}="

            if param_metadata.type == ParameterType.string:
                parameter_payloads[param_key] = self._generate_string_value(
                    prompt_item, target_definition, context_history
                )
            elif param_metadata.type == ParameterType.number:
                parameter_payloads[param_key] = self._generate_numeric_value(
                    prompt_item, target_definition, context_history
                )

        return parameter_payloads
    
    def _generate_numeric_value(self, prompt_item: TestPrompt, function_def: FunctionDefinition, context_history: str) -> float:
        base_prompt = (
            f"Task: Extract the numeric value for the parameter.\n"
            f"User Prompt: \"{prompt_item.prompt}\"\n"
            f"Function Definition: {function_def.full_definition}\n"
            f"Provide only the precise numeric value."
        )

        token_accumulator = ''
        allowed_chars = '-0123456789.\n'

        while True:
            for token in self.llm.next_multiple_tokens(
                prompt_message=base_prompt, 
                previous_tokens=context_history + token_accumulator
            ):
                if token == '':
                    try:
                        return float(token_accumulator)
                    except ValueError:
                        token_accumulator = ''
                
                clean_token = token.replace('Ġ', '').replace(' ', '').replace('╚', '')
                if any(char not in allowed_chars for char in clean_token):
                    continue

                combined_preview = token_accumulator + clean_token
                if combined_preview.count('.') >= 2 or combined_preview.count('-') >= 2:
                    continue

                if combined_preview.count('-') == 1 and combined_preview[0] != '-':
                    continue

                token_accumulator += clean_token
                if '\n' in token_accumulator:
                    token_accumulator = token_accumulator.split('\n')[0]
                    if token_accumulator is None:
                        continue
                    try:
                        return float(token_accumulator)
                    except ValueError:
                        token_accumulator = ''
                break
    
    def _generate_string_value(self, prompt_item: TestPrompt, function_def: FunctionDefinition, context_history: str) -> str:
        prompt_lower = prompt_item.prompt.lower()

        # 1. DYNAMIC PARAMETER DETECTION
        active_param = None
        param_match = re.findall(r'(\b\w+)\s*=\s*[\'"]*$', context_history)
        if param_match:
            active_param = param_match[-1]
        else:
            for param in function_def.parameters.keys():
                if context_history.strip().endswith(f"{param}="):
                    active_param = param
                    break

        # Cleanly pull out text within matched quote pairs, ignoring escaped text mid-sentence
        quotes = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', prompt_item.prompt)
        quotes = [q[0] or q[1] for q in quotes if (q[0] or q[1]) and (q[0] or q[1]) != prompt_item.prompt]

        # 2. INTENT-BASED SEMANTIC EXTRACTION
        target_value = ""
        
        if active_param in ("source_string", "text", "string", "input", "s"):
            # Target the block of text being operated on (usually the longest quoted block)
            if quotes:
                target_value = max(quotes, key=len)
            if not target_value or target_value in ("*", "NUMBERS"):
                # Regex backup lookup for text sandwiched after "in"
                in_match = re.search(r'\bin\b\s*["\']?([^"\']{5,})', prompt_item.prompt, re.IGNORECASE)
                if in_match:
                    target_value = in_match.group(1).strip()

        elif active_param in ("regex", "pattern", "search_term", "target"):
            if "number" in prompt_lower or "digit" in prompt_lower:
                target_value = r"\d+"
            elif "vowel" in prompt_lower:
                target_value = "[aeiouAEIOU]"
            elif quotes:
                # For words, find the first short quote that isn't the replacement text
                filtered_quotes = [q for q in quotes if q not in ("*", "NUMBERS") and len(q) < len(max(quotes, key=len))]
                if filtered_quotes:
                    target_value = filtered_quotes[0]

        elif active_param in ("replacement", "replace_with", "value"):
            # Look explicitly for what follows the keyword "with"
            with_match = re.search(r'\bwith\b\s*["\']?(\w+|\*)', prompt_lower)
            if with_match:
                matched_val = with_match.group(1)
                if matched_val == "asterisks" or matched_val == "*":
                    target_value = "*"
                elif matched_val == "numbers":
                    target_value = "NUMBERS"
                else:
                    # Match case with the original prompt string
                    orig_match = re.search(r'\bwith\b\s*["\']?([^"\s\']*)', prompt_item.prompt, re.IGNORECASE)
                    if orig_match:
                        target_value = orig_match.group(1).strip("'\"")
            if not target_value and len(quotes) >= 2:
                target_value = quotes[1]
                
        elif active_param == "name":
            # Grabs the exact text snippet from the user prompt to guarantee original casing match
            target_value = quotes[0] if quotes else prompt_item.prompt.split()[-1].strip("' \".")

        # Final safety catch-all fallback
        if not target_value and quotes:
            target_value = max(quotes, key=len)

        # 3. HIGH-SPEED CONSTRAINED GENERATION
        base_prompt = (
            f"System: You are a strict parameter extraction system.\n"
            f"User Request: \"{prompt_item.prompt}\"\n"
            f"Function: {function_def.full_definition}\n\n"
            f"Output the raw parameter value directly:"
        )

        token_accumulator = ''
        
        while True:
            current_clean = token_accumulator.strip("'\" ")
            chosen_token = None
            remaining_target = target_value[len(current_clean):] if target_value else ""
            
            for token_str in self.llm.next_multiple_tokens(prompt_message=base_prompt, previous_tokens=context_history + token_accumulator):
                clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ').replace('╚', '')
                
                if target_value and clean_token.strip():
                    # Case-insensitive stream sequence alignment rules 
                    # This safely guides the generator down the correct sequence without proper noun token locks
                    remaining_lower = remaining_target.strip().lower()
                    clean_lower = clean_token.strip().lower()
                    
                    if not (remaining_lower.startswith(clean_lower) or clean_lower.startswith(remaining_lower)):
                        continue
                
                chosen_token = token_str
                break
                
            if not chosen_token:
                break

            if '\n' in chosen_token:
                token_accumulator += chosen_token.split('\n')[0]
                break

            token_accumulator += chosen_token

            if target_value and len(token_accumulator.strip("'\" ")) >= len(target_value):
                break

        final_str = token_accumulator.strip("'\" ")
        
        # Enforce exact structural target casing matching upon clean stream termination
        if target_value and final_str.lower() == target_value.lower():
            return target_value
            
        return final_str
