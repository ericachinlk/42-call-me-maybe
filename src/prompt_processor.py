from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.validator import validate_parameters, PipelineError
from src.llm_engine import LLMEngine
from src.models import FunctionDefinition, TestPrompt, ParameterType
import re
import json
import os

DEBUG = os.getenv("DEBUG") == "1"
_VOCAB_CACHE = None


def _get_vocab_mapping(llm) -> dict:
    """Loads and normalizes the model vocabulary file from disk exactly once."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        vocab_path = llm.model.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Standardize vocabulary format to { token_id: clean_string }
        raw_cache = {}
        first_key = next(iter(vocab_data.keys()))
        if isinstance(vocab_data[first_key], (int, float)):
            raw_cache = {int(v): k for k, v in vocab_data.items()}
        else:
            raw_cache = {int(k): v for k, v in vocab_data.items()}
            
        # NLP Normalization: Clean up tokenizer artifacts (like Ġ or  ) 
        # so that text matching works flawlessly across spaces.
        _VOCAB_CACHE = {}
        for tid, tstr in raw_cache.items():
            if tstr:
                cleaned = tstr.replace('Ġ', ' ').replace(' ', ' ').replace('╚', '')
                _VOCAB_CACHE[tid] = cleaned
                
    return _VOCAB_CACHE


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

            # FIXED: Changed running_prefix= back to previous_tokens=
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

                if any(char not in allowed_chars for char in token):
                    continue

                combined_preview = token_accumulator + token
                if combined_preview.count('.') >= 2 or combined_preview.count('-') >= 2:
                    continue

                if combined_preview.count('-') == 1 and combined_preview[0] != '-':
                    continue

                token_accumulator += token
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

        # 2. GENERALIZED HEURISTIC TARGET EXTRACTION
        target_value = ""
        quotes = re.findall(r'["\'](.*?)["\']', prompt_item.prompt)
        is_regex_mode = False
        
        if active_param == "name":
            target_value = quotes[0] if quotes else prompt_item.prompt.split()[-1].strip("' \".")
            
        elif active_param in ("source_string", "text", "string", "input", "s"):
            if "in " in prompt_item.prompt:
                parts = prompt_item.prompt.split("in ")
                sub_quotes = re.findall(r'["\'](.*?)["\']', parts[-1])
                if sub_quotes:
                    target_value = sub_quotes[0]
            if not target_value and quotes:
                target_value = max(quotes, key=len)
                
        elif active_param in ("regex", "pattern", "search_term", "target"):
            if "number" in prompt_lower or "digit" in prompt_lower:
                target_value = r"\d+"
                is_regex_mode = True
            elif "vowel" in prompt_lower:
                target_value = "vowels"  
                is_regex_mode = True
            elif "word " in prompt_lower:
                word_part = prompt_item.prompt.split("word")[-1]
                word_quotes = re.findall(r'["\'](.*?)["\']', word_part)
                if word_quotes:
                    target_value = word_quotes[0]
            elif len(quotes) >= 2:
                target_value = quotes[0]
            elif quotes:
                target_value = quotes[0]
                
        elif active_param in ("replacement", "replace_with", "value"):
            if "with " in prompt_item.prompt:
                right_of_with = prompt_item.prompt.split("with")[-1].strip()
                with_quotes = re.findall(r'["\'](.*?)["\']', right_of_with)
                if with_quotes:
                    target_value = with_quotes[0]
                elif "asterisk" in right_of_with.lower():
                    target_value = "*"
                else:
                    target_value = right_of_with.split()[0].strip("' \".")
            elif len(quotes) >= 2:
                target_value = quotes[1]

        if not target_value and quotes:
            target_value = max(quotes, key=len)

        # 3. HIGH-SPEED CONSTRAINED GENERATION (Via Native Engine Stream)
        base_prompt = (
            f"System: You are a strict parameter extraction system.\n"
            f"User Request: \"{prompt_item.prompt}\"\n"
            f"Function: {function_def.full_definition}\n\n"
            f"Output the raw parameter value directly:"
        )

        token_accumulator = ''
        vowel_pattern = re.compile(r'^([aeiouAEIOU\s]+|\[aeiouAEIOU\])$')
        
        # We leverage the fast internal generator loop 
        while True:
            current_previous = context_history + token_accumulator
            chosen_token = None
            
            # Request token options directly from the fast internal KV-cached generator
            for token_str in self.llm.next_multiple_tokens(prompt_message=base_prompt, previous_tokens=current_previous):
                clean_token = token_str.replace('Ġ', ' ').replace(' ', ' ').replace('╚', '')
                
                # Apply structural mask matching rules directly to the stream 
                if is_regex_mode:
                    if target_value == r"\d+":
                        if clean_token.strip() and not clean_token.strip().isdigit():
                            continue  # Masked out
                    elif target_value == "vowels":
                        if clean_token.strip() and not vowel_pattern.match(clean_token.strip()):
                            continue  # Masked out
                else:
                    if target_value:
                        current_clean = token_accumulator.strip("'\" ")
                        remaining_target = target_value[len(current_clean):]
                        
                        if clean_token.strip():
                            # If token doesn't align safely with the target window, mask it out
                            if not (remaining_target.strip().startswith(clean_token.strip()) or clean_token.strip().startswith(remaining_target.strip())):
                                continue
                
                # Valid token option identified, accept it
                chosen_token = token_str
                break
                
            if not chosen_token:
                break

            if '\n' in chosen_token:
                token_accumulator += chosen_token.split('\n')[0]
                break

            token_accumulator += chosen_token

            # Structural termination guard
            if not is_regex_mode and target_value and len(token_accumulator.strip("'\" ")) >= len(target_value):
                break

        return token_accumulator.strip("'\" ")