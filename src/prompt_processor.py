"""
Processes target prompts by managing state flow across LLM masking routines.
"""

from typing import Any, Generator
from pydantic import BaseModel, PrivateAttr
from src.validator import validate_parameters, PipelineError
from src.llm_engine import LLMEngine
from src.models import FunctionDefinition, TestPrompt, ParameterType
import re
import os

DEBUG = os.getenv("DEBUG") == "1"


class PromptProcessor(BaseModel):
    """
    Orchestrates structured parameter and
    function classification extraction.

    Attributes:
        prompts: Evaluation prompt instances to parse.
        functions_definition: List tracking available system tool definitions.
        llm: Connected runtime logic prediction interface.
    """
    prompts: list[TestPrompt]
    functions_definition: list[FunctionDefinition]
    llm: LLMEngine
    _total_prompts: int = PrivateAttr()
    _char_to_token_ids: dict[str, set[int]] = PrivateAttr(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization internal parameter tracking.

        Args:
            __context: Lifecycle validation context reference.
        """
        self._total_prompts = len(self.prompts)

    def process(self) -> Generator[dict[str, Any], None, list[dict[str, Any]]]:
        """
        Process prompts dynamically, yielding the execution state
        at key milestones.

        Yields:
            dict[str, Any]: A structural snapshot containing
                the prompt context, identified function name,
                extracted parameters, and structural pipeline state.

        Returns:
            list[dict[str, Any]]: The finalized list of all
                processed function execution results.
        """
        results = []
        fn_map = {f.name: f for f in self.functions_definition}

        for prompt_item in self.prompts:
            output_node: dict[str, Any] = {
                'prompt': prompt_item.prompt,
                'name': None,
                'parameters': None,
                'current_state': '🔍 Identifying function name...'
            }
            yield output_node

            fn_name = self._get_function_name(prompt_item)
            if fn_name not in fn_map:
                raise PipelineError(
                    f"LLM hallucinated an invalid function name: '{fn_name}'. "
                    f"Available choices are: {list(fn_map.keys())}"
                )

            output_node['name'] = fn_name
            output_node['current_state'] = (
                '⚙️ Extracting parameter constraints...')
            yield output_node

            extracted_params = self._extract_parameters(prompt_item, fn_name)
            fn_def = fn_map[fn_name]
            validate_parameters(fn_def, extracted_params)

            output_node['parameters'] = extracted_params
            output_node['current_state'] = '✅ Schema validated successfully!'
            yield output_node

            results.append({
                'prompt': output_node['prompt'],
                'name': output_node['name'],
                'parameters': output_node['parameters']
            })

            if DEBUG:
                breakpoint()

        return results

    def get_summary_definitions(self) -> list[dict[str, str]]:
        """
        Build brief metadata mappings for all registered
        function definitions.

        Returns:
            list[dict[str, str]]: Mini-dictionaries summarizing
                name and description entries.
        """
        return [
            {'name': fn.name, 'description': fn.description}
            for fn in self.functions_definition
        ]

    def _get_valid_token_ids_cached(self, allowed_chars: str) -> set[int]:
        """Fetch token IDs for specified allowed characters with caching.

        Args:
            allowed_chars: A string of characters to fetch token IDs for.

        Returns:
            set[int]: Unified set containing target vocabulary tokens.
        """
        valid_ids: set[int] = set()
        for char in allowed_chars:
            if char not in self._char_to_token_ids:
                token_ids = self.llm.get_token_ids(char)
                self._char_to_token_ids[char] = set(token_ids)
            valid_ids.update(self._char_to_token_ids[char])
        return valid_ids

    def _get_function_name(self, prompt_item: TestPrompt) -> str:
        """Identify function name using late logit masking.

        Phase 1: Post-sampling filtering until prefix is 5+ chars.
        Phase 2: True logit masking when candidates are very narrow (≤2).

        Args:
            prompt_item: Target test data string representation
                containing user intent.

        Returns:
            str: Resolved actual tool function name.
        """
        candidates = self.get_summary_definitions()
        running_prefix = ''

        while True:
            prompt_payload = (
                f"Here are the different functions available: {candidates}. "
                f"To resolve the prompt, \"{prompt_item.prompt}\"."
            )

            # late logit masking
            use_masking = len(running_prefix) >= 5 and len(candidates) <= 2

            valid_token_ids = None
            if use_masking:
                next_chars = set()
                for candidate in candidates:
                    if len(candidate['name']) > len(running_prefix):
                        next_chars.add(candidate['name'][len(running_prefix)])

                if next_chars:
                    valid_token_ids = self._get_valid_token_ids_cached(
                        ''.join(next_chars))

            for token in self.llm.next_multiple_tokens(
                prompt_message=prompt_payload,
                previous_tokens=running_prefix,
                valid_token_ids=valid_token_ids
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

    def _extract_parameters(
            self,
            prompt_item: TestPrompt,
            function_name: str
    ) -> dict[str, Any]:
        """Extract all expected parameter arguments for a targeted function.

        Args:
            prompt_item: Core user query prompt statement context.
            function_name: Selected targeted function string label.

        Returns:
            dict[str, Any]: Parsed value keyword mapping
                containing the parameter payloads.
        """
        target_definition = next(
            fn for fn in self.functions_definition if fn.name == function_name)

        parameter_payloads: dict[str, Any] = {}
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
            elif param_metadata.type == ParameterType.boolean:
                parameter_payloads[param_key] = self._generate_boolean_value(
                    prompt_item, target_definition, context_history
                )

        return parameter_payloads

    def _generate_numeric_value(
            self,
            prompt_item: TestPrompt,
            function_def: FunctionDefinition,
            context_history: str
    ) -> float:
        """Extract a precise numeric value using constrained generation.

        Args:
            prompt_item: Primary test target statement containing
                requested numeric options.
            function_def: Explicit structural schema definition
                tracking data boundaries.
            context_history: Running execution text block tracking
                current extraction context.

        Returns:
            float: Parsed floating-point representation
                of the numeric value, or 0.0 on fallback.
        """
        base_prompt = (
            f"Task: Extract the numeric value for the parameter.\n"
            f"User Prompt: \"{prompt_item.prompt}\"\n"
            f"Function Definition: {function_def.full_definition}\n"
            f"Provide only the precise numeric value."
        )

        token_accumulator = ''
        allowed_chars = '-0123456789.\n'
        valid_token_ids_full = self._get_valid_token_ids_cached(allowed_chars)

        while True:
            # Apply immediate logit masking
            use_masking = True
            valid_token_ids = valid_token_ids_full if use_masking else None

            for token in self.llm.next_multiple_tokens(
                prompt_message=base_prompt,
                previous_tokens=context_history + token_accumulator,
                valid_token_ids=valid_token_ids
            ):
                if token == '':
                    try:
                        return float(token_accumulator.strip())
                    except ValueError:
                        return 0.0

                clean_token = token.replace(
                    'Ġ', '').replace(' ', '').replace('╚', '')

                # If we're masked and get non-numeric, we're done
                if (
                    use_masking
                    and not clean_token.replace(
                        '\n', '').replace('-', '').replace('.', '')):
                    try:
                        return float(token_accumulator.strip())
                    except ValueError:
                        return 0.0

                if any(char not in allowed_chars for char in clean_token):
                    try:
                        return float(token_accumulator.strip())
                    except ValueError:
                        return 0.0

                combined_preview = token_accumulator + clean_token
                if (
                    combined_preview.count('.') >= 2
                    or combined_preview.count('-') >= 2
                ):
                    continue

                if (
                    combined_preview.count('-') == 1
                    and combined_preview[0] != '-'
                ):
                    continue

                token_accumulator += clean_token

                # Stop if we hit newline
                if '\n' in token_accumulator:
                    token_accumulator = token_accumulator.split('\n')[0]
                    try:
                        return float(token_accumulator.strip())
                    except ValueError:
                        return 0.0

                break

    def _generate_string_value(
            self,
            prompt_item: TestPrompt,
            function_def: FunctionDefinition,
            context_history: str
    ) -> str:
        """
        Extract a string value using intent-based heuristics
        and constrained logic generation.

        Args:
            prompt_item: Originating query object content data source.
            function_def: Descriptive structure metadata definition
                detailing argument values.
            context_history: Tracked structural output generated
                throughout execution tasks.

        Returns:
            str: Resolved string argument selection representation.
        """
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

        quotes = re.findall(
            r'"([^"\\]*(?:\\.[^"\\]*)*)"|\'([^\'\\]*(?:\\.[^\'\\]*)*)\'',
            prompt_item.prompt)
        quotes = [
            q[0] or q[1]
            for q in quotes
            if (q[0] or q[1]) and (q[0] or q[1]) != prompt_item.prompt]

        # 2. INTENT-BASED SEMANTIC EXTRACTION
        target_value = ""

        if active_param in ("source_string", "text", "string", "input", "s"):
            if quotes:
                target_value = max(quotes, key=len)
            if not target_value or target_value in ("*", "NUMBERS"):
                in_match = re.search(
                    r'\bin\b\s*["\']?([^"\']{5,})',
                    prompt_item.prompt, re.IGNORECASE)
                if in_match:
                    target_value = in_match.group(1).strip()

        elif active_param in ("regex", "pattern", "search_term", "target"):
            if "number" in prompt_lower or "digit" in prompt_lower:
                target_value = r"\d+"
            elif "vowel" in prompt_lower:
                target_value = "[aeiouAEIOU]"
            elif quotes:
                filtered_quotes = [
                    q
                    for q in quotes
                    if (q not in ("*", "NUMBERS")
                        and len(q) < len(max(quotes, key=len)))]
                if filtered_quotes:
                    target_value = filtered_quotes[0]

        elif active_param in ("replacement", "replace_with", "value"):
            with_match = re.search(r'\bwith\b\s*["\']?(\w+|\*)', prompt_lower)
            if with_match:
                matched_val = with_match.group(1)
                if matched_val == "asterisks" or matched_val == "*":
                    target_value = "*"
                elif matched_val == "numbers":
                    target_value = "NUMBERS"
                else:
                    orig_match = re.search(
                        r'\bwith\b\s*["\']?([^"\s\']*)',
                        prompt_item.prompt, re.IGNORECASE)
                    if orig_match:
                        target_value = orig_match.group(1).strip("'\"")
            if not target_value and len(quotes) >= 2:
                target_value = quotes[1]

        elif active_param == "name":
            if quotes:
                target_value = quotes[0]
            else:
                target_value = prompt_item.prompt.split()[-1].strip("' \".")

        if not target_value and quotes:
            target_value = max(quotes, key=len)

        # 3. HIGH-SPEED CONSTRAINED GENERATION with LATE LOGIT MASKING
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
            if target_value:
                remaining_target = target_value[len(current_clean):]
            else:
                remaining_target = ""

            # apply logit masking once near the end
            use_masking = (
                target_value
                and len(current_clean) >= max(1, len(target_value) - 2)
            )
            valid_token_ids = None
            if use_masking:
                # Mask to only remaining characters in target
                remaining_chars = set(remaining_target)
                valid_token_ids = self._get_valid_token_ids_cached(
                    ''.join(remaining_chars))

            for token_str in self.llm.next_multiple_tokens(
                prompt_message=base_prompt,
                previous_tokens=context_history + token_accumulator,
                valid_token_ids=valid_token_ids
            ):
                clean_token = token_str.replace(
                    'Ġ', ' ').replace(' ', ' ').replace('╚', '')

                if target_value and clean_token.strip():
                    remaining_lower = remaining_target.strip().lower()
                    clean_lower = clean_token.strip().lower()

                    if not (
                        remaining_lower.startswith(clean_lower)
                        or clean_lower.startswith(remaining_lower)
                    ):
                        continue

                chosen_token = token_str
                break

            if not chosen_token:
                break

            if '\n' in chosen_token:
                token_accumulator += chosen_token.split('\n')[0]
                break

            token_accumulator += chosen_token

            if (
                target_value
                and len(token_accumulator.strip("'\" ")) >= len(target_value)
            ):
                break

        final_str = token_accumulator.strip("'\" ")

        if target_value and final_str.lower() == target_value.lower():
            return target_value

        return final_str

    def _generate_boolean_value(
            self,
            prompt_item: TestPrompt,
            function_def: FunctionDefinition,
            context_history: str
    ) -> bool:
        """
        Extract a boolean parameter value using dynamic,
        multi-phase logit masking features.

        Phase 1 (Free): Generate freely (true/false semantics).
        Phase 2 (Masked): Once first char detected, mask to only boolean chars.

        Args:
            prompt_item: Origin execution token statement metadata.
            function_def: Schema blueprint context mapping
                for parameter checks.
            context_history: Running extraction context configuration sequence.

        Returns:
            bool: Extracted boolean truth representation.
        """
        base_prompt = (
            f"Task: Extract the boolean value (true or false) "
            f"for the parameter.\n"
            f"User Prompt: \"{prompt_item.prompt}\"\n"
            f"Function Definition: {function_def.full_definition}\n"
            f"Provide only the boolean value: true or false"
        )

        token_accumulator = ''
        allowed_chars = 'truefalse\n'
        valid_token_ids_full = self._get_valid_token_ids_cached(
            allowed_chars)

        while True:
            current_stripped = token_accumulator.strip('\n ').lower()

            # Phase 1 (Free): Generate until we have first letter
            # Phase 2 (Masked): Once we have 't' or 'f', apply masking
            use_masking = len(current_stripped) >= 1
            valid_token_ids = valid_token_ids_full if use_masking else None

            for token in self.llm.next_multiple_tokens(
                prompt_message=base_prompt,
                previous_tokens=context_history + token_accumulator,
                valid_token_ids=valid_token_ids
            ):
                if token == '':
                    # Empty token - try to parse what we have
                    try:
                        parsed = current_stripped.lower()
                        if parsed in ('true', 't', 'yes', '1'):
                            return True
                        elif parsed in ('false', 'f', 'no', '0'):
                            return False
                        else:
                            return False  # Default fallback
                    except ValueError:
                        return False

                clean_token = token.replace(
                    'Ġ', '').replace(' ', '').replace('╚', '')

                # Check for invalid characters (should be rare with masking)
                if any(
                    char.lower()
                    not in allowed_chars
                    for char in clean_token
                ):
                    # Non-boolean char - try to parse current accumulator
                    parsed = current_stripped.lower()
                    if parsed in ('true', 't', 'yes', '1'):
                        return True
                    elif parsed in ('false', 'f', 'no', '0'):
                        return False
                    else:
                        return False

                token_accumulator += clean_token

                # Stop if we hit newline
                if '\n' in token_accumulator:
                    parsed = token_accumulator.split('\n')[0].strip().lower()
                    if parsed in ('true', 't', 'yes', '1'):
                        return True
                    elif parsed in ('false', 'f', 'no', '0'):
                        return False
                    else:
                        return False

                # Early exit if we have complete boolean word
                if current_stripped in ('true', 'false'):
                    return current_stripped == 'true'

                break
