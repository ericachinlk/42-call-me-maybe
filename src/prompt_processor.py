from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.llm_engine import LLMEngine
from src.models import FunctionDefinition, TestPrompt, ParameterType


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

        for prompt_item in self.prompts:
            output_node = {}
            output_node['prompt'] = prompt_item.prompt

            selected_fn = self._identify_function_name(prompt_item)
            output_node['name'] = selected_fn

            extracted_params = self._extract_parameters(prompt_item, selected_fn)
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

            for token in self.llm.predict_multiple_tokens(
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
            f"To solve the prompt {prompt_item}, you will use the following function: "
            f"{function_def.full_definition}. Provide each parameter. Keep it concise and "
            f"don't add custom fields."
        )

        token_accumulator = ''
        allowed_chars = '-0123456789.\n'

        while True:
            for token in self.llm.predict_multiple_tokens(
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
        base_prompt = (
            f"To solve the prompt {prompt_item}, you will use the following function: "
            f"{function_def.full_definition}. Provide each parameter. Keep it concise and "
            f"don't add custom fields."
        )

        token_accumulator = ''
        max_token_budget = 50 
        
        for _ in range(max_token_budget):
            token = self.llm.predict_token(
                prompt_message=base_prompt, 
                previous_tokens=context_history + token_accumulator
            )
            
            if not token:
                break

            if '\n' in token:
                token_accumulator += token.split('\n')[0]
                break

            token_accumulator += token

            for alternate_key in function_def.parameters:
                spacer_marker = f", {alternate_key}="
                dense_marker = f",{alternate_key}="
                
                if spacer_marker in token_accumulator:
                    return token_accumulator.split(spacer_marker)[0].strip("'\" ")
                if dense_marker in token_accumulator:
                    return token_accumulator.split(dense_marker)[0].strip("'\" ")

        return token_accumulator.strip("'\" ")