import re
from typing import Any


class ParameterExtractor:
    def __init__(self, llm) -> None:
        self.llm = llm

    def extract(self, fn_def, prompt: str) -> dict[str, Any]:
        params = {}
        numbers = self._get_numbers(prompt)
        strings = self._extract_strings(prompt)

        num_i = 0
        for param_name, param_type in fn_def.parameters.items():
            value: Any = None
            name = param_name.lower()

            if param_type.type == "number":
                value = numbers[num_i] if num_i < len(numbers) else 0.0
                num_i += 1

            elif param_type.type == "string":
                if "source" in name or "input" in name:
                    value = self._get_source_string(prompt, strings)
                elif "replacement" in name:
                    value = self._extract_replacement(prompt)
                elif "regex" in name:
                    value = self._infer_regex(prompt)
                else:
                    value = strings[0] if strings else ""

            if value is None:
                value = self._default_value(name, param_type.type)
            params[param_name] = self._cast(value, param_type.type)
        return params

    def _get_source_string(self, prompt, strings) -> str:
        if strings:
            return max(strings, key=len)
        return prompt

    def _extract_replacement(self, prompt: str) -> str:
        alias_map = {
            "asterisks": "*",
            "commas": ",",
            "colons": ":",
            "hyphens": "-",
            "spaces": " ",
            "dots": "."
        }

        match = re.search(
            r"\bwith\s+['\"]?(.*?)['\"]?\s*(?:$|in\b)",
            prompt,
            re.IGNORECASE
        )

        if match:
            result = match.group(1).strip(" '\"")
            if result in alias_map.keys():
                result = alias_map[result]
            return result
        return ""

    def _infer_regex(self, prompt: str) -> str:
        p = prompt.lower()

        if "vowel" in p:
            return r"[aeiouAEIOU]"
        if "number" in p or "digit" in p:
            return r"\d+"
        if "space" in p:
            return r"\s+"
        if "replace the word" in p or "substitute the word" in p:
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
            quoted = [a or b for a, b in quoted]
            if quoted:
                return quoted[0]
        return r"\w+"

    def _get_numbers(self, prompt) -> list[float]:
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", prompt)]

    def _extract_strings(self, prompt: str) -> list[str]:
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
        strings = [a or b for a, b in quoted]
        if strings:
            return strings

        cleaned = re.sub(
            r"\b(greet|reverse|replace|calculate|sum|substitute)\b",
            "",
            prompt,
            flags=re.IGNORECASE
        )
        words = re.findall(r"[A-Za-z]+", cleaned)
        stop = {"what", "is", "the", "and", "with", "in", "to", "of"}
        return [w for w in words if w.lower() not in stop]

    def _default_value(self, name, t) -> Any:
        if t == "number":
            return 0.0
        if t == "string":
            if "regex" in name:
                return ".*"
            return ""
        return None

    def _cast(self, value, t) -> Any:
        if value is None:
            return None
        if t == "number":
            try:
                return float(value)
            except:
                return 0.0
        if t == "string":
            return str(value).strip("'\"")
        if t == "boolean":
            return str(value).lower() in {"true", "1", "yes"}
        return value
