from llm_sdk.llm_sdk import Small_LLM_Model


class LLMEngine:
    def __init__(self) -> None:
        self.model = Small_LLM_Model()

    def encode(self, text: str) -> list[int]:
        return self.model.encode(text)[0].tolist()

    def decode(self, tokens: list[int]) -> str:
        return self.model.decode(tokens)

    def logits(self, tokens: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(tokens)
