class ConstrainedDecoder:
    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def mask(self, logits, allowed):
        for i in range(len(logits)):
            if i not in allowed:
                logits[i] = float("-inf")
        return logits

    def pick(self, logits):
        return max(range(len(logits)), key=lambda i: logits[i])
