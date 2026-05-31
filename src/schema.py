class Schema:
    def __init__(self, function_map):
        self.function_map = function_map

    # -------------------------
    # FUNCTION NAME TOKENS
    # -------------------------
    def allowed_function_tokens(self, llm, prefix):
        allowed = set()

        for fn in self.function_map:
            seq = llm.encode(fn)

            if seq[:len(prefix)] == prefix:
                if len(prefix) < len(seq):
                    allowed.add(seq[len(prefix)])

        return allowed

    # -------------------------
    # PARAMETER KEY TOKENS
    # -------------------------
    def allowed_keys(self, function_name, llm, prefix):
        fn = self.function_map[function_name]

        allowed = set()

        for key in fn["parameters"]:
            seq = llm.encode(key)

            if seq[:len(prefix)] == prefix:
                if len(prefix) < len(seq):
                    allowed.add(seq[len(prefix)])

        return allowed
