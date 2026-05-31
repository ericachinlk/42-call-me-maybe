from src.models import FunctionCallOutput


class Validator:
    def validate(self, prompt, name, parameters):
        try:
            obj = FunctionCallOutput(
                prompt=prompt,
                name=name,
                parameters=parameters,
            )
            return obj.model_dump()

        except Exception as e:
            # NEVER crash (assignment requirement)
            return {
                "prompt": prompt,
                "name": name,
                "parameters": {},
                "error": str(e),
            }