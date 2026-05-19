import json


def load_json(path: str) -> list[dict[str, str]] | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
