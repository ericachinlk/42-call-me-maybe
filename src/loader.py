import json


def load_json(path: str) -> list[dict] | None:
    try:
        with open(path, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {path}")
        return None

    except Exception as e:
        print(f"Unexpected error loading {path}: {e}")
        return None