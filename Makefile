FUNCTION_PATH = src/data/input/functions_definition.json
INPUT_PATH = src/data/input/function_calling_tests.json
OUTPUT_PATH = src/data/output/function_calls.json

install:
	uv sync

install-lint:
	uv sync --extra dev

run:
	uv run python -m src \
		--functions_definition $(FUNCTION_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

debug:
	DEBUG=1 uv run python -m src \
		--functions_definition $(FUNCTION_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run python -m flake8 . --exclude=.venv,__pycache__,.mypy_cache,llm_sdk
	uv run python -m mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs --exclude '(.venv|llm_sdk)'

lint-strict:
	uv run python -m flake8 . --exclude=.venv,__pycache__,.mypy_cache,llm_sdk
	uv run python -m mypy . --strict --exclude '(.venv|llm_sdk)'

.PHONY: install install-lint run debug clean lint lint-strict
