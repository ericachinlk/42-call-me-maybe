FUNCTION_PATH = data/input/functions_definition.json
INPUT_PATH = data/input/function_calling_tests.json
OUTPUT_PATH = data/output/function_calls.json
EDGE_CASE_INPUT = data/input/edge_cases_tests.json
EDGE_CASE_OUTPUT = data/output/edge_cases_function_calls.json

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTION_PATH) \
		--input $(INPUT_PATH) \
		--output $(OUTPUT_PATH)

run-edge:
	uv run python -m src \
		--input $(EDGE_CASE_INPUT) \
		--output $(EDGE_CASE_OUTPUT)

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
	uv run python -m flake8 . \
	--exclude=.venv,__pycache__,.mypy_cache,llm_sdk,tests
	uv run python -m mypy . --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --follow-imports=skip \
	--check-untyped-defs --exclude '(.venv|llm_sdk|tests)'

lint-strict:
	uv run python -m flake8 . \
	--exclude=.venv,__pycache__,.mypy_cache,llm_sdk,tests
	uv run python -m mypy . --strict --follow-imports=skip \
	--exclude '(.venv|llm_sdk|tests)'

.PHONY: install run debug clean lint lint-strict
