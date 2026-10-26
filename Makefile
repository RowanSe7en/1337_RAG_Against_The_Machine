.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name "pycache" -prune -exec rm -rf {} + find . -type f -name "*.py[co]" -delete rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov

lint:
	uv run flake8 . uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . uv run mypy . --strict
