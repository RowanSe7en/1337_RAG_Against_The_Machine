GOINFRE_DIR = /home/$(USER)/goinfre/rag_against_brouane

UV_CACHE    = $(GOINFRE_DIR)/uv-cache
HF_CACHE    = $(GOINFRE_DIR)/huggingface

EXPORT_UV_CACHE = export UV_CACHE_DIR="$(GOINFRE_DIR)/uv-cache"
EXPORT_HF_HOME  = export HF_HOME="$(GOINFRE_DIR)/huggingface"
EXPORT_VENV     = export UV_PROJECT_ENVIRONMENT="$(GOINFRE_DIR)/.venv"

QUERY ?= How to configure the OpenAI server?
K ?= 5

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run: index 

index:
	uv run python -m src index 

search:
	uv run python -m src search "$(QUERY)" --k $(K)

debug:
	uv run python -m pdb -m src


lint:
	uv run flake8 . uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . uv run mypy . --strict

setup_goinfre:
	mkdir -p $(UV_CACHE)
	mkdir -p $(HF_CACHE)

	@echo "echo '$(EXPORT_UV_CACHE)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_UV_CACHE)' ~/.zshrc || echo '$(EXPORT_UV_CACHE)' >> ~/.zshrc

	@echo "echo '$(EXPORT_HF_HOME)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_HF_HOME)' ~/.zshrc || echo '$(EXPORT_HF_HOME)' >> ~/.zshrc

	@echo "echo '$(EXPORT_VENV)' >> ~/.zshrc"
	@grep -qxF '$(EXPORT_VENV)' ~/.zshrc || echo '$(EXPORT_VENV)' >> ~/.zshrc

	@echo ""
	@echo "If your current shell is listed below, follow the corresponding steps:"
	@echo "┌─────────┬──────────────────────────────────────────────┐"
	@echo "│ Shell   │ Commands                                     │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ zsh     │ ✔ Nothing to do                              │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ fish    │ zsh                                          │"
	@echo "│         │ source ~/.zshrc                              │"
	@echo "│         │ fish                                         │"
	@echo "├─────────┼──────────────────────────────────────────────┤"
	@echo "│ bash    │ zsh                                          │"
	@echo "│         │ source ~/.zshrc                              │"
	@echo "│         │ bash                                         │"
	@echo "└─────────┴──────────────────────────────────────────────┘"
	@echo ""

clean:
	@echo "Removing temporary files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf \
	.mypy_cache \
	src/.mypy_cache \
	llm_sdk/.mypy_cache \

fclean_goinfre: clean
	@echo "Removing the 'goinfre' project directory..."
	rm -rf $(GOINFRE_DIR)

fclean: clean
	@echo "Removing files created without setup_goinfre..."
	@rm -rf \
	.venv \
	~/.cache/uv \
	~/.cache/huggingface \
	~/.cache/transformers \
	~/.cache/torch

re_goinfre: fclean_goinfre install

re: fclean install