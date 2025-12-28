.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests dev

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/

test:
	uv run --with-editable . pytest $(TEST_FILE) -k "not trio"

test_watch:
	uv run --with-editable . ptw --snapshot-update --now . -- -vv tests/

test_profile:
	uv run --with-editable . pytest -vv tests/ --profile-svg

extended_tests:
	uv run --with-editable . pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	uv run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || uv run ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_mcp                     - test MCP stdio server (simple)'
	@echo 'test_mcp_full                - test full MCP stdio protocol'
	@echo 'dev                          - start LangGraph development server'
	@echo 'local                        - start MCP stdio server'

dev:
	@echo "Starting development server..."
	@uv run langgraph dev

local:
	@echo "Starting local MCP server..."
	@uv run python main.py

test_mcp:
	@echo "Testing MCP stdio server..."
	@uv run python tests/test_simple_mcp.py

inspect:
	@echo "Inspecting local MCP server..."
	@npx @modelcontextprotocol/inspector uv --directory `pwd` run python main.py
