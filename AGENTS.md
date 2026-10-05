# Development Guidelines for Deep Event Research

## Build/Test Commands
- **Run all tests**: `make test` or `uv run pytest -v -s`
- **Run single test**: `uv run pytest src/test/test_file.py::test_function -v`
- **Run tests without LLM calls**: `uv run pytest -v -m 'not llm'`
- **Run LLM integration tests**: `uv run pytest -v -m llm`
- **Lint code**: `uv run ruff check src/`
- **Format code**: `uv run ruff format src/`
- **Start dev server**: `make dev`

## Code Style Guidelines
- **Python**: 3.12+ with type hints required
- **Imports**: Use `from src.module import name` for internal imports, standard library first
- **Formatting**: Ruff with Google docstring convention
- **Error handling**: Use `@with_error_handling` decorator for graph nodes, raise `GraphError` for known failures
- **Async**: All graph functions must be async and return `Command`
- **Testing**: Use pytest with asyncio mode, mock LLM calls by default, mark real LLM tests with `@pytest.mark.llm`
- **State management**: Use TypedDict classes from `src.state.py` for all state objects
- **Services**: Static methods in service classes, no instance state