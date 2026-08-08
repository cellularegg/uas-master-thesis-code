# Coding guidelines

These guidelines complement the project setup and workflow described in
`README.md`.

## Python

- Use Python 3.12 and manage dependencies and commands with `uv`.
- Keep reusable application logic in `src/`; use notebooks for pipeline
  orchestration and analysis.
- Add type annotations to new and modified code, and validate them with
  `uv run mypy src tests`.
- Run Ruff formatting and linting with `make format` and `make lint`.

## Tests

- Write tests with pytest for new or changed behavior.
- Isolate external APIs in tests by using fakes or mocks.
- Run `make test` before submitting changes.

## Notebooks and data

- Preserve the established notebook execution order.
- Do not commit notebook outputs; use the configured nbstripout hook.
- Normalize timestamps to timezone-aware UTC values when loading or passing
  time-based data between pipeline stages.

## Docstrings and comments

- Use Google-style docstrings for public modules, classes, functions, and
  methods. Include `Args:`, `Returns:`, and `Raises:` sections when applicable.
- Prefer clear code over comments. Use comments to explain non-obvious
  reasoning, assumptions, or workarounds.
- Keep docstrings and comments accurate when behavior changes, and do not
  leave commented-out code behind.

## Before submitting

- Run `make format`, `make lint`, `make test`, and `make hooks` as appropriate
  for the changes.
