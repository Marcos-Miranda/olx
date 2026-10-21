# AGENTS.md

## Project

A tool that manages the ML model lifecycle for data scientists who only write Python. It runs training in isolated, reproducible Docker containers (SageMaker-like jobs and steps, following the `/opt/ml` container contract), tracks runs, metrics and artifacts in MLflow, and serves trained models through a generic REST API (following the SageMaker inference contract).

> [!IMPORTANT]
> This is a demo project for a technical case, not a production system. Keep implementations simple and focused on the end-to-end flow. Edge cases and production concerns (scale, security, hardening) are low priority unless explicitly requested.

## Tooling

- `uv` manages the project and its dependencies: `uv add <pkg>` (`--dev` for dev tools) and `uv run <cmd>`. Never use pip or edit dependency lists by hand.
- After every implementation, run these and fix what they report:
  ```bash
  uv run ruff check --fix && uv run ruff format
  uv run mypy .
  ```
- Tests: `uv run pytest`.

## Rules

- Public classes, functions and methods have docstrings; private (`_`-prefixed) ones don't need them.
- Tests check behavior (return values, persisted state, API responses), never whether or how something was called.
- Test names describe the behavior under test, e.g. `test_predict_rejects_an_instance_missing_a_feature`.
- Test only critical paths. Don't retest third-party libraries, thin wrappers or trivial code.
