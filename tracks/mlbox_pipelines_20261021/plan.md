# Plan: mlbox_pipelines_20261021

Modules under `src/mlbox/`:

| Module | Responsibility |
| ------ | -------------- |
| `step.py` | `step` decorator, `Step`, `StepOutput` references, JSON checks |
| `pipeline.py` | `Pipeline`, execution graph, loading a pipeline from a file |
| `tracking.py` | All MLflow access: creating runs, step statuses, listing runs |
| `context.py` | `Context`, `get_context` |
| `worker.py` | Container entrypoint that runs one step under the `/opt/ml` contract |
| `runner.py` | Orchestration of an execution over an abstract executor |
| `docker.py` | Pipeline image build and the Docker executor |
| `main.py` | Typer CLI |

`mlbox/__init__.py` re-exports the user-facing API: `step`, `Step`, `StepOutput`, `Pipeline`, `Context`, `get_context`.

Tests use a local MLflow store (sqlite in `tmp_path`, set through `MLFLOW_TRACKING_URI`), so they check real persisted state without the compose server or Docker. Docker execution is verified end to end by the example (AC7–AC9).

## Block 1: Steps and pipeline graph
- [ ] Add dependencies: `pydantic`, `mlflow-skinny` (tracking client only, keeps the image small) and `docker`. Add `mlflow` as a dev dependency so tests can use a local sqlite store.
- [ ] Test: defining steps and calling them at module level runs no step code, and a pipeline built from only the final step includes its upstream steps, with `train` depending on `preprocess` and `preprocess` coming first in the execution order (AC1).
- [ ] Test: calling a step with an argument that is neither JSON nor a `StepOutput` raises, calling a step twice raises, and a pipeline listing a step that was never called raises (AC2).
- [ ] Test: a step's bound arguments resolve their references, including indexed ones like `data["train"]`, against a mapping of upstream results.
- [ ] `step.py`:
  - `step(cpus: float = 1, memory: str = "1g") -> Callable[[Callable], Step]`.
  - `Step` keeps the function (`func`), `name`, `cpus` and `memory`, plus the arguments bound by its single call. `__call__` checks every argument (JSON literal or `StepOutput`, nested in lists and dicts too), stores them and returns `StepOutput(step)`. `dependencies` returns the steps referenced by the arguments. `resolve_arguments(results: dict[str, Any]) -> tuple[list, dict]` replaces references with values.
  - `StepOutput(step, path: tuple[str | int, ...] = ())`: `__getitem__` returns a new reference with the key appended. `resolve(results)` takes the step's result and walks the path.
  - `to_json_value(value) -> Any`: converts `Path` to `str` (recursively) and raises if the result isn't JSON-serializable. Used on step return values.
- [ ] `pipeline.py`:
  - `Pipeline(name, parameters, steps, requirements=None)`: collects the listed steps and their transitive dependencies, raises if a listed step was never called or if two steps share a name. Exposes `steps: dict[str, Step]` (in execution order, a topological sort) and `dependencies(name) -> set[str]`.
- [ ] Test: `load_pipeline` returns the only `Pipeline` defined in a file, and raises when the file defines none or several.
- [ ] `load_pipeline(path: Path) -> Pipeline`: imports the file as a module (adding its directory to `sys.path` so sibling modules can be imported) and returns the single `Pipeline` instance among its globals.

## Block 2: MLflow tracking
- [ ] Test: `start_run` creates a `RUNNING` run in the experiment named after the pipeline, with the parameters and the seed as params and every step `pending`.
- [ ] Test: step status changes and `end_run` are reflected by `get_run`, and `list_runs` returns only mlbox runs, newest first, filtered by pipeline when asked.
- [ ] `tracking.py`, using `MlflowClient` (no fluent active run on the host side):
  - The tracking URI comes from `MLFLOW_TRACKING_URI`, defaulting to `http://localhost:5001` when it isn't set.
  - `start_run(pipeline: Pipeline, params: BaseModel, seed: int) -> str`: creates the experiment if needed, creates the run, logs the parameters (`model_dump(mode="json")`) and `random_seed` as params, tags `mlbox.pipeline=<name>` and `mlbox.step.<name>.status=pending` for each step, and returns the run ID.
  - `set_step_status(run_id, step_name, status)` and `end_run(run_id, succeeded: bool)` (`FINISHED` or `FAILED`).
  - `list_runs(pipeline: str | None = None) -> list[RunInfo]` and `get_run(run_id) -> RunInfo`. `RunInfo` is a dataclass with run ID, pipeline, status, start and end time, params, seed and `step_statuses: dict[str, str]`. Runs without the `mlbox.pipeline` tag are ignored.

## Block 3: Context and step worker
- [ ] Test: `get_context()` outside a step execution raises.
- [ ] Test: given an `/opt/ml`-style directory (code, config, data, output) under `tmp_path` and an MLflow run, the worker runs the named step with its arguments and writes `result.json` (with `Path` values as strings). Inside the step, `get_context()` returns the validated parameters, the data path and the seed, `random` is seeded with it, and a metric logged with plain `mlflow.log_metric` lands in the run, which is still `RUNNING` afterwards (AC3, AC4).
- [ ] Test: a step that raises, or returns a non-JSON value, makes the worker write `output/failure` with the error, write no `result.json` and return a non-zero exit code (AC3).
- [ ] `context.py`:
  - `Context[P: BaseModel]`, a frozen dataclass with `input_params: P`, `data_path: Path`, `random_seed: int` and an `mlflow` property returning the `mlflow` module.
  - The current context lives in a `ContextVar`. `get_context(params_type: type[P] | None = None)` returns it (typed with overloads) or raises outside a step. It also raises if `params_type` isn't the pipeline's parameter model. A private `_use_context(ctx)` context manager sets it for the worker.
- [ ] `worker.py`:
  - `run_step(root: Path = Path("/opt/ml")) -> int`: reads `input/config/step.json` (pipeline file name, step name, args, kwargs, seed, run ID) and `hyperparameters.json`, loads the pipeline from `root/code`, validates the parameters, seeds `random` and NumPy (if importable), resumes the MLflow run with `mlflow.start_run(run_id=...)`, runs `step.func(*args, **kwargs)` inside `_use_context`, converts the result with `to_json_value` and writes `output/result.json`. It then ends the fluent run with `RUNNING` status so the runner stays in control of the final status. Any exception writes the traceback to `output/failure` and returns 1.
  - `python -m mlbox.worker` calls `run_step()` and exits with its return code.

## Block 4: Runner
- [ ] Test: with a fake executor recording start and end times, independent steps overlap, a step starts only after its dependencies finished and receives their resolved values, `run` returns the run ID, and MLflow shows the run `FINISHED` with every step `succeeded` (AC5).
- [ ] Test: when a step fails, steps already running finish, steps not started are `skipped`, the failed step is `failed` and the run is `FAILED` (AC5).
- [ ] `runner.py`:
  - `StepJob` dataclass: run ID, step, resolved args and kwargs, the parameters JSON, seed and the host run directory. `StepResult` dataclass: `succeeded`, `value`, `error`.
  - `Executor` protocol with `execute(job: StepJob) -> StepResult`.
  - `Runner(executor, runs_dir: Path = Path(".mlbox/runs"))` with `run(pipeline, params, seed) -> str`. It creates the MLflow run, creates `runs_dir/<run_id>/data`, then loops: submits every step whose dependencies all succeeded to a `ThreadPoolExecutor` (marking it `running`), waits for the first completion, records the result and the step status. After a failure it submits nothing else, waits for the running steps, marks the rest `skipped` and ends the run.

## Block 5: Docker execution
- [ ] `Dockerfile` at the repo root: `python:3.13-slim` with `uv`, installs the `mlbox` package from `pyproject.toml`/`uv.lock`/`src` (without dev dependencies), `WORKDIR /opt/ml/code`, `ENTRYPOINT ["python", "-m", "mlbox.worker"]`. Built as `mlbox:latest`.
- [ ] `docker-compose.yml`: an `mlflow` service (official MLflow image) running `mlflow server` with a sqlite backend and served artifacts on a named volume, listening on 5000 inside the network and published on host port 5001, requests addressed to the `mlflow` hostname allowed. It sits on a network explicitly named `mlbox`.
- [ ] `docker.py`:
  - `build_pipeline_image(pipeline: Pipeline, pipeline_dir: Path) -> str`: without requirements, returns `mlbox:latest`. Otherwise returns `mlbox-<pipeline>:<first 12 chars of the requirements sha256>`, building it first if it doesn't exist, from an in-memory build context with a generated Dockerfile (`FROM mlbox:latest`, copy the requirements, `uv pip install --system -r`). Raises a clear error if `mlbox:latest` is missing, telling the user to run `docker build -t mlbox .`.
  - `DockerExecutor(image, pipeline_file: Path)` implementing `Executor`: writes `step.json` and `hyperparameters.json` to `<run_dir>/steps/<step>/config`, runs a container with the step's `cpus`/`memory` limits, the host UID/GID (so the files in `.mlbox` stay owned by the user), `HOME=/tmp`, the `mlbox` network and `MLFLOW_TRACKING_URI` (from `MLBOX_CONTAINER_TRACKING_URI`, default `http://mlflow:5000`), and these mounts: project directory at `/opt/ml/code` (read-only), config dir at `/opt/ml/input/config`, run data dir at `/opt/ml/data`, step output dir at `/opt/ml/output`. It streams the container logs as `[<step>] <line>`, waits for exit, removes the container and builds the `StepResult` from `result.json` or `failure`. If neither exists, the error is the exit code (e.g. 137 when the memory limit was hit).

## Block 6: CLI
- [ ] Test: `mlbox run` with parameters that fail the pipeline's model exits non-zero, prints the validation error and creates no MLflow run (AC6).
- [ ] `main.py` (remove the `hello` placeholder):
  - `run(pipeline_file, -p/--param key=value (repeatable), --seed)`: loads the pipeline, parses each value as JSON when possible and as a string otherwise, validates the parameters with the model (exit 1 on error), generates a seed if none was given, builds the image, runs the `Runner` with the `DockerExecutor`, prints the run ID and the final status, and exits 1 if the run failed.
  - `list(--pipeline)`: a table of run ID, pipeline, status, start time and duration.
  - `show(run_id)`: pipeline, status, parameters, seed and one line per step with its status.

## Block 7: Example and docs
- [ ] `examples/iris/`: `iris.csv` (the Iris dataset with a `target` column), `requirements.txt` (`pandas`, `scikit-learn`, `joblib`) and `pipeline.py` with parameters (`dataset_path`, `test_size`, `n_estimators`) and these steps:
  - `preprocess`: splits the dataset into train and test CSVs in `data_path` and returns their paths.
  - `train_random_forest` and `train_logistic_regression`: run in parallel, each saves a model trained with `random_state=ctx.random_seed` and returns its path.
  - `evaluate`: logs each model's test accuracy as an MLflow metric and returns them.
- [ ] Add `.mlbox/` to `.gitignore`.
- [ ] `docs/project.md` (the README is left untouched): what mlbox is, setup (`docker build -t mlbox .`, `docker compose up -d`), defining a pipeline (steps, references, context), the `/opt/ml` container contract, and the CLI commands, using the iris example.

## Final verification
- [ ] `uv run ruff check` and `uv run ruff format --check` pass
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest` passes
- [ ] AC1: Defining a pipeline like the task's example runs no step code, and `train` depends on `preprocess` (Block 1 test)
- [ ] AC2: Non-JSON arguments and calling a step twice raise (Block 1 test)
- [ ] AC3: The worker writes `result.json`, or `failure` for a raising or non-JSON step (Block 3 tests)
- [ ] AC4: `get_context()` works inside a step and raises outside one (Block 3 tests)
- [ ] AC5: The runner runs steps in parallel, resolves values and handles failures (Block 4 tests)
- [ ] AC6: `mlbox run` with invalid parameters exits non-zero and creates no run (Block 6 test)
- [ ] AC7: With `docker compose up -d`, `uv run mlbox run examples/iris/pipeline.py -p dataset_path=iris.csv` succeeds. The MLflow UI shows the run in the `iris` experiment with params, seed, metrics and every step `succeeded`, and the step files are in `.mlbox/runs/<run_id>/data`
- [ ] AC8: `mlbox list` shows the run `FINISHED` and `mlbox show` shows every step `succeeded`. A run with a failing step (e.g. a `dataset_path` that doesn't exist) shows `FAILED`, with the failed and skipped steps
- [ ] AC9: Two runs of the example with the same parameters and `--seed` log the same metrics
