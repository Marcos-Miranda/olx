# Spec: mlbox_pipelines_20261021

Build the core of `mlbox`: an SDK to define ML pipelines as Python functions (steps), a runner that executes each step in its own resource-limited Docker container following an `/opt/ml` contract, MLflow tracking of each pipeline execution, and a CLI to run pipelines and list their runs.

## Functional requirements

### SDK: steps
- FR1: `@step(cpus: float = 1, memory: str = "1g")` turns a function into a `Step`. `cpus` and `memory` are the container limits (Docker `--cpus` / `--memory` semantics, e.g. `memory="4g"`). The step name is the function name.
- FR2: Calling a `Step` at module level does not run it. It binds the call's arguments and returns a `StepOutput` reference to the step's future return value. Indexing a reference (`data["train"]`, `items[0]`) returns a reference to that part of the value. A step may be called only once.
- FR3: Step arguments (positional or keyword) are JSON literals (`dict`, `list`, `int`, `float`, `str`, `bool`, `None`) or `StepOutput` references. Passing a reference creates a dependency on the referenced step. Any other argument type is rejected when the step is called.
- FR4: A step's return value must be JSON-serializable. `pathlib.Path` values are accepted and serialized as strings. A return value that can't be serialized fails the step with a clear error.

### SDK: pipeline
- FR5: `Pipeline(name: str, parameters: type[BaseModel], steps: list[Step], requirements: str | None = None)` defines a pipeline. `steps` must contain at least the final steps; their upstream dependencies are included automatically. Every listed step must have been called. `requirements` is a path to a requirements file, relative to the pipeline file's directory.
- FR6: The pipeline exposes its execution graph: which steps depend on which, and an order in which they can run.

### SDK: context
- FR7: `get_context()` returns the `Context` of the current step execution. It raises a clear error when called outside a step execution. `get_context(InputParams)` returns a `Context[InputParams]`, typed for the pipeline's parameter model.
- FR8: `Context` exposes:
  - `input_params`: the pipeline's parameters model, validated from the values given at run time.
  - `data_path`: a `pathlib.Path` to a directory shared by all steps of one execution. Each execution gets its own directory, and files written there by one step can be read by later steps.
  - `mlflow`: the `mlflow` module, with the execution's MLflow run already active, so both `ctx.mlflow.log_metric(...)` and a plain `mlflow.log_metric(...)` log to that run.
  - `random_seed`: the execution's integer seed. Before a step runs, Python's `random` and NumPy's global RNG (if NumPy is installed) are seeded with it.

### Execution
- FR9: A `Dockerfile` at the repository root builds the base image `mlbox:latest`: Python plus the `mlbox` library. Before an execution, the runner builds a pipeline image from it with the pipeline's requirements installed. The image is tagged with a hash of the requirements file, so Docker reuses it while the file doesn't change.
- FR10: Each step runs in its own container, created from the pipeline image with the step's CPU and memory limits. Container contract:
  - `/opt/ml/code`: the pipeline project directory (the pipeline file's directory), read-only. This is the working directory, so paths relative to the project (e.g. `data/iris.csv`) work.
  - `/opt/ml/input/config/hyperparameters.json`: the pipeline parameters.
  - `/opt/ml/input/config/step.json`: the step name, its resolved arguments, the random seed and the MLflow run ID.
  - `/opt/ml/data`: the execution's shared data directory (`Context.data_path`).
  - `/opt/ml/output/result.json`: written by the container with the step's return value on success.
  - `/opt/ml/output/failure`: written by the container with the error and traceback on failure (SageMaker convention).
  The container's entrypoint is the `mlbox` step worker, which imports the pipeline file, runs the step's function with its arguments, and writes the output files.
- FR11: The runner runs a step as soon as all its dependencies have succeeded. Independent steps run in parallel. Before a step starts, the references in its arguments are replaced with the matching values from upstream results.
- FR12: If a step fails, the runner starts no new steps, waits for the running ones to finish, marks the steps that never started as `skipped` and the execution as failed.
- FR13: Host-side data for an execution lives in `.mlbox/runs/<run_id>/` under the current directory: the shared data directory and each step's output directory.

### Tracking
- FR14: A `docker-compose.yml` runs an MLflow tracking server with artifact serving. Step containers reach it on a shared Docker network. The server's port is published on host port 5001 (5000 conflicts with AirPlay on macOS), and the host reaches it at `MLFLOW_TRACKING_URI` (default `http://localhost:5001`).
- FR15: Each execution is one MLflow run in an experiment named after the pipeline. The run is created before the first step starts, and its ID is the mlbox run ID. The pipeline parameters and the random seed are logged as run params.
- FR16: MLflow is the store of execution state: the run status is `RUNNING` during the execution and `FINISHED` or `FAILED` at the end, and each step's status (`pending`, `running`, `succeeded`, `failed`, `skipped`) is a run tag (`mlbox.step.<name>.status`). Runs created by mlbox are tagged so they can be told apart from other MLflow runs.

### CLI
- FR17: `mlbox run <pipeline_file> [-p key=value ...] [--seed N]` loads the single `Pipeline` defined in the file, validates the parameters against its model (reporting validation errors without starting anything), builds the image and runs the pipeline synchronously. Container logs are streamed with the step name as prefix. It prints the run ID at start and the final status at the end, and exits non-zero if the execution fails. Without `--seed`, a random seed is generated (and logged, per FR15).
- FR18: `mlbox list [--pipeline NAME]` lists mlbox runs, newest first: run ID, pipeline, status, start time and duration.
- FR19: `mlbox show <run_id>` shows a run's details: pipeline, status, parameters, seed and the status of each step.

### Example
- FR20: `examples/iris/` contains a runnable pipeline (`pipeline.py`, `requirements.txt`, a small CSV dataset) with a preprocess step, two independent steps that run in parallel, and a final evaluate step that logs metrics to MLflow.

## Acceptance criteria
- AC1: Defining a pipeline like the task's example (`data = preprocess()`, `train(data["train"])`, `Pipeline(..., steps=[train])`) runs no step code and produces a graph where `train` depends on `preprocess`.
- AC2: Calling a step with an argument that is neither JSON nor a `StepOutput` raises an error, and so does calling the same step twice.
- AC3: The step worker, given an `/opt/ml`-style directory, runs the step's function with its arguments and writes `result.json`. A function that raises or returns a non-JSON value produces a `failure` file instead.
- AC4: Inside a step, `get_context()` returns the validated parameters, the shared data path, the seed and an active MLflow run. Outside a step, it raises.
- AC5: The runner runs independent steps concurrently, starts a step only after its dependencies succeed, passes it the resolved upstream values, and on a failure skips the steps that haven't started and marks the execution failed.
- AC6: `mlbox run` with invalid parameters exits non-zero with the validation error and creates no run.
- AC7: With `docker compose up` running, `uv run mlbox run examples/iris/pipeline.py -p ...` completes successfully. The MLflow UI shows a run in the `iris` experiment with the params, seed, metrics and every step tagged `succeeded`, and the files written by the steps are in `.mlbox/runs/<run_id>/data`.
- AC8: `mlbox list` shows that run as `FINISHED`, and `mlbox show <run_id>` shows each step as `succeeded`. A run in which a step fails shows as `FAILED`, with the failed step and the skipped steps.
- AC9: Running the same pipeline twice with the same parameters and seed logs the same metrics.
