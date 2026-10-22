# mlbox

mlbox manages the lifecycle of ML pipelines for data scientists who only write Python. A pipeline is a set of Python functions (*steps*). mlbox runs each step in its own resource-limited Docker container, tracks every execution in MLflow, and shows the result in the MLflow UI.

## Setup

Requirements: Docker, [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                    # install mlbox and its CLI
docker build -t mlbox .    # base image: Python + the mlbox library
docker compose up -d       # MLflow tracking server on http://localhost:5001
```

The CLI reaches MLflow at `MLFLOW_TRACKING_URI` (default `http://localhost:5001`). Step containers reach it through the `mlbox` Docker network, at `MLBOX_CONTAINER_TRACKING_URI` (default `http://mlflow:5000`).

## Running the example

```bash
uv run mlbox run examples/iris/pipeline.py -p dataset_path=iris.csv --seed 7
uv run mlbox list
uv run mlbox show <run_id>
```

Open <http://localhost:5001> to see the run in the `iris` experiment: its parameters, seed, metrics and the status of every step. The files written by the steps are in `.mlbox/runs/<run_id>/data`.

## Defining a pipeline

A pipeline file defines the parameters model, the steps and a single `Pipeline` ([examples/iris/pipeline.py](../examples/iris/pipeline.py)).

```python
class InputParams(BaseModel):
    dataset_path: str = "iris.csv"
    test_size: float = 0.2
    n_estimators: int = 100


@step()
def preprocess() -> dict[str, Path]:
    ctx = get_context(InputParams)
    ...
    return {"train": train_path, "test": test_path}


@step(cpus=2, memory="2g")
def train_random_forest(train_path: str) -> Path: ...


data = preprocess()
model = train_random_forest(data["train"])
pipeline = Pipeline("iris", InputParams, steps=[evaluate], requirements="requirements.txt")
```

### Steps

`@step(cpus=1, memory="1g")` turns a function into a step. `cpus` and `memory` are the container limits (Docker `--cpus` and `--memory`).

Calling a step at module level doesn't run it. It returns a reference to the step's future return value, and you can index it (`data["train"]`). Passing a reference to another step makes the second step depend on the first: it only runs after the first succeeded, and receives its value. Independent steps run in parallel.

- Arguments must be JSON literals (`dict`, `list`, `int`, `float`, `str`, `bool`, `None`) or references. Anything else is rejected.
- A step can be called only once.
- Return values must be JSON-serializable. `Path` values are accepted and become strings.

### Pipeline

`Pipeline(name, parameters, steps, requirements=None)` takes the final steps; their upstream steps are included automatically. `requirements` is a requirements file, relative to the pipeline file. The pipeline image is built from it once and reused while the file doesn't change.

The pipeline file is also imported on your machine, so libraries that are only in `requirements.txt` must be imported inside the steps.

### Context

Inside a step, `get_context(InputParams)` returns:

| Attribute | Description |
| --------- | ----------- |
| `input_params` | The parameters, validated against the model |
| `data_path` | Directory shared by all steps of the execution, for passing files between them |
| `mlflow` | The `mlflow` module, with the execution's run already active (plain `mlflow.log_metric(...)` works too) |
| `random_seed` | The execution's seed. `random` and NumPy are seeded with it before the step runs |

Outside a step execution, `get_context()` raises.

## Container contract

Each step runs in a container created from the pipeline image, following the SageMaker `/opt/ml` layout.

| Path | Content |
| ---- | ------- |
| `/opt/ml/code` | The pipeline project directory, read-only. It is the working directory, so `data/iris.csv` works |
| `/opt/ml/input/config/hyperparameters.json` | The pipeline parameters |
| `/opt/ml/input/config/step.json` | The step name, its resolved arguments, the seed and the MLflow run ID |
| `/opt/ml/data` | The execution's shared data directory (`Context.data_path`) |
| `/opt/ml/output/result.json` | Written on success, with the step's return value |
| `/opt/ml/output/failure` | Written on failure, with the error and traceback |

The entrypoint is the `mlbox` step worker (`python -m mlbox.worker`). On the host, everything lives in `.mlbox/runs/<run_id>/`: `data/` and, for each step, `steps/<step>/config` and `steps/<step>/output`.

If a step fails, no new steps start, running steps finish and the steps that never started are marked `skipped`.

## Tracking

Each execution is one MLflow run in an experiment named after the pipeline. Its ID is the mlbox run ID. The parameters and the seed are logged as run params, and metrics logged by the steps land in the same run.

MLflow is also where the execution state lives: the run status is `RUNNING`, then `FINISHED` or `FAILED`, and each step's status (`pending`, `running`, `succeeded`, `failed`, `skipped`) is the run tag `mlbox.step.<name>.status`. Runs created by mlbox carry the tag `mlbox.pipeline`, and the tag `mlbox.image` with the image their steps ran in (so a model trained by the run can be served with it).

## Serving

```bash
uv run mlbox serve --model iris                  # latest version, on http://localhost:8080
uv run mlbox serve --model iris --version 2 --port 9000 --workers 2
```

`mlbox serve` serves a model registered in MLflow (the one a pipeline registers with `ctx.register_model`, named after the pipeline). Without `--version` it serves the latest version, the highest number. Any extra argument goes to uvicorn (`--workers 2` above). It runs in the foreground, streaming the container logs, until Ctrl+C, which stops and removes the container.

The API runs in a container created from the **image that trained the model**, found in the `mlbox.image` tag of the training run, so serving uses the same dependencies as training. The container joins the `mlbox` network, reaches MLflow like a step does, loads the model at startup and publishes its port 8080 on `--port` (default 8080). The command exits with an error, without starting a container, if the model or version doesn't exist, if its run has no `mlbox.image` tag (a model registered outside mlbox) or if the image is no longer on your machine. The API needs FastAPI, which is part of the base image: rebuild it (`docker build -t mlbox .`) and run the pipeline again to serve models trained before it was added.

### Endpoints

`GET /health` returns `{"status": "ok"}`.

`GET /info` returns the model name, version and input schema (the MLflow signature):

```bash
curl localhost:8080/info
```

```json
{
  "name": "iris",
  "version": 1,
  "input_schema": [
    {"name": "sepal_length", "type": "double", "required": true},
    {"name": "sepal_width", "type": "double", "required": true},
    {"name": "petal_length", "type": "double", "required": true},
    {"name": "petal_width", "type": "double", "required": true}
  ]
}
```

`POST /predict` takes a JSON array with one object per instance, mapping each feature to its value, and returns one list of floats per instance: a single value for a regressor, one probability per class for a classifier.

```bash
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
       {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}]'
```

```json
[[1.0, 0.0, 0.0], [0.0, 0.01, 0.99]]
```

Values are checked against the model's schema by MLflow. An instance with a missing feature or a value of the wrong type (a string for a number) is rejected with HTTP 422 and MLflow's message, and so is an empty array. An integer is accepted for a `double` feature.

Interactive docs are at <http://localhost:8080/docs>, with the model's input example pre-filled in the `/predict` request body.

## CLI

| Command | Description |
| ------- | ----------- |
| `mlbox run <pipeline_file> [-p key=value ...] [--seed N]` | Validate the parameters, build the image and run the pipeline. Streams logs prefixed with the step name, prints the run ID and the final status, and exits non-zero on failure. Without `--seed`, one is generated and logged |
| `mlbox list [--pipeline NAME]` | List runs, newest first: run ID, pipeline, status, start time and duration |
| `mlbox show <run_id>` | Show a run's pipeline, status, image, parameters, seed and the status of each step |
| `mlbox serve --model NAME [--version N] [--port PORT] [UVICORN_ARGS...]` | Serve a registered model through a REST API. See [Serving](#serving) |

Parameter values are parsed as JSON when possible (`test_size=0.3`, `n_estimators=50`) and as strings otherwise (`dataset_path=iris.csv`).
