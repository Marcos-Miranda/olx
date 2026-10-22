# Spec: model_serving_20261022

Serve a model registered in MLflow through a FastAPI REST API, started by `mlbox serve` in a container created from the same image that trained the model, so serving uses the same dependencies as training.

## Functional requirements

### Training image tracking
- FR1: When an execution starts, the image its steps run in (e.g. `mlbox-iris:<digest>` or `mlbox:latest`) is logged as the run tag `mlbox.image`.

### REST API
- FR2: `mlbox.serving` defines a FastAPI app. On startup (lifespan) it loads the registered model given by the environment variables `MLBOX_MODEL_NAME` and `MLBOX_MODEL_VERSION` from MLflow (`MLFLOW_TRACKING_URI`), and keeps the model with its name, version, input schema and input example.
- FR3: Endpoints get the loaded model through a FastAPI dependency, so tests can replace it with `app.dependency_overrides`.
- FR4: `GET /health` returns `{"status": "ok"}`.
- FR5: `GET /info` returns the model's name, version and input schema (the MLflow signature's input columns: name, type and whether it is required).
- FR6: `POST /predict` takes a JSON array with one or more instances, each an object mapping feature names to values (`list[dict[str, bool | int | float | str]]`), with each value's JSON type kept as is so MLflow checks it against the schema. The one exception is an integer given for a `double`/`float` feature, which becomes a float, because JSON doesn't tell `5` from `5.0`. It runs the model on them and returns one list of floats per instance (`list[list[float]]`): a single value for models like regressors, one value per class for probabilities.
- FR7: Input validation is left to MLflow's schema enforcement: an `MlflowException` raised by the prediction (e.g. a missing feature) becomes an HTTP 422 with the error message. An empty array is rejected with 422 too.
- FR8: The model's input example (the records stored with the model) is the example of the `/predict` request body in the OpenAPI schema, so it shows up pre-filled in `/docs`. A model without an input example has no example.

### CLI
- FR9: `mlbox serve --model NAME [--version N] [--port PORT] [UVICORN_ARGS...]` serves the version `N` of the registered model `NAME`, or its latest version (the highest number) when `--version` is omitted.
- FR10: It finds the training run of that version and the image in its `mlbox.image` tag, and runs a container from that image with `uvicorn mlbox.serving:app --host 0.0.0.0 --port 8080` plus the extra arguments (e.g. `--workers 2`), on the `mlbox` network with `MLFLOW_TRACKING_URI` set as for training steps. Container port 8080 is published on host port `PORT` (default 8080).
- FR11: It runs in the foreground, streaming the container logs, until interrupted (Ctrl+C), then stops and removes the container.
- FR12: It exits non-zero with a clear message, without starting a container, when the model or version doesn't exist, when the training run has no `mlbox.image` tag (e.g. a model registered outside mlbox) or when the image doesn't exist locally.

### Dependencies and docs
- FR13: `fastapi` and `uvicorn` are runtime dependencies of mlbox, so they are installed in the base image and in every pipeline image.
- FR14: `docs/project.md` documents serving: the `mlbox serve` command, the endpoints with request and response examples, and the use of the training image.

## Acceptance criteria
- AC1: After an execution, its MLflow run has the tag `mlbox.image` with the image the steps ran in.
- AC2: Given a registered model, `/health` returns ok, `/info` returns its name, version and input schema, and `/predict` with several valid instances returns one list of floats per instance. This holds for a model whose features are integers, booleans and strings too.
- AC3: `/predict` returns 422 for an instance missing a feature, for a value of the wrong type (e.g. a string for a float feature) and for an empty array.
- AC4: The OpenAPI schema of `/predict` has the model's input example as the request body example.
- AC5: Without `--version`, the latest version of the model is the one served. `mlbox serve` for an unknown model, or for a version whose run has no `mlbox.image` tag, exits non-zero with a clear message.
- AC6: End to end: after `docker build -t mlbox .`, `docker compose up -d` and a run of the iris example, `uv run mlbox serve --model iris` starts the API on port 8080. `curl` on `/health`, `/info` and `/predict` (with rows from `iris.csv`) returns ok, the model info and one list of 3 class probabilities per row. `/docs` shows the input example, and Ctrl+C stops and removes the container.
