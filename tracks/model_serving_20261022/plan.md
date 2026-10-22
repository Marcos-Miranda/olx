# Plan: model_serving_20261022

Changed and new modules under `src/mlbox/`:

| Module | Change |
| ------ | ------ |
| `tracking.py` | Log the image as the run tag `mlbox.image`; look up a model version and the image of its training run |
| `runner.py` | Pass the image on to `tracking.start_run` |
| `serving.py` | New: the FastAPI app, the model loading and the `get_model` dependency |
| `docker.py` | Run the serving container |
| `main.py` | `serve` command; `run` passes the image to the runner; `show` prints it |

Tests keep using the local sqlite MLflow store. A shared fixture registers a small real model (features of type double, long, boolean and string, with a `predict` method returning one value per row and a `predict_proba` method returning two) through `Context.register_model`, inside a run created by `tracking.start_run`. So lookup and serving tests run against real persisted models. The container itself is verified end to end with the iris example (AC6).

## Block 1: Training image tracking
- [x] Test: `start_run` with an image logs it, and `get_run` returns it as `RunInfo.image` (extends the existing `start_run` test) (AC1).
- [x] `tracking.py`: `start_run(pipeline, params, seed, image: str)` also sets the tag `mlbox.image`. `RunInfo` gets `image: str | None` (`None` for runs created before this track).
- [x] `runner.py`: `Runner.run(pipeline, params, seed, image: str)` passes the image to `start_run`. Update the runner tests to pass one.
- [x] `main.py`: `run` passes the image returned by `build_pipeline_image` to the runner. `show` prints an `Image:` line.

## Block 2: Model version lookup
- [x] Test fixture (`conftest.py`): a factory that registers the small test model under a given name, in a run with a given image (or in a plain MLflow run without mlbox tags), and returns the version.
- [x] Test: `get_model_version(name)` returns the highest version and the image of its training run. With `version`, it returns that version.
- [x] Test: `get_model_version` raises `ValueError` with a clear message for an unknown model, an unknown version, and a version whose run has no `mlbox.image` tag.
- [x] `tracking.py`:
  - `ModelVersionInfo` dataclass: `name`, `version: int`, `image: str`.
  - `get_model_version(name: str, version: int | None = None) -> ModelVersionInfo`. Without `version`, it searches the model's versions and takes the highest number. With it, it reads that version. MLflow's "does not exist" errors become `ValueError`. It reads the image from the `mlbox.image` tag of the version's `run_id`.

## Block 3: Serving API
- [x] Add `fastapi` and `uvicorn` as runtime dependencies, and `httpx` as a dev dependency (FastAPI's `TestClient` needs it).
- [x] Test: with `MLBOX_MODEL_NAME`/`MLBOX_MODEL_VERSION` pointing at a registered model, a `TestClient` used as a context manager (so the lifespan runs) gets `{"status": "ok"}` from `/health` and the model's name, version and input schema from `/info`. In `/openapi.json`, the `/predict` request body schema has the model's input example, as records, among its `examples` (AC2, AC4).
- [x] Test: with `get_model` overridden by a model loaded through `load_served_model`, `/predict` with several instances, given as JSON integers, booleans, strings and floats, returns one list per instance: two floats for a model registered with `predict_proba`, one float for a model registered with `predict`. An integer given for a double feature is accepted (AC2).
- [x] Test: `/predict` returns 422 with MLflow's message for an instance missing a feature and for a value of the wrong type (a string for a double feature), and returns 422 for an empty array (AC3).
- [x] `serving.py`:
  - `ServedModel` dataclass: `name`, `version`, `model` (the MLflow `PyFuncModel`), `input_schema` (the signature's input columns as `[{"name", "type", "required"}]`) and `input_example` (records, or `None`).
  - `load_served_model(name: str, version: int) -> ServedModel`: loads `models:/<name>/<version>` with `mlflow.pyfunc.load_model` and reads the schema and the input example from it.
  - `PredictRequest`, a Pydantic `RootModel` over `list[dict[str, bool | int | float | str]]` with at least one item (`bool` first in the union so `true` stays a boolean). The body stays a plain JSON array.
  - `app = FastAPI(lifespan=...)`. The lifespan reads `MLBOX_MODEL_NAME` and `MLBOX_MODEL_VERSION` and loads the model into `app.state.model`. If the model has an input example, the lifespan adds it as `examples` to the `PredictRequest` schema of the OpenAPI document (FastAPI builds the body schema when the route is defined, so changing the Pydantic model later has no effect), so `/docs` shows it pre-filled.
  - `get_model(request: Request) -> ServedModel` returns `request.app.state.model`. Endpoints receive it with `Depends(get_model)`.
  - `GET /health` returns `{"status": "ok"}`. `GET /info` returns `name`, `version` and `input_schema` (Pydantic response model).
  - `POST /predict` takes a `PredictRequest` body. It builds a DataFrame from the records. Integer values in columns the schema declares `double`/`float` become floats, because JSON doesn't tell `5` from `5.0` and MLflow rejects `int64` for a `double` column. Other values go to MLflow as they are. It returns the prediction as `list[list[float]]`: the output turned into a float array and reshaped to one row per instance. An `MlflowException` from `predict` becomes an HTTP 422 with the error message.

## Block 4: `mlbox serve`
- [x] Test: `mlbox serve --model <unknown>` and `mlbox serve` for a model whose run has no `mlbox.image` tag exit non-zero with the lookup error, before touching Docker (AC5).
- [x] `docker.py`: `serve_model(model: ModelVersionInfo, port: int, uvicorn_args: list[str]) -> int`:
  - Raises `RuntimeError` if `model.image` doesn't exist locally (e.g. it was removed or rebuilt).
  - Runs a detached container from the image with entrypoint `uvicorn` and command `mlbox.serving:app --host 0.0.0.0 --port 8080 <uvicorn_args>`. The container gets the `mlbox` network, the host UID/GID, `HOME=/tmp`, `MLFLOW_TRACKING_URI` (as for steps), `MLBOX_MODEL_NAME` and `MLBOX_MODEL_VERSION`, and port 8080 published on `port`.
  - Streams its logs (prefix `[serve]`, reusing `_stream_logs`) until it exits or Ctrl+C. It always stops and removes the container, and returns the exit code (0 on Ctrl+C).
- [x] `main.py`: `serve(--model, --version, --port=8080)`, with Typer's `allow_extra_args` and `ignore_unknown_options` so the extra arguments (`ctx.args`) go to uvicorn. It looks up the version with `get_model_version` (exit 1 with the message on `ValueError`) and prints the model, version, image and `http://localhost:<port>`. It then calls `serve_model` (exit 1 on `RuntimeError`) and exits with its code.

## Block 5: Docs
- [x] `docs/project.md`:
  - A "Serving" section: `mlbox serve`, its options and uvicorn passthrough, the reuse of the training image (and that the base image must be rebuilt to include FastAPI), and the three endpoints with `curl` examples using iris rows.
  - A new row in the CLI table and the `mlbox.image` tag in the Tracking section.

## Final verification
- [ ] `uv run ruff check` and `uv run ruff format --check` pass
- [ ] `uv run mypy .` passes
- [ ] `uv run pytest` passes
- [ ] AC1: An execution's run has the `mlbox.image` tag (Block 1 test)
- [ ] AC2: `/health`, `/info` and `/predict` work for a registered model, including integer, boolean and string features (Block 3 tests)
- [ ] AC3: `/predict` returns 422 for a missing feature, a value of the wrong type and an empty array (Block 3 test)
- [ ] AC4: The `/predict` request body example in the OpenAPI schema is the model's input example (Block 3 test)
- [ ] AC5: Without `--version` the latest version is used, and `mlbox serve` fails clearly for an unknown model or a run without `mlbox.image` (Block 2 and Block 4 tests)
- [ ] AC6: After `docker build -t mlbox .`, `docker compose up -d` and `uv run mlbox run examples/iris/pipeline.py`, `uv run mlbox serve --model iris` serves on port 8080. `curl` on `/health`, `/info` and `/predict` (rows from `iris.csv`) returns ok, the model info and 3 class probabilities per row. `/docs` shows the input example, and Ctrl+C stops and removes the container
