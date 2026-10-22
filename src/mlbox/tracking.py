import os
from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities import Run, RunStatus
from pydantic import BaseModel

from mlbox.pipeline import Pipeline

PIPELINE_TAG = "mlbox.pipeline"
IMAGE_TAG = "mlbox.image"
SEED_PARAM = "random_seed"
_STEP_TAG_PREFIX = "mlbox.step."
_STEP_TAG_SUFFIX = ".status"


@dataclass
class RunInfo:
    """The state of one mlbox execution, as stored in MLflow."""

    run_id: str
    pipeline: str
    status: str
    start_time: int
    end_time: int | None
    params: dict[str, str]
    seed: int
    step_statuses: dict[str, str]
    image: str | None


def start_run(pipeline: Pipeline, params: BaseModel, seed: int, image: str) -> str:
    """Create the MLflow run of an execution, with every step pending, and return its ID.

    The image the steps run in is stored as the `mlbox.image` tag, so a model trained by the run can be served with it.
    """
    client = _client()
    experiment = client.get_experiment_by_name(pipeline.name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(pipeline.name)
    tags = {PIPELINE_TAG: pipeline.name, IMAGE_TAG: image} | {_step_tag(name): "pending" for name in pipeline.steps}
    run_id = client.create_run(experiment_id, tags=tags).info.run_id
    for key, value in params.model_dump(mode="json").items():
        client.log_param(run_id, key, value)
    client.log_param(run_id, SEED_PARAM, seed)
    return run_id


def set_step_status(run_id: str, step_name: str, status: str) -> None:
    """Record the status of a step (`pending`, `running`, `succeeded`, `failed` or `skipped`)."""
    _client().set_tag(run_id, _step_tag(step_name), status)


def end_run(run_id: str, succeeded: bool) -> None:
    """Mark the run as `FINISHED` or `FAILED`."""
    _client().set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED if succeeded else RunStatus.FAILED))


def get_run(run_id: str) -> RunInfo:
    """Return the state of one execution."""
    return _to_run_info(_client().get_run(run_id))


def list_runs(pipeline: str | None = None) -> list[RunInfo]:
    """Return the mlbox runs, newest first, optionally only those of one pipeline."""
    client = _client()
    experiment_ids = [experiment.experiment_id for experiment in client.search_experiments()]
    if not experiment_ids:
        return []
    runs = client.search_runs(experiment_ids, order_by=["attributes.start_time DESC"], max_results=1000)
    infos = [_to_run_info(run) for run in runs if PIPELINE_TAG in run.data.tags]
    return [info for info in infos if pipeline is None or info.pipeline == pipeline]


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))


def _step_tag(step_name: str) -> str:
    return f"{_STEP_TAG_PREFIX}{step_name}{_STEP_TAG_SUFFIX}"


def _to_run_info(run: Run) -> RunInfo:
    tags = run.data.tags
    params = dict(run.data.params)
    step_statuses = {
        key.removeprefix(_STEP_TAG_PREFIX).removesuffix(_STEP_TAG_SUFFIX): value
        for key, value in tags.items()
        if key.startswith(_STEP_TAG_PREFIX) and key.endswith(_STEP_TAG_SUFFIX)
    }
    return RunInfo(
        run_id=run.info.run_id,
        pipeline=tags[PIPELINE_TAG],
        status=run.info.status,
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        params=params,
        seed=int(params.pop(SEED_PARAM)),
        step_statuses=step_statuses,
        image=tags.get(IMAGE_TAG),
    )
