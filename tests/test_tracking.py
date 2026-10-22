import mlflow
from pydantic import BaseModel

from mlbox import Pipeline, step
from mlbox.tracking import end_run, get_run, list_runs, set_step_status, start_run


class Params(BaseModel):
    n: int = 3
    name: str = "x"


def _pipeline(name: str = "demo") -> Pipeline:
    @step()
    def first() -> int:
        return 1

    @step()
    def second(x: int) -> int:
        return x

    second(first())
    return Pipeline(name, Params, steps=[second])


def test_start_run_creates_a_running_run_with_params_seed_image_and_pending_steps() -> None:
    run_id = start_run(_pipeline(), Params(n=5), seed=42, image="mlbox:test")

    run = mlflow.MlflowClient().get_run(run_id)
    experiment = mlflow.MlflowClient().get_experiment(run.info.experiment_id)
    info = get_run(run_id)
    assert experiment.name == "demo"
    assert info.run_id == run_id
    assert info.pipeline == "demo"
    assert info.status == "RUNNING"
    assert info.params == {"n": "5", "name": "x"}
    assert info.seed == 42
    assert info.image == "mlbox:test"
    assert info.step_statuses == {"first": "pending", "second": "pending"}


def test_step_statuses_and_the_final_status_are_reflected_by_get_run() -> None:
    run_id = start_run(_pipeline(), Params(), seed=1, image="mlbox:test")

    set_step_status(run_id, "first", "succeeded")
    set_step_status(run_id, "second", "skipped")
    end_run(run_id, succeeded=False)

    info = get_run(run_id)
    assert info.step_statuses == {"first": "succeeded", "second": "skipped"}
    assert info.status == "FAILED"
    assert info.end_time is not None


def test_end_run_marks_a_successful_run_as_finished() -> None:
    run_id = start_run(_pipeline(), Params(), seed=1, image="mlbox:test")

    end_run(run_id, succeeded=True)

    assert get_run(run_id).status == "FINISHED"


def test_list_runs_returns_only_mlbox_runs_newest_first_and_filters_by_pipeline() -> None:
    older = start_run(_pipeline("alpha"), Params(), seed=1, image="mlbox:test")
    newer = start_run(_pipeline("beta"), Params(), seed=2, image="mlbox:test")
    newest = start_run(_pipeline("alpha"), Params(), seed=3, image="mlbox:test")
    with mlflow.start_run(run_name="not-mlbox"):
        pass

    assert [run.run_id for run in list_runs()] == [newest, newer, older]
    assert [run.run_id for run in list_runs(pipeline="alpha")] == [newest, older]
