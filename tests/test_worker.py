import json
import random
from pathlib import Path

import mlflow
import pytest

from mlbox.pipeline import load_pipeline
from mlbox.tracking import get_run, start_run
from mlbox.worker import run_step

PIPELINE_SOURCE = """
import random
from pathlib import Path

import mlflow
from pydantic import BaseModel

from mlbox import Pipeline, get_context, step


class Params(BaseModel):
    n: int = 1


@step()
def work(a: int, b: int = 0) -> dict:
    ctx = get_context(Params)
    mlflow.log_metric("score", a + b)
    ctx.mlflow.log_metric("via_context", 1)
    return {
        "n": ctx.input_params.n,
        "data_path": ctx.data_path,
        "seed": ctx.random_seed,
        "random": random.random(),
        "sum": a + b,
    }


@step()
def explode() -> int:
    raise RuntimeError("boom")


@step()
def bad_return() -> object:
    return object()


work(1, b=2)
explode()
bad_return()
pipeline = Pipeline("demo", Params, steps=[work, explode, bad_return])
"""


def _prepare(root: Path, step_name: str) -> str:
    """Lay out an /opt/ml-style directory for one step and return the MLflow run ID."""
    code = root / "code"
    config = root / "input" / "config"
    code.mkdir(parents=True)
    config.mkdir(parents=True)
    (root / "data").mkdir()
    (code / "pipeline.py").write_text(PIPELINE_SOURCE)
    pipeline = load_pipeline(code / "pipeline.py")
    run_id = start_run(pipeline, pipeline.parameters(n=7), seed=123, image="mlbox:test")
    (config / "hyperparameters.json").write_text(json.dumps({"n": 7}))
    step = {"pipeline_file": "pipeline.py", "step": step_name, "seed": 123, "run_id": run_id}
    step |= {"args": [1], "kwargs": {"b": 2}} if step_name == "work" else {"args": [], "kwargs": {}}
    (config / "step.json").write_text(json.dumps(step))
    return run_id


def test_worker_runs_the_step_and_writes_its_result_inside_an_active_mlflow_run(tmp_path: Path) -> None:
    root = tmp_path / "ml"
    run_id = _prepare(root, "work")

    exit_code = run_step(root)

    assert exit_code == 0
    assert json.loads((root / "output" / "result.json").read_text()) == {
        "n": 7,
        "data_path": str(root / "data"),
        "seed": 123,
        "random": random.Random(123).random(),
        "sum": 3,
    }
    assert mlflow.MlflowClient().get_run(run_id).data.metrics == {"score": 3.0, "via_context": 1.0}
    assert get_run(run_id).status == "RUNNING"


@pytest.mark.parametrize("step_name, error", [("explode", "boom"), ("bad_return", "JSON")])
def test_worker_writes_a_failure_file_when_the_step_fails(tmp_path: Path, step_name: str, error: str) -> None:
    root = tmp_path / "ml"
    _prepare(root, step_name)

    exit_code = run_step(root)

    assert exit_code != 0
    assert error in (root / "output" / "failure").read_text()
    assert not (root / "output" / "result.json").exists()
