import time
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel

from mlbox import Pipeline, step
from mlbox.runner import Runner, StepJob, StepResult
from mlbox.tracking import get_run


class Params(BaseModel):
    n: int = 1


class FakeExecutor:
    """Records when each step runs and answers with canned values or failures."""

    def __init__(
        self,
        values: dict[str, Any],
        failing: set[str] | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.values = values
        self.failing = failing or set()
        self.delays = delays or {}
        self.jobs: dict[str, StepJob] = {}
        self.started: dict[str, float] = {}
        self.ended: dict[str, float] = {}
        self._lock = Lock()

    def execute(self, job: StepJob) -> StepResult:
        name = job.step.name
        with self._lock:
            self.jobs[name] = job
            self.started[name] = time.monotonic()
        time.sleep(self.delays.get(name, 0.2))
        with self._lock:
            self.ended[name] = time.monotonic()
        if name in self.failing:
            return StepResult(succeeded=False, value=None, error="boom")
        return StepResult(succeeded=True, value=self.values.get(name), error=None)


def _diamond() -> Pipeline:
    @step()
    def a() -> dict[str, int]:
        return {}

    @step()
    def b(x: int) -> int:
        return x

    @step()
    def c(x: int) -> int:
        return x

    @step()
    def d(left: int, right: int) -> int:
        return left + right

    root = a()
    d(b(root["x"]), c(root["x"]))
    return Pipeline("diamond", Params, steps=[d])


def test_runner_runs_independent_steps_in_parallel_after_their_dependencies(tmp_path: Path) -> None:
    executor = FakeExecutor({"a": {"x": 1}, "b": 10, "c": 20, "d": 30})
    runner = Runner(executor, runs_dir=tmp_path / "runs")

    run_id = runner.run(_diamond(), Params(n=4), seed=9)

    assert executor.started["b"] < executor.ended["c"] and executor.started["c"] < executor.ended["b"]
    assert executor.started["b"] >= executor.ended["a"] and executor.started["c"] >= executor.ended["a"]
    assert executor.started["d"] >= max(executor.ended["b"], executor.ended["c"])
    assert executor.jobs["b"].args == [1]
    assert executor.jobs["d"].args == [10, 20]
    assert executor.jobs["d"].parameters == {"n": 4}
    assert executor.jobs["d"].seed == 9
    assert executor.jobs["d"].run_id == run_id
    assert (tmp_path / "runs" / run_id / "data").is_dir()
    info = get_run(run_id)
    assert info.status == "FINISHED"
    assert info.step_statuses == {"a": "succeeded", "b": "succeeded", "c": "succeeded", "d": "succeeded"}


def test_runner_skips_unstarted_steps_after_a_failure_and_lets_running_ones_finish(tmp_path: Path) -> None:
    @step()
    def first() -> int:
        return 1

    @step()
    def fails(x: int) -> int:
        return x

    @step()
    def slow(x: int) -> int:
        return x

    @step()
    def after_fails(x: int) -> int:
        return x

    @step()
    def after_slow(x: int) -> int:
        return x

    root = first()
    after_fails(fails(root))
    after_slow(slow(root))
    pipeline = Pipeline("failing", Params, steps=[after_fails, after_slow])
    executor = FakeExecutor({}, failing={"fails"}, delays={"fails": 0.05, "slow": 0.4})

    run_id = Runner(executor, runs_dir=tmp_path / "runs").run(pipeline, Params(), seed=1)

    assert set(executor.jobs) == {"first", "fails", "slow"}
    info = get_run(run_id)
    assert info.status == "FAILED"
    assert info.step_statuses == {
        "first": "succeeded",
        "fails": "failed",
        "slow": "succeeded",
        "after_fails": "skipped",
        "after_slow": "skipped",
    }
