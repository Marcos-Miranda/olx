import sys
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from mlbox import tracking
from mlbox.pipeline import Pipeline
from mlbox.step import Step


@dataclass
class StepJob:
    """Everything an executor needs to run one step."""

    run_id: str
    step: Step
    args: list[Any]
    kwargs: dict[str, Any]
    parameters: dict[str, Any]
    seed: int
    run_dir: Path


@dataclass
class StepResult:
    """The outcome of one step: its return value on success, an error message on failure."""

    succeeded: bool
    value: Any
    error: str | None


class Executor(Protocol):
    """Runs a single step somewhere (a container, in practice) and reports the outcome."""

    def execute(self, job: StepJob) -> StepResult:
        """Run the step described by `job`, blocking until it ends."""
        ...


class Runner:
    """Runs a pipeline: each step as soon as its dependencies have succeeded, independent steps in parallel.

    `on_start` is called with the run ID once the run exists, before any step starts.
    """

    def __init__(
        self,
        executor: Executor,
        runs_dir: Path = Path(".mlbox/runs"),
        on_start: Callable[[str], None] | None = None,
    ) -> None:
        self.executor = executor
        self.runs_dir = runs_dir
        self.on_start = on_start

    def run(self, pipeline: Pipeline, params: BaseModel, seed: int, image: str) -> str:
        """Execute the pipeline, its steps running in `image`, and return the run ID. Final status goes to MLflow."""
        run_id = tracking.start_run(pipeline, params, seed, image)
        if self.on_start:
            self.on_start(run_id)
        run_dir = self.runs_dir / run_id
        (run_dir / "data").mkdir(parents=True)

        statuses = dict.fromkeys(pipeline.steps, "pending")
        results: dict[str, Any] = {}
        running: dict[Future[StepResult], str] = {}
        failed = False

        def set_status(name: str, status: str) -> None:
            statuses[name] = status
            tracking.set_step_status(run_id, name, status)

        with ThreadPoolExecutor(max_workers=len(pipeline.steps)) as pool:
            while True:
                if not failed:
                    for name, step in pipeline.steps.items():
                        if statuses[name] == "pending" and all(
                            statuses[dependency] == "succeeded" for dependency in pipeline.dependencies(name)
                        ):
                            args, kwargs = step.resolve_arguments(results)
                            job = StepJob(run_id, step, args, kwargs, params.model_dump(mode="json"), seed, run_dir)
                            set_status(name, "running")
                            running[pool.submit(self._execute, job)] = name
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    name = running.pop(future)
                    result = future.result()
                    if result.succeeded:
                        results[name] = result.value
                        set_status(name, "succeeded")
                    else:
                        failed = True
                        set_status(name, "failed")
                        print(f"[{name}] failed: {result.error}", file=sys.stderr)

        for name, status in statuses.items():
            if status == "pending":
                set_status(name, "skipped")
        tracking.end_run(run_id, succeeded=not failed)
        return run_id

    def _execute(self, job: StepJob) -> StepResult:
        try:
            return self.executor.execute(job)
        except Exception as error:
            return StepResult(succeeded=False, value=None, error=f"{type(error).__name__}: {error}")
