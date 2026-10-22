import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel

from mlbox.step import Step


class Pipeline:
    """A set of steps and the parameters model they share, with the execution graph derived from the steps."""

    def __init__(
        self,
        name: str,
        parameters: type[BaseModel],
        steps: list[Step],
        requirements: str | None = None,
    ) -> None:
        self.name = name
        self.parameters = parameters
        self.requirements = requirements
        self.steps: dict[str, Step] = {}
        for final_step in steps:
            self._add(final_step)

    def dependencies(self, name: str) -> set[str]:
        """Names of the steps that the given step depends on."""
        return {dependency.name for dependency in self.steps[name].dependencies}

    def _add(self, step: Step) -> None:
        if not step.called:
            raise ValueError(f"Step '{step.name}' was never called, so it can't be part of a pipeline")
        known = self.steps.get(step.name)
        if known is step:
            return
        if known is not None:
            raise ValueError(f"Two steps are named '{step.name}'; step names must be unique in a pipeline")
        for dependency in step.dependencies:
            self._add(dependency)
        # Dependencies are added first, so insertion order is a valid execution order
        self.steps[step.name] = step


def load_pipeline(path: Path) -> Pipeline:
    """Import a pipeline file and return the single `Pipeline` it defines."""
    path = path.resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module_name = f"mlbox_pipeline_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Can't import {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    # Registered so that pydantic and dataclasses can resolve names defined in the file
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    pipelines = [value for value in vars(module).values() if isinstance(value, Pipeline)]
    if not pipelines:
        raise ValueError(f"{path} defines no Pipeline")
    if len(pipelines) > 1:
        raise ValueError(f"{path} defines several Pipelines; it must define exactly one")
    return pipelines[0]
