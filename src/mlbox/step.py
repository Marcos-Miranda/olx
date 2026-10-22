import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

type PathKey = str | int


@dataclass(frozen=True)
class StepOutput:
    """Reference to (a part of) the future return value of a step."""

    step: "Step"
    path: tuple[PathKey, ...] = ()

    def __getitem__(self, key: PathKey) -> "StepOutput":
        return StepOutput(self.step, (*self.path, key))

    def __iter__(self) -> Any:
        raise TypeError("A step output is a reference and can't be iterated; index it instead")

    def resolve(self, results: dict[str, Any]) -> Any:
        """Return the referenced value, given the results of the upstream steps by step name."""
        value = results[self.step.name]
        for key in self.path:
            value = value[key]
        return value


class Step:
    """A function that runs in its own container, with the arguments bound by its single call."""

    def __init__(self, func: Callable[..., Any], cpus: float, memory: str) -> None:
        self.func = func
        self.name = func.__name__
        self.cpus = cpus
        self.memory = memory
        self.args: list[Any] = []
        self.kwargs: dict[str, Any] = {}
        self.called = False

    def __call__(self, *args: Any, **kwargs: Any) -> StepOutput:
        """Bind the arguments and return a reference to the step's future return value."""
        if self.called:
            raise RuntimeError(f"Step '{self.name}' was already called; a step can only be called once")
        for value in [*args, *kwargs.values()]:
            self._check_argument(value)
        self.args = list(args)
        self.kwargs = kwargs
        self.called = True
        return StepOutput(self)

    @property
    def dependencies(self) -> set["Step"]:
        """The steps whose outputs are referenced by the arguments."""
        found: set[Step] = set()
        _collect_references([*self.args, *self.kwargs.values()], found)
        return found

    def resolve_arguments(self, results: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Replace the references in the arguments with values from the upstream results."""
        args = [_resolve(value, results) for value in self.args]
        kwargs = {key: _resolve(value, results) for key, value in self.kwargs.items()}
        return args, kwargs

    def _check_argument(self, value: Any) -> None:
        if isinstance(value, StepOutput) or value is None or isinstance(value, bool | int | float | str):
            return
        if isinstance(value, list):
            for item in value:
                self._check_argument(item)
        elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
            for item in value.values():
                self._check_argument(item)
        else:
            raise TypeError(
                f"Step '{self.name}' got an argument of type {type(value).__name__}; "
                "arguments must be JSON literals or outputs of other steps"
            )


def step(cpus: float = 1, memory: str = "1g") -> Callable[[Callable[..., Any]], Step]:
    """Turn a function into a step that runs in a container limited to `cpus` CPUs and `memory` of RAM."""

    def decorator(func: Callable[..., Any]) -> Step:
        return Step(func, cpus, memory)

    return decorator


def to_json_value(value: Any) -> Any:
    """Convert `Path` values to strings (recursively) and check that the result is JSON-serializable."""
    converted = _paths_to_str(value)
    try:
        json.dumps(converted)
    except (TypeError, ValueError) as error:
        raise TypeError(f"The step's return value is not JSON-serializable: {error}") from error
    return converted


def _paths_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_paths_to_str(item) for item in value]
    if isinstance(value, dict):
        return {key: _paths_to_str(item) for key, item in value.items()}
    return value


def _collect_references(value: Any, found: set[Step]) -> None:
    if isinstance(value, StepOutput):
        found.add(value.step)
    elif isinstance(value, list):
        for item in value:
            _collect_references(item, found)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_references(item, found)


def _resolve(value: Any, results: dict[str, Any]) -> Any:
    if isinstance(value, StepOutput):
        return value.resolve(results)
    if isinstance(value, list):
        return [_resolve(item, results) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, results) for key, item in value.items()}
    return value
