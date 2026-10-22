from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, overload

from pydantic import BaseModel


@dataclass(frozen=True)
class Context[P: BaseModel]:
    """What a step can access while it runs: parameters, shared data directory, seed and MLflow."""

    input_params: P
    data_path: Path
    random_seed: int

    @property
    def mlflow(self) -> ModuleType:
        """The `mlflow` module, with the execution's run already active."""
        import mlflow

        return mlflow


_current: ContextVar[Context[Any] | None] = ContextVar("mlbox_context", default=None)


@overload
def get_context() -> Context[BaseModel]: ...


@overload
def get_context[P: BaseModel](params_type: type[P]) -> Context[P]: ...


def get_context(params_type: type[BaseModel] | None = None) -> Context[Any]:
    """Return the context of the current step execution; `params_type` is the pipeline's parameters model."""
    context = _current.get()
    if context is None:
        raise RuntimeError("get_context() can only be called from inside a step execution")
    if params_type is not None and not isinstance(context.input_params, params_type):
        raise TypeError(
            f"The pipeline's parameters are {type(context.input_params).__name__}, not {params_type.__name__}"
        )
    return context


@contextmanager
def _use_context(context: Context[Any]) -> Iterator[None]:
    token = _current.set(context)
    try:
        yield
    finally:
        _current.reset(token)
