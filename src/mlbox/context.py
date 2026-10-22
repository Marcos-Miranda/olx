from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from pydantic import BaseModel


@dataclass(frozen=True)
class Context[P: BaseModel]:
    """What a step can access while it runs: parameters, shared data directory, seed and MLflow."""

    input_params: P
    data_path: Path
    random_seed: int
    pipeline_name: str

    @property
    def mlflow(self) -> "mlflow":  # type: ignore [valid-type]
        """The `mlflow` module, with the execution's run already active."""
        return mlflow

    def register_model(self, model: Any, input_data: pd.DataFrame, predict_function: str) -> int:
        """Log `model` to the active run, register it under the pipeline's name and return the new version.

        `predict_function` names the method of `model` that makes predictions. The signature is inferred from
        `input_data`.
        """
        predict = getattr(model, predict_function)
        model_info = mlflow.pyfunc.log_model(
            name="model",
            # MLflow requires a callable with exactly one parameter, and methods like `(X, **params)` have two
            python_model=lambda model_input: predict(model_input),
            signature=infer_signature(input_data, predict(input_data)),
            input_example=input_data.head(5),
            registered_model_name=self.pipeline_name,
        )
        return model_info.registered_model_version


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
