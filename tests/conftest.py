from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from mlbox import Pipeline, step
from mlbox.context import Context
from mlbox.tracking import start_run

TEST_MODEL_DATA = pd.DataFrame(
    {
        "height": [1.5, 2.0, 3.5],
        "count": [1, 2, 3],
        "active": [True, False, True],
        "kind": ["a", "b", "a"],
    }
)


class _TestModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return (data["height"] * data["count"]).to_numpy()

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        positive = data["active"].astype(float).to_numpy()
        return np.column_stack([1 - positive, positive])


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point MLflow at a throwaway sqlite store."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


@pytest.fixture
def register_test_model(tmp_path: Path) -> Callable[..., int]:
    """Return a factory that registers a small real model, with features of type double, long, boolean and string.

    `image` is the training image stored in the tags of the run that registers it, or `None` to register it in a plain
    MLflow run without mlbox tags. `predict_function` is `predict` (one value per row) or `predict_proba` (two).
    The factory returns the new version.
    """

    class Params(BaseModel):
        pass

    def register(name: str, image: str | None = "mlbox:test", predict_function: str = "predict") -> int:
        context = Context(Params(), tmp_path, random_seed=1, pipeline_name=name)
        if image is None:
            run = mlflow.start_run()
        else:

            @step()
            def train() -> int:
                return 1

            train()
            run = mlflow.start_run(run_id=start_run(Pipeline(name, Params, steps=[train]), Params(), 1, image))
        with run:
            return context.register_model(_TestModel(), TEST_MODEL_DATA, predict_function)

    return register
