import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel, Field, RootModel

_FLOAT_TYPES = {"double", "float"}


@dataclass
class ServedModel:
    """A registered model loaded for serving, with what the API exposes about it."""

    name: str
    version: int
    model: PyFuncModel
    input_schema: list[dict[str, Any]]
    input_example: list[dict[str, Any]] | None


class PredictRequest(RootModel[Annotated[list[dict[str, bool | int | float | str]], Field(min_length=1)]]):
    """One or more instances, each mapping feature names to values."""


class InfoResponse(BaseModel):
    """The served model's name, version and input schema."""

    name: str
    version: int
    input_schema: list[dict[str, Any]]


def load_served_model(name: str, version: int) -> ServedModel:
    """Load version `version` of the registered model `name` from MLflow."""
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    schema = model.metadata.get_input_schema()
    input_schema = [
        {"name": column.name, "type": column.type.name, "required": column.required}
        for column in (schema.inputs if schema else [])
    ]
    example = model.input_example
    input_example = (
        [{str(key): value for key, value in record.items()} for record in example.to_dict(orient="records")]
        if isinstance(example, pd.DataFrame)
        else None
    )
    return ServedModel(name, version, model, input_schema, input_example)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    served = load_served_model(os.environ["MLBOX_MODEL_NAME"], int(os.environ["MLBOX_MODEL_VERSION"]))
    app.state.model = served
    app.openapi_schema = None
    if served.input_example is not None:
        # FastAPI builds the body schema when the route is defined, so the example is added to the generated document
        app.openapi()["components"]["schemas"][PredictRequest.__name__]["examples"] = [served.input_example]
    yield


app = FastAPI(title="mlbox model", lifespan=_lifespan)


def get_model(request: Request) -> ServedModel:
    """The model loaded at startup."""
    model: ServedModel = request.app.state.model
    return model


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API is up."""
    return {"status": "ok"}


@app.get("/info")
async def info(served: Annotated[ServedModel, Depends(get_model)]) -> InfoResponse:
    """Return the model's name, version and input schema."""
    return InfoResponse(name=served.name, version=served.version, input_schema=served.input_schema)


@app.post("/predict")
def predict(body: PredictRequest, served: Annotated[ServedModel, Depends(get_model)]) -> list[list[float]]:
    """Run the model on the instances and return one list of floats per instance."""
    data = pd.DataFrame(body.root)
    for column in served.input_schema:
        # JSON doesn't tell `5` from `5.0`, and MLflow rejects `int64` for a double column
        if column["type"] in _FLOAT_TYPES and column["name"] in data:
            data[column["name"]] = data[column["name"]].map(lambda value: float(value) if type(value) is int else value)
    try:
        prediction = served.model.predict(data)
    except MlflowException as error:
        raise HTTPException(status_code=422, detail=error.message) from error
    return np.asarray(prediction, dtype=float).reshape(len(data), -1).tolist()
