from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlbox.serving import app, get_model, load_served_model

INSTANCES = [
    {"height": 2, "count": 3, "active": True, "kind": "a"},
    {"height": 1.5, "count": 2, "active": False, "kind": "b"},
]


@pytest.fixture
def client(register_test_model: Callable[..., int], request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """A client of the app with `get_model` overridden by a registered model (`predict` or `predict_proba`)."""
    predict_function = getattr(request, "param", "predict")
    version = register_test_model("demo", predict_function=predict_function)
    model = load_served_model("demo", version)
    app.dependency_overrides[get_model] = lambda: model
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_app_loads_the_configured_model_and_exposes_health_info_and_the_input_example(
    register_test_model: Callable[..., int], monkeypatch: pytest.MonkeyPatch
) -> None:
    version = register_test_model("demo")
    monkeypatch.setenv("MLBOX_MODEL_NAME", "demo")
    monkeypatch.setenv("MLBOX_MODEL_VERSION", str(version))

    with TestClient(app) as client:
        health = client.get("/health")
        info = client.get("/info")
        openapi = client.get("/openapi.json").json()

    assert health.json() == {"status": "ok"}
    assert info.json() == {
        "name": "demo",
        "version": version,
        "input_schema": [
            {"name": "height", "type": "double", "required": True},
            {"name": "count", "type": "long", "required": True},
            {"name": "active", "type": "boolean", "required": True},
            {"name": "kind", "type": "string", "required": True},
        ],
    }
    body = openapi["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/PredictRequest"}
    assert openapi["components"]["schemas"]["PredictRequest"]["examples"] == [
        [
            {"height": 1.5, "count": 1, "active": True, "kind": "a"},
            {"height": 2.0, "count": 2, "active": False, "kind": "b"},
            {"height": 3.5, "count": 3, "active": True, "kind": "a"},
        ]
    ]


def test_predict_returns_one_float_per_instance_and_accepts_an_integer_for_a_double_feature(client: TestClient) -> None:
    response = client.post("/predict", json=INSTANCES)

    assert response.status_code == 200
    assert response.json() == [[6.0], [3.0]]


@pytest.mark.parametrize("client", ["predict_proba"], indirect=True)
def test_predict_returns_one_value_per_class_for_a_model_with_probabilities(client: TestClient) -> None:
    response = client.post("/predict", json=INSTANCES)

    assert response.status_code == 200
    assert response.json() == [[0.0, 1.0], [1.0, 0.0]]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([{"height": 2.0, "count": 3, "active": True}], "missing"),
        ([{"height": "tall", "count": 3, "active": True, "kind": "a"}], "Incompatible input types"),
    ],
    ids=["missing_feature", "wrong_type"],
)
def test_predict_rejects_an_instance_that_does_not_match_the_schema_with_the_mlflow_message(
    client: TestClient, body: list[dict[str, Any]], message: str
) -> None:
    response = client.post("/predict", json=body)

    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_predict_rejects_an_empty_array(client: TestClient) -> None:
    assert client.post("/predict", json=[]).status_code == 422
