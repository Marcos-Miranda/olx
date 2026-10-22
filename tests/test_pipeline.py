from pathlib import Path

import pytest
from pydantic import BaseModel

from mlbox import Pipeline, step
from mlbox.pipeline import load_pipeline


class Params(BaseModel):
    n: int = 1


def test_defining_a_pipeline_runs_no_step_code_and_includes_upstream_steps() -> None:
    executed: list[str] = []

    @step()
    def preprocess() -> dict[str, list[int]]:
        executed.append("preprocess")
        return {"train": [1, 2], "test": [3]}

    @step()
    def train(data: list[int]) -> str:
        executed.append("train")
        return "model"

    data = preprocess()
    model = train(data["train"])
    pipeline = Pipeline("demo", Params, steps=[train])

    assert executed == []
    assert model is not None
    assert list(pipeline.steps) == ["preprocess", "train"]
    assert pipeline.dependencies("train") == {"preprocess"}
    assert pipeline.dependencies("preprocess") == set()


def test_pipeline_orders_a_diamond_graph_with_dependencies_first() -> None:
    @step()
    def a() -> int:
        return 1

    @step()
    def b(x: int) -> int:
        return x

    @step()
    def c(x: int) -> int:
        return x

    @step()
    def d(x: int, y: int) -> int:
        return x + y

    root = a()
    d(b(root), c(root))
    pipeline = Pipeline("diamond", Params, steps=[d])

    order = list(pipeline.steps)
    assert order[0] == "a"
    assert order[-1] == "d"
    assert pipeline.dependencies("d") == {"b", "c"}


def test_pipeline_rejects_a_step_that_was_never_called() -> None:
    @step()
    def lonely() -> int:
        return 1

    with pytest.raises(ValueError, match="lonely"):
        Pipeline("demo", Params, steps=[lonely])


PIPELINE_SOURCE = """
from pydantic import BaseModel

from helpers import double
from mlbox import Pipeline, step


class Params(BaseModel):
    n: int = 1


@step()
def work(n: int) -> int:
    return double(n)


work(1)
{definitions}
"""


def _write_pipeline(directory: Path, definitions: str) -> Path:
    (directory / "helpers.py").write_text("def double(x):\n    return 2 * x\n")
    path = directory / "pipeline.py"
    path.write_text(PIPELINE_SOURCE.format(definitions=definitions))
    return path


def test_load_pipeline_returns_the_only_pipeline_defined_in_the_file(tmp_path: Path) -> None:
    path = _write_pipeline(tmp_path, 'pipeline = Pipeline("one", Params, steps=[work])')

    pipeline = load_pipeline(path)

    assert pipeline.name == "one"
    assert list(pipeline.steps) == ["work"]


def test_load_pipeline_raises_when_the_file_defines_no_pipeline(tmp_path: Path) -> None:
    path = _write_pipeline(tmp_path, "")

    with pytest.raises(ValueError, match="no Pipeline"):
        load_pipeline(path)


def test_load_pipeline_raises_when_the_file_defines_several_pipelines(tmp_path: Path) -> None:
    path = _write_pipeline(
        tmp_path,
        'first = Pipeline("first", Params, steps=[work])\nsecond = Pipeline("second", Params, steps=[work])',
    )

    with pytest.raises(ValueError, match="several"):
        load_pipeline(path)
