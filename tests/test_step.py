from pathlib import Path

import pytest

from mlbox import StepOutput, step
from mlbox.step import to_json_value


def test_calling_a_step_with_a_non_json_argument_raises() -> None:
    @step()
    def consume(x: object) -> None:
        pass

    with pytest.raises(TypeError, match="consume"):
        consume(object())


def test_calling_a_step_with_a_non_json_argument_nested_in_a_list_raises() -> None:
    @step()
    def consume(x: object) -> None:
        pass

    with pytest.raises(TypeError):
        consume([1, {"a": {1, 2}}])


def test_calling_a_step_twice_raises() -> None:
    @step()
    def once() -> int:
        return 1

    once()
    with pytest.raises(RuntimeError, match="once"):
        once()


def test_arguments_resolve_references_including_indexed_ones() -> None:
    @step()
    def source() -> dict[str, list[int]]:
        return {}

    @step()
    def other() -> list[str]:
        return []

    @step()
    def consume(a: object, b: object, c: object, *, k: object) -> None:
        pass

    data = source()
    items = other()
    consume(data["train"], items[0], 5, k=[data["test"], "x"])

    args, kwargs = consume.resolve_arguments({"source": {"train": [1, 2], "test": [3]}, "other": ["first"]})

    assert args == [[1, 2], "first", 5]
    assert kwargs == {"k": [[3], "x"]}
    assert consume.dependencies == {source, other}


def test_a_step_output_reference_is_returned_when_calling_a_step() -> None:
    @step(cpus=2, memory="4g")
    def produce() -> int:
        return 1

    ref = produce()

    assert isinstance(ref, StepOutput)
    assert (produce.name, produce.cpus, produce.memory) == ("produce", 2, "4g")


def test_to_json_value_serializes_paths_as_strings() -> None:
    assert to_json_value({"p": Path("a/b"), "l": [Path("c")], "n": 1}) == {"p": "a/b", "l": ["c"], "n": 1}


def test_to_json_value_rejects_a_non_serializable_value() -> None:
    with pytest.raises(TypeError):
        to_json_value({"x": object()})
