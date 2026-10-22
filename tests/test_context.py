import pytest

from mlbox import get_context


def test_get_context_outside_a_step_execution_raises() -> None:
    with pytest.raises(RuntimeError, match="step"):
        get_context()
