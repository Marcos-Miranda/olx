from pathlib import Path

from typer.testing import CliRunner

from mlbox.main import app
from mlbox.tracking import list_runs

PIPELINE_SOURCE = """
from pydantic import BaseModel

from mlbox import Pipeline, step


class Params(BaseModel):
    n: int


@step()
def work(n: int) -> int:
    return n


work(1)
pipeline = Pipeline("cli-demo", Params, steps=[work])
"""


def test_run_with_invalid_parameters_exits_with_the_validation_error_and_creates_no_run(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(PIPELINE_SOURCE)

    result = CliRunner().invoke(app, ["run", str(pipeline_file), "-p", "n=not-a-number"])

    assert result.exit_code != 0
    assert "n" in result.output and "valid integer" in result.output
    assert list_runs() == []
