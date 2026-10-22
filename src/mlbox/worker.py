import json
import random
import sys
import traceback
from pathlib import Path

import mlflow

from mlbox.context import Context, _use_context
from mlbox.pipeline import load_pipeline
from mlbox.step import to_json_value


def run_step(root: Path = Path("/opt/ml")) -> int:
    """Run the step described under `root` (the `/opt/ml` contract) and write its result or failure file."""
    output = root / "output"
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = root / "input" / "config"
        spec = json.loads((config / "step.json").read_text())
        hyperparameters = json.loads((config / "hyperparameters.json").read_text())

        pipeline = load_pipeline(root / "code" / spec["pipeline_file"])
        step = pipeline.steps[spec["step"]]
        context = Context(
            input_params=pipeline.parameters.model_validate(hyperparameters),
            data_path=root / "data",
            random_seed=spec["seed"],
            pipeline_name=pipeline.name,
        )
        _seed(context.random_seed)

        mlflow.start_run(run_id=spec["run_id"])
        with _use_context(context):
            result = to_json_value(step.func(*spec["args"], **spec["kwargs"]))
        (output / "result.json").write_text(json.dumps(result))
    except BaseException:
        (output / "failure").write_text(traceback.format_exc())
        return 1
    finally:
        # The runner owns the final status of the run, so leave it RUNNING
        mlflow.end_run(status="RUNNING")
    return 0


def _seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(seed)


if __name__ == "__main__":
    sys.exit(run_step())
