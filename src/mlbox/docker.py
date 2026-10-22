import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path

import docker
from docker.errors import ImageNotFound
from docker.models.containers import Container

from mlbox.pipeline import Pipeline
from mlbox.runner import StepJob, StepResult

BASE_IMAGE = "mlbox:latest"
NETWORK = "mlbox"

_PIPELINE_DOCKERFILE = f"""FROM {BASE_IMAGE}
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt
"""


def build_pipeline_image(pipeline: Pipeline, pipeline_dir: Path) -> str:
    """Return the image to run the pipeline's steps in, building it if the requirements aren't installed yet."""
    client = docker.from_env()
    try:
        client.images.get(BASE_IMAGE)
    except ImageNotFound:
        raise RuntimeError(f"The base image {BASE_IMAGE} doesn't exist; build it with `docker build -t mlbox .`")
    if pipeline.requirements is None:
        return BASE_IMAGE

    requirements = (pipeline_dir / pipeline.requirements).read_bytes()
    name = re.sub(r"[^a-z0-9_.-]", "-", pipeline.name.lower())
    tag = f"mlbox-{name}:{hashlib.sha256(requirements).hexdigest()[:12]}"
    try:
        client.images.get(tag)
    except ImageNotFound:
        client.images.build(fileobj=_build_context(requirements), custom_context=True, tag=tag, rm=True)
    return tag


class DockerExecutor:
    """Runs each step in its own container, following the `/opt/ml` contract."""

    def __init__(self, image: str, pipeline_file: Path) -> None:
        self.image = image
        self.pipeline_file = pipeline_file.resolve()

    def execute(self, job: StepJob) -> StepResult:
        """Run the step in a container, streaming its logs, and read its result or failure file."""
        name = job.step.name
        step_dir = (job.run_dir / "steps" / name).resolve()
        config_dir, output_dir = step_dir / "config", step_dir / "output"
        config_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        step_config = {
            "pipeline_file": self.pipeline_file.name,
            "step": name,
            "args": job.args,
            "kwargs": job.kwargs,
            "seed": job.seed,
            "run_id": job.run_id,
        }
        (config_dir / "step.json").write_text(json.dumps(step_config))
        (config_dir / "hyperparameters.json").write_text(json.dumps(job.parameters))

        client = docker.from_env()
        container = client.containers.run(
            self.image,
            detach=True,
            nano_cpus=int(job.step.cpus * 1e9),
            mem_limit=job.step.memory,
            user=f"{os.getuid()}:{os.getgid()}",
            environment={
                "HOME": "/tmp",
                "MLFLOW_TRACKING_URI": os.environ.get("MLBOX_CONTAINER_TRACKING_URI", "http://mlflow:5000"),
            },
            network=NETWORK,
            volumes={
                str(self.pipeline_file.parent): {"bind": "/opt/ml/code", "mode": "ro"},
                str(config_dir): {"bind": "/opt/ml/input/config", "mode": "ro"},
                str((job.run_dir / "data").resolve()): {"bind": "/opt/ml/data", "mode": "rw"},
                str(output_dir): {"bind": "/opt/ml/output", "mode": "rw"},
            },
        )
        try:
            _stream_logs(container, name)
            exit_code = container.wait()["StatusCode"]
        finally:
            container.remove(force=True)

        if (output_dir / "result.json").exists():
            return StepResult(succeeded=True, value=json.loads((output_dir / "result.json").read_text()), error=None)
        if (output_dir / "failure").exists():
            return StepResult(succeeded=False, value=None, error=(output_dir / "failure").read_text())
        return StepResult(succeeded=False, value=None, error=f"The container exited with code {exit_code}")


def _build_context(requirements: bytes) -> io.BytesIO:
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as tar:
        for name, content in [("Dockerfile", _PIPELINE_DOCKERFILE.encode()), ("requirements.txt", requirements)]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    context.seek(0)
    return context


def _stream_logs(container: Container, step_name: str) -> None:
    pending = ""
    for chunk in container.logs(stream=True, follow=True):
        pending += chunk.decode(errors="replace")
        *lines, pending = pending.split("\n")
        for line in lines:
            print(f"[{step_name}] {line}", flush=True)
    if pending:
        print(f"[{step_name}] {pending}", flush=True)
