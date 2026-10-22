import json
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from mlbox import tracking
from mlbox.docker import DockerExecutor, build_pipeline_image, serve_model
from mlbox.pipeline import load_pipeline
from mlbox.runner import Runner

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    pipeline_file: Annotated[Path, typer.Argument(help="Python file that defines the pipeline")],
    param: Annotated[
        list[str] | None, typer.Option("--param", "-p", help="Pipeline parameter as key=value (repeatable)")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed (generated when omitted)")] = None,
) -> None:
    """Run a pipeline, each step in its own container, and wait for it to finish."""
    try:
        pipeline = load_pipeline(pipeline_file)
    except (OSError, ValueError) as error:
        _fail(str(error))
    try:
        params = pipeline.parameters.model_validate(_parse_params(param or []))
    except ValidationError as error:
        _fail(str(error))

    try:
        image = build_pipeline_image(pipeline, pipeline_file.resolve().parent)
    except RuntimeError as error:
        _fail(str(error))
    runner = Runner(DockerExecutor(image, pipeline_file), on_start=lambda run_id: typer.echo(f"Run ID: {run_id}"))
    run_id = runner.run(pipeline, params, seed if seed is not None else random.randint(0, 2**31 - 1), image)

    status = tracking.get_run(run_id).status
    typer.echo(f"Status: {status}")
    if status != "FINISHED":
        raise typer.Exit(1)


@app.command("list")
def list_runs(
    pipeline: Annotated[str | None, typer.Option(help="Only show runs of this pipeline")] = None,
) -> None:
    """List the runs, newest first."""
    typer.echo(f"{'RUN ID':<34}{'PIPELINE':<20}{'STATUS':<10}{'STARTED':<20}DURATION")
    for info in tracking.list_runs(pipeline):
        duration = f"{(info.end_time - info.start_time) / 1000:.1f}s" if info.end_time else "-"
        typer.echo(
            f"{info.run_id:<34}{info.pipeline:<20}{info.status:<10}{_format_time(info.start_time):<20}{duration}"
        )


@app.command()
def show(run_id: Annotated[str, typer.Argument(help="ID of the run")]) -> None:
    """Show the details of a run."""
    info = tracking.get_run(run_id)
    typer.echo(f"Run ID:   {info.run_id}")
    typer.echo(f"Pipeline: {info.pipeline}")
    typer.echo(f"Status:   {info.status}")
    typer.echo(f"Seed:     {info.seed}")
    typer.echo(f"Image:    {info.image or '-'}")
    typer.echo("Parameters:")
    for key, value in info.params.items():
        typer.echo(f"  {key} = {value}")
    typer.echo("Steps:")
    for name, status in info.step_statuses.items():
        typer.echo(f"  {name}: {status}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def serve(
    context: typer.Context,
    model: Annotated[str, typer.Option(help="Name of the registered model")],
    version: Annotated[int | None, typer.Option(help="Model version (the latest when omitted)")] = None,
    port: Annotated[int, typer.Option(help="Host port to publish the API on")] = 8080,
) -> None:
    """Serve a registered model through a REST API, in a container of the image that trained it.

    Extra arguments are passed to uvicorn, e.g. `mlbox serve --model iris --workers 2`. Runs until Ctrl+C.
    """
    try:
        info = tracking.get_model_version(model, version)
    except ValueError as error:
        _fail(str(error))
    typer.echo(f"Model:   {info.name}")
    typer.echo(f"Version: {info.version}")
    typer.echo(f"Image:   {info.image}")
    typer.echo(f"URL:     http://localhost:{port}")
    try:
        exit_code = serve_model(info, port, context.args)
    except RuntimeError as error:
        _fail(str(error))
    raise typer.Exit(exit_code)


def _fail(message: str) -> Any:
    typer.echo(message, err=True)
    raise typer.Exit(1)


def _parse_params(params: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in params:
        key, separator, value = item.partition("=")
        if not separator:
            _fail(f"Invalid parameter '{item}'; expected key=value")
        try:
            parsed[key] = json.loads(value)
        except json.JSONDecodeError:
            parsed[key] = value
    return parsed


def _format_time(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    app()
