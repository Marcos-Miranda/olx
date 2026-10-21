import typer

app = typer.Typer()


# TODO: Remove it
@app.command()
def hello(name: str) -> None:
    """Print \"Hello <name>\""""
    print(f"Hello {name}")


if __name__ == "__main__":
    app()
