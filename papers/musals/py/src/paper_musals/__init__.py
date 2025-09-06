"""Reproducible results for Musals paper."""

import typer

from . import datasets
from . import models
from . import plots
from . import utils

app = typer.Typer()
app.add_typer(plots.app, name="plots")

__all__ = ["app", "datasets", "models", "plots", "utils"]
