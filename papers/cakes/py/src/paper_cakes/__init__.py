"""Reproducible results for CAKES paper."""

import typer

from . import comparisons
from . import datasets
from . import plots
from . import recurrence_relations
from . import utils

app = typer.Typer()
app.add_typer(recurrence_relations.app, name="recurrence-relations")
app.add_typer(plots.app, name="plots")
app.add_typer(comparisons.app, name="comparisons")

__all__ = ["app", "comparisons", "datasets", "plots", "recurrence_relations", "utils"]
