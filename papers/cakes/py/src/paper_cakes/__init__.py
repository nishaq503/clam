"""Reproducible results for CAKES paper."""

import typer

from . import plots
from . import recurrence_relations

app = typer.Typer()
app.add_typer(recurrence_relations.app, name="recurrence-relations")
app.add_typer(plots.app, name="plots")

__all__ = ["app", "plots", "recurrence_relations"]
