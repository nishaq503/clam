"""Reproducible results for CAKES paper."""

import typer

from . import recurrence_relations

app = typer.Typer()
app.add_typer(recurrence_relations.app, name="recurrence-relations")

__all__ = ["app", "recurrence_relations"]
