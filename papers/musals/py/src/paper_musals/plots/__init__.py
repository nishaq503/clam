"""Utilities for plotting results for the Musals paper."""

import typer

from . import polar_distances

app = typer.Typer()
app.command()(polar_distances.polar_distances)

__all__ = ["app", "polar_distances"]
