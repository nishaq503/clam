"""Utilities for plotting results for the CAKES paper."""

import typer

from . import cluster_properties
from . import search_performance

app = typer.Typer()
app.command()(cluster_properties.plot_cluster_properties)
app.command()(search_performance.plot_search_performance)

__all__ = ["app", "cluster_properties", "search_performance"]
