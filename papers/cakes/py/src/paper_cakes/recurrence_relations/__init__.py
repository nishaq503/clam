"""Exploring the recurrence relations of CLAM clustering with different branching factors."""

import typer

from . import gen_ratios

app = typer.Typer()
app.command()(gen_ratios.gen_ratios)

__all__ = ["app", "gen_ratios"]
