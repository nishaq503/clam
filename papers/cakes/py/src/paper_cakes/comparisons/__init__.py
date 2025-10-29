"""Running other KNN algorithms for comparison against CAKES."""

import typer

from . import knn_faiss

app = typer.Typer()
app.command()(knn_faiss.faiss_flat)

__all__ = ["app", "knn_faiss"]
