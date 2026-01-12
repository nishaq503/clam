"""Running other MSA algorithms for comparison against Musals."""

import typer

from . import knn_faiss

app = typer.Typer()
app.command()(knn_faiss.faiss_flat)

__all__ = ["app", "knn_faiss"]
