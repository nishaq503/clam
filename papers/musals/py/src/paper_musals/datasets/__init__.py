"""Helpers for dealing with the datasets and metrics used in the paper."""

import enum


class Metric(enum.StrEnum):
    """Enum of distance metrics used in the paper."""
    Euclidean = "euclidean"
    Cosine = "cosine"
    Levenshtein = "levenshtein"

    def short_name(self) -> str:
        """Get the short name of the metric."""
        if self == Metric.Euclidean:
            return "euc"
        if self == Metric.Cosine:
            return "cos"
        if self == Metric.Levenshtein:
            return "lev"
        raise ValueError(f"Unknown metric: {self.value}")
