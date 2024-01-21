"""Classes for running rival search algorithms."""

import enum
import typing

from . import ann_oh_yeah
from . import base
from . import faiss_flat
from . import faiss_ivf_flat
from . import hnsw
from .base import Metric  # noqa: F401
from . import utils  # noqa: F401


class Rival(str, enum.Enum):
    FaissFlat = "faiss-flat"
    FaissIvfFlat = "faiss-ivf-flat"
    Hnsw = "hnsw"
    Annoy = "annoy"

    def model(self) -> typing.Type[base.Algorithm]:
        """Get the model class for the algorithm."""
        m: typing.Type[base.Algorithm]

        if self == Rival.FaissFlat:
            m = faiss_flat.FaissFlat
        elif self == Rival.FaissIvfFlat:
            m = faiss_ivf_flat.FaissIVFFlat
        elif self == Rival.Hnsw:
            m = hnsw.Hnsw
        elif self == Rival.Annoy:
            m = ann_oh_yeah.Annoy
        else:
            raise ValueError(f"Invalid algorithm {self}.")

        return m
