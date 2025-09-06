"""Pydantic models for structs from Rust."""

import typing

import pydantic

Id = typing.TypeVar("Id")
I = typing.TypeVar("I")  # noqa: E741
T = typing.TypeVar("T")
A = typing.TypeVar("A")


class Cluster(pydantic.BaseModel, typing.Generic[T, A]):
    """A cluster in the tree."""

    depth: int
    center_index: int
    cardinality: int
    radius: T
    lfd: float
    children: tuple[list[typing.Self], T] | None
    annotation: A | None


class Tree(pydantic.BaseModel, typing.Generic[Id, I, T, A]):
    """A tree."""

    items: list[tuple[Id, I]]
    root: Cluster[T, A]
    metric: typing.Callable[[I, I], T]


__all__ = ["A", "Cluster", "I", "Id", "T", "Tree"]
