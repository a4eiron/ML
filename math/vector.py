from __future__ import annotations
from typing import override

import math


class Vector:
    elements: list[int | float] = []

    def __init__(self, elements: list[int | float]) -> None:
        self.elements = elements

    @override
    def __str__(self) -> str:
        return f"Vector({[round(x, 4) for x in self.elements]})"

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, i: int) -> int | float:
        return self.elements[i]

    def __add__(self, other: Vector) -> Vector:
        if len(self) != len(other):
            raise ValueError("Vectors must be of the same length")

        return Vector([self[i] + other[i] for i in range(len(self.elements))])

    def __sub__(self, other: Vector) -> Vector:
        if len(self) != len(other):
            raise ValueError("Vectors must be of the same length")

        return Vector([self[i] - other[i] for i in range(len(self.elements))])

    def __mul__(self, scalar: int | float) -> Vector:
        return Vector([x * scalar for x in self.elements])

    def __rmul__(self, scalar: int | float) -> Vector:
        return self.__mul__(scalar)

    def __neg__(self) -> Vector:
        return Vector([-x for x in self.elements])

    def __iter__(self):
        return iter(self.elements)

    def dot(self, other: Vector) -> int | float:
        if len(self) != len(other):
            raise ValueError("Vectors must be of the same length for dot")
        return sum(self[i] * other[i] for i in range(len(self)))

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> Vector:
        v_norm = self.norm()
        if v_norm == 0:
            raise ZeroDivisionError("Cannot normalize a zero vector")
        return self * (1.0 / v_norm)
