import math


class Vector:
    def __init__(self, elements: list[float]) -> None:
        self.elements = elements

    def __str__(self) -> str:
        return f"Vector({[round(x, 4) for x in self.elements]})"

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, i: int) -> float:
        if i >= len(self) or i < 0:
            raise IndexError("Index out of bounds")
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
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector([x * scalar for x in self.elements])

    def __rmul__(self, scalar: int | float) -> Vector:
        return self.__mul__(scalar)

    def __neg__(self) -> Vector:
        return Vector([-x for x in self.elements])

    def dot(self, other: Vector) -> float:
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
