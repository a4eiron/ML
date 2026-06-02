from __future__ import annotations
from typing import override
from .vector import Vector


class Matrix:
    elements: list[list[int | float]] = []

    def __init__(self, elements: list[list[int | float]]) -> None:
        row_len = len(elements[0])
        for i in range(1, len(elements)):
            if len(elements[i]) != row_len:
                raise TypeError("All rows must have the same length")
        self.elements = elements

    @override
    def __str__(self) -> str:
        rows_str = ",\n ".join((str(row) for row in self.elements))
        return f"Matrix([\n {rows_str}\n])"

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, i: int) -> list[int | float]:
        return self.elements[i]

    def shape(self) -> tuple[int, int]:
        return (len(self.elements), len(self.elements[0]))

    def row(self, i: int) -> Vector:
        """row i as a vector object"""
        return Vector(self[i])

    def col(self, j: int) -> Vector:
        """col j as a vector object"""
        return Vector([row[j] for row in self.elements])

    def transpose(self) -> Matrix:
        return Matrix(
            [[row[j] for row in self.elements] for j in range(self.shape()[1])]
        )

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape() != other.shape():
            raise ValueError("Matrices must be of the same shape to add")

        result = [(self.row(i) + other.row(i)).elements for i in range(len(self))]
        return Matrix(result)

    def __mul__(self, scalar: int | float) -> Matrix:
        return Matrix([[val * scalar for val in row] for row in self.elements])

    def __rmul__(self, scalar: int | float) -> Matrix:
        return self.__mul__(scalar)

    def __matmul__(self, other: Matrix | Vector) -> Matrix | Vector:
        if isinstance(other, Matrix):
            if self.shape()[1] != other.shape()[0]:
                raise ValueError("Matrix multiplication is not possible")

            result_mat = [
                [self.row(i).dot(other.col(j)) for j in range(other.shape()[1])]
                for i in range(self.shape()[0])
            ]
            return Matrix(result_mat)
        else:
            if self.shape()[1] != len(other):
                raise ValueError("Matrix columns must match Vector length")

            result_vec = [self.row(i).dot(other) for i in range(self.shape()[0])]
            return Vector(result_vec)


def matmul_rowcol(A: Matrix, B: Matrix) -> Matrix:
    if A.shape()[1] != B.shape()[0]:
        raise ValueError("Matrix multiplication not possible for the given matrices")

    result = [
        [A.row(i).dot(B.col(j)) for j in range(B.shape()[1])]
        for i in range(A.shape()[0])
    ]
    return Matrix(result)


def vector_outer_product(u: Vector, v: Vector) -> Matrix:
    return Matrix([[u_val * v_val for v_val in v.elements] for u_val in u.elements])


def matmul_outer(A: Matrix, B: Matrix) -> Matrix:
    if A.shape()[1] != B.shape()[0]:
        raise ValueError("Matrix multiplication not possible for the given matrices")

    m = A.shape()[0]
    q = B.shape()[1]
    n = A.shape()[1]

    zero_elements = [[0.0 for _ in range(q)] for _ in range(m)]
    result = Matrix(zero_elements)

    for k in range(n):
        layer = vector_outer_product(A.col(k), B.row(k))
        result = result + layer
    return result


def matmul_rowwise(A: Matrix, B: Matrix) -> Matrix:
    if A.shape()[1] != B.shape()[0]:
        raise ValueError("Matrix multiplication not possible for the given matrices")

    m = A.shape()[0]

    result: list[list[float]] = []

    for i in range(m):
        combined_row = Vector([0.0] * B.shape()[1])
        for k in range(len(A[i])):
            scalar = A[i][k]
            row_of_B = B.row(k)
            combined_row = combined_row + (row_of_B * scalar)
        result.append(combined_row.elements)

    return Matrix(result)
