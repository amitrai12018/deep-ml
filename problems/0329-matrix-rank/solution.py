import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    A = A.astype(float)
    m, n = A.shape

    row = 0

    for col in range(n):
        if row >= m:
            break

        # Find pivot (partial pivoting)
        pivot = np.argmax(np.abs(A[row:, col])) + row

        if abs(A[pivot, col]) < tol:
            continue  # no pivot in this column

        # Swap rows
        A[[row, pivot]] = A[[pivot, row]]

        # Eliminate below
        for i in range(row + 1, m):
            factor = A[i, col] / A[row, col]
            A[i] -= factor * A[row]

        row += 1

    # Rank = number of non-zero rows
    rank = 0
    for i in range(m):
        if np.any(np.abs(A[i]) > tol):
            rank += 1

    return rank