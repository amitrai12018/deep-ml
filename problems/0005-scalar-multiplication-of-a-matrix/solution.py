def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    import numpy as np
    return scalar*np.array(matrix)
	return result