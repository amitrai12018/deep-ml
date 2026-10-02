def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:


	import numpy as np
	if mode=="column":
		return np.mean(matrix,axis=0)

	if mode=="row":
		return np.mean(matrix,axis=1)