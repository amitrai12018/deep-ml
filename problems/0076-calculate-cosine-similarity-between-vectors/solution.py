import numpy as np

# from scipy import linalg, sparse
def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	dot=np.dot(v1,v2)
	modv1=(np.linalg.norm(v1))
	modv2=np.linalg.norm(v2)
	return dot/(modv1*modv2)

	# Implement your code here
	pass