import numpy as np

def make_diagonal(x):
	# print(np.shape(x)[0])
	identity=np.identity(np.shape(x) [0])
	# print(identity)
	"""
	3,1 3*3 

	"""
	return  identity * x.T