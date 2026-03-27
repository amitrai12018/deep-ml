import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

	'''
	
	x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i))

	x=vector[1/aii]*(vector(b)- matrix(a without diagonal elements) @ matrix(x other than ith element))
	'''
	x=np.zeros(b.shape[0])
	# hstack=x
	# # print(hstack,"adfa")
	# for i in range(x.shape[0]-1): 
	# 	hstack=np.vstack((hstack,x))
		
	# hstack=hstack-np.eye(hstack.shape[0])*x
	# print(hstack.shape)
	# print(x.shape)
	# print((A-np.eye(A.shape[0])*np.diag(A)))

	for i in range(n):
		x=1/np.diag(A)*(b-((A-np.eye(A.shape[0])*np.diag(A)) @ x ))
	

	return np.round(x,4)