import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

	A=np.array(A)
	T=np.array(T)
	S=np.array(S)
	try: 
		inv1= np.linalg.inv(T)
		inv2=np.linalg.inv(S)
		res1= np.matmul(inv1,A)
		res2=np.matmul(res1,S)
		return res2
	except:
		return -1

	