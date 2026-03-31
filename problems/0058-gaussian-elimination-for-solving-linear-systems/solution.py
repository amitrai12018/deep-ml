import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	b=b.reshape((b.shape[0],1))
	# print(b.shape)
	A=np.hstack([A,b])
	# print(A)
	row=0
	for col in range(A.shape[1]): 
		if col>=A.shape[0]:
			break

		nonzero=-1

		for r in range(row,A.shape[0]): 
			if A[r][col]!=0: 
				nonzero=r

				break

		if nonzero==-1:
			continue 

		A[[row,nonzero]]=A[[nonzero,row]]

		A[row]=A[row]/A[row,col]

		for r in range(0,A.shape[0]): 
			if r!=row: 
				A[r]=A[r]-A[row]*A[r,col]

		row+=1
	# print(A)
	return A[:,-1]
		
		