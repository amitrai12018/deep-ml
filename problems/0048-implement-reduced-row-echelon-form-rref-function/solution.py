import numpy as np

def rref(matrix):
	A=np.array(matrix,dtype=float)
	B=A.copy()

	n,m= A.shape
	
	row=0
	for col in range(m): 
		# print(col)
		if col>=n: 
			break
		
		nonzero=-1

		for r in range(row,n): 
			# print(A[r][col])
			if A[r][col]!=0: 
				nonzero=r
				# print(nonzero,"dfladjf")
				break 
		
		if nonzero==-1:
			continue
		# print(nonzero,"nonzero")
		A[[row,nonzero]]=A[[nonzero,row]]
		
		A[row]=A[row]/A[row,col]
		# print(A[row])
		for j in range(n):
			if j!=row: 
				A[j]=A[j]-A[row]*A[j,col]
		
		row+=1
	
	return A

			



