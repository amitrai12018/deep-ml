
import numpy as np

def matrix_image(A):
	
	B=np.array(A)
	def rref(A):
		A=np.array(A)
		
		n,m=A.shape

		row=0

		for col in range(m): 
			if col>=n:
				break

			nonzero=-1

			for r in range(row,n): 
				if A[r,col]!=0:
					nonzero=r
					break

			if nonzero==-1:
				continue

			A[[row,nonzero]]=A[[nonzero,row]]
			A[row]=A[row]/A[row,col]

			for r in range(0,n): 
				if r!=row: 
					A[r]=A[r]-A[row]*A[r,col]

			row+=1

		return A
	ans=np.array([])
	# print(ans.shape)
	A=rref(A)
	for i in range(A.shape[0]): 
		for j in range(A.shape[1]): 
			if A[i][j]==1: 
				if ans.shape==(0,): 
					ans=B[:,j].reshape(A.shape[0],1)
					# print(1)
				else:
					ans=np.hstack([ans,B[:,j].reshape(A.shape[0],1)])
	# print(np.hstack([A[:,0],A[:,1]]))




	return ans