import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	"""
	"""
	x_ini=np.zeros(b.shape)
	A=np.array(A)
	b=np.array(b)
	x=np.array(x_ini)
	# lower_tri=
	# x

	for i in range(n): 
		x_cpy=x
		for i in range(b.shape[0]): 
			x_cpy[i]=1/np.diag(A)[i]*(b[i]-np.sum(np.tril(A,k=-1)*x_cpy.T,axis=1)[i]-np.sum(np.triu(A,k=1)*x.T,axis=1)[i])
		x=x_cpy

	return x

