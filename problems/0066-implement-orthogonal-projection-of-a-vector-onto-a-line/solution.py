
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
    val=[i**2 for i in L]
    va=sum(val)
    # pfint(val)
    va=va**(1/2)
    
    import numpy as np
    V=np.array(v)
    L=np.array(L)
    ans=np.multiply(V,L)
   
    return ans/va
	pass
