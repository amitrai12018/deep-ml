import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    """
    orhtonormal vectors 

    the vectors that are othogonal and have unit length 

    given 2 d vectors 

    u1=[1,0],u2=[1,1]

    v1=u1

    v2=u2-u2.u1/||U1||^2 U1


    """
    basis=[]

    for i in range(len(vectors)):
        v=np.array(vectors[i],dtype=float)

        for j in range(len(basis)):
            k=np.array(basis[j],dtype=float)
            v-=np.dot(v,k)*k
        
        if np.sqrt(np.dot(v,v))>10**(-10):
            v=v/np.sqrt((np.dot(v,v)))
            basis.append(v)

    return basis
            