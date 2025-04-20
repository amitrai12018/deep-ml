def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    let v is vector in B
    we want to see how would it look in C

    so first lets find out how it would look in our coordinate system

    Bv-> v is in B space, Bv is how it will look in our space

    now lets find out how Bv will look in C's coordinate space

    let x is a vector which seen from our space is Bv
    so Cx=Bv

    x=C^-1.Bv
    x=Pv
    P=C^-1B

    """
    import numpy as np
    B=np.array(B)
    C=np.array(C)
    P=np.matmul(np.linalg.inv(C),B)
	return P