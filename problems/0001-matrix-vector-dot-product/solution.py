def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    import numpy as np
    a=np.array(a)
    b=np.array(b)
    try : 
        v=np.dot(a,b.T)
        return v

    except:
        return -1
    return np.dot(a,b.T)
	pass