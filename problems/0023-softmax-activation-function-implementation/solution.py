import math

def softmax(scores: list[float]) -> list[float]:
	# Your code he
    """

    """
    import numpy as np
    v=np.array(scores)
    v=np.exp(v-np.max(v))/np.sum(np.exp(v-np.max(v)))
    return v
	