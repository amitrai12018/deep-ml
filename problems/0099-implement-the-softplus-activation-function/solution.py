def softplus(x: float) -> float:
	"""
	Compute the softplus activation function.

	Args:
		x: Input value

	Returns:
		The softplus value: log(1 + e^x)
	"""
    import math
    return round(math.log(1+math.exp(x)),4)
	# Your code here
	pass
	return round(val,4)