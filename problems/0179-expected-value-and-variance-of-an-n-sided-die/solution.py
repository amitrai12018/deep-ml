def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	
	"""
	e=(n+1)/2

	v=0

	for i in range(1,n+1): 
		v+=(i-(n+1)/2)**2
	var=v/n 
	return e,var