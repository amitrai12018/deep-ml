import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	# Your code here
	ans=[]

	for i in range(len(data)): 
		a=[]
		for j in range(degree+1): 
			a.append(data[i]**j)
		
		ans.append(a)

	return ans