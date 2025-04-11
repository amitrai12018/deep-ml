import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	# Your code here
	def factorial(k):
		v=1
		for i in range(1,k+1):
			v*=i
		return v

	val=(lam**k *math.exp(-lam)) / math.factorial(k)

	return round(val,5)