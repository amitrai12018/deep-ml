import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	"""

	softmax : 

	[1,2,3]
	
	np.exp(A)
	"""
	scores_stable=scores-np.max(scores)
	# print(np.max(scores))
	exp_scores=np.exp(scores_stable)
	softmax=exp_scores/np.sum(exp_scores)
	return np.log(softmax)