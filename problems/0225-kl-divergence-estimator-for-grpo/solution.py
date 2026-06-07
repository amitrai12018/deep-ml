import torch

def kl_divergence_estimator(pi_theta: torch.Tensor, pi_ref: torch.Tensor) -> torch.Tensor:
	"""
	Compute the unbiased KL divergence estimator using PyTorch.
	
	Args:
		pi_theta: Current policy probabilities
		pi_ref: Reference policy probabilities
		
	Returns:
		Per-sample KL divergence estimates
	"""
	r=pi_ref/pi_theta

	return r-torch.log(r)-1
	# Your code here

	
	pass