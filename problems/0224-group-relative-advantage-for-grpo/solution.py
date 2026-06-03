import torch

def compute_group_relative_advantage(rewards: torch.Tensor) -> torch.Tensor:
	"""
	Compute the Group Relative Advantage for GRPO using PyTorch.
	
	Args:
		rewards: 1D tensor of rewards for a group of outputs
		
	Returns:
		1D tensor of normalized advantages
	"""
	# Your code here
	mean = rewards.mean()
	std = rewards.std(unbiased=False)

	if std < 1e-8:
		return torch.zeros_like(rewards)

	return (rewards - mean) / std
	pass