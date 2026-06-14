import torch
from collections import Counter

def pass_at_1(responses_correct: torch.Tensor) -> float:
	"""
	Compute pass@1 using PyTorch.
	
	Args:
		responses_correct: Boolean tensor
		
	Returns:
		pass@1 score
	"""
	ans=0
	for i in range(responses_correct.shape[0]): 
		if responses_correct[i]==True:
			ans+=1

	length=responses_correct.shape[0]
	return ans/length

def majority_voting(responses: list[str]) -> str:
	"""
	Return most common response.
	"""
	from collections import Counter
	d=Counter(responses)
	val=-1
	count=-1
	for k,v in d.items(): 
		if v>val: 
			val=v
			count=k

	print(count,val)
	return count





def pass_at_k(n: int, c: int, k: int) -> float:
	"""
	Compute unbiased pass@k estimator.
	(n-c-k+1)* .. (n-c)/* (n-k+1)...n()
	"""
	numerator=1
	for i in range(n-c-k+1,n-c+1): 
		numerator*=i
	deno=1
	for i in range(n-k+1,n+1): 
		deno*=i 
	return 1-numerator/deno
	

