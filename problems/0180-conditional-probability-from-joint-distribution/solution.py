def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)=P(A)
	"""
	# Your code here
	p = joint_distribution[('A', 'B')] / (
        joint_distribution[('A', 'B')] +
        joint_distribution[('`A', 'B')]
    )

    return p