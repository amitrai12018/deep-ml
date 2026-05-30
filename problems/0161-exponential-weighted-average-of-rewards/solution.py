def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k=len(rewards)
    result = Q1 * ( 1 - alpha)**k

    for i in range(k): 
        result+= alpha * (1-alpha)**(k-i-1) * rewards[i]

    return round(result,4)
    pass
