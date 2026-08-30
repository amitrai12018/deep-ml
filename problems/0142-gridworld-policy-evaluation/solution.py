def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats

    policy: P[a|s] at each position, going to up down left right has equal probability of 0.25

    gamma = 0.9

    R(s',a,s)=-1

    iterate 

    V(s)=E[y^t*R]= pi(a|s) X p(s'|s,a)[R(s',s,a)+gamma(V(s'))]


    """
    V = [[0.0 for _ in range(5)] for _ in range(5)]

    terminal = {(0, 0), (0, 4), (4, 0), (4, 4)}

    actions = ["up", "down", "left", "right"]

    def next_state(i, j, action):

        if action == "up":
            return max(i - 1, 0), j

        if action == "down":
            return min(i + 1, 4), j

        if action == "left":
            return i, max(j - 1, 0)

        if action == "right":
            return i, min(j + 1, 4)

    while True:

        new_V = [[0.0 for _ in range(5)] for _ in range(5)]

        delta = 0

        for i in range(5):
            for j in range(5):

                if (i, j) in terminal:
                    continue

                v = 0

                for action in actions:

                    ni, nj = next_state(i, j, action)

                    v += policy[(i, j)][action] * (
                        -1 + gamma * V[ni][nj]
                    )

                new_V[i][j] = v

                delta = max(delta, abs(v - V[i][j]))

        V = new_V

        if delta < threshold:
            break

    return V   
                    
