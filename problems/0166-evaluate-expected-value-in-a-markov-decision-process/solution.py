import numpy as np

def expected_action_value(state, action, P, R, V, gamma):
    """
    Computes the expected value of taking `action` in `state` for the given MDP.
    Args:
      state: int or str, the current state
      action: str, the chosen action
      P: dict of dicts, P[s][a][s'] = prob of next state s' if a in s
      R: dict of dicts, R[s][a][s'] = reward for (s, a, s')
      V: np.ndarray, the value function vector, indexed by state
      gamma: float, discount factor
    Returns:
      float: expected value
    """
    # Your code here
    Q=0
    states=list(P.keys())
    # print(states)
    
    for s_prime in states: 
      # print(P[state][action][s_prime])
      # print(R[state][action][s_prime])
      # print(V[s_prime])
      Q+=P[state][action][s_prime]*(R[state][action][s_prime]+gamma*V[s_prime])
    
    return Q
    # return 1
