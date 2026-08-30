import numpy as np

def compare_policy_evaluation(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float, n_sweeps: int, method: str) -> np.ndarray:
    """
    Perform policy evaluation sweeps using the specified update method.
    
    Args:
        V: np.ndarray of shape (n_states,), initial value function
        P: np.ndarray of shape (n_states, n_states), transition probability matrix under the policy
        R: np.ndarray of shape (n_states,), expected immediate reward for each state
        gamma: float, discount factor
        n_sweeps: int, number of sweeps to perform
        method: str, either 'synchronous' or 'in_place'
    
    Returns:
        np.ndarray: updated value function after n_sweeps
    """
    V = V.copy()   # don't modify the original

    for _ in range(n_sweeps):

        if method == "synchronous":
            V = R + gamma * P @ V

        elif method == "in_place":
            for s in range(len(V)):
                V[s] = R[s] + gamma * np.dot(P[s], V)

        else:
            raise ValueError("method must be 'synchronous' or 'in_place'")

    return V