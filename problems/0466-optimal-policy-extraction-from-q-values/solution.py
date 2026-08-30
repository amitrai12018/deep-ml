import numpy as np

def extract_optimal_policy(Q: np.ndarray) -> dict:
    """
    Extract the optimal policy, state-value function, and advantage
    function from a Q-value table.
    
    Args:
        Q: Q-value table of shape (num_states, num_actions)
    
    Returns:
        Dictionary with keys:
        - 'optimal_actions': list of int (optimal action per state)
        - 'state_values': list of float (V*(s) per state)
        - 'advantages': nested list of float (A(s,a) for all pairs)

    Q[s][a] : E[y^t|s,a]]
    
    optimal actions: argmax(Q,axis=0)

    V*(s)=max_value

    advantage:

    """
    optimal_actions=np.argmax(Q,axis=1)
    state_values=np.max(Q,axis=1)
    advantage=np.round(np.subtract(Q,np.max(Q,axis=1,keepdims=True)),4)
    return {'optimal_actions':optimal_actions.tolist(),'state_values': state_values.tolist(),'advantages': advantage.tolist()}
