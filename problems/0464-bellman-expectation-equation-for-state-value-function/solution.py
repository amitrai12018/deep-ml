import numpy as np

def bellman_expectation_value(P, R, policy, gamma):
    """
    Compute the state-value function V^pi for a given policy using
    the Bellman expectation equation.
    
    Args:
        P: Transition probabilities, shape (num_states, num_actions, num_states)
        R: Rewards, shape (num_states, num_actions, num_states)
        policy: Stochastic policy, shape (num_states, num_actions)
        gamma: Discount factor
    
    Returns:
        State-value function as numpy array of shape (num_states,)

    MDP : (S,A,y,P,R)

    R:(S,a)
    Pi :(a,S)

    V(s,pi)=R(S,pi)+sigma(P(s'|s,pi)*(V(S',PI)))
    
    V_pi(s) =E_pi[sum y^t Rt+1 |S0=s]
    P(s'|s,a)
    pi(a|s)=P(a|s)
    P(s'|s)=sum(P(s'|s,a)*P(a|s))

    """

    num_states=P.shape[0]
    P_pi=np.sum(policy[:,:,None]*P,axis=1)
    r = np.sum(
        P * R,
        axis=2
    )
    R_pi = np.sum(
        policy * r,
        axis=1
    )
    I = np.eye(num_states)

    V_pi = np.linalg.solve(
        I - gamma * P_pi,
        R_pi
    )

    return V_pi

