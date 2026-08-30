import numpy as np

def bellman_q_value(states: list, actions: list, transition_probs: dict, rewards: dict, policy: dict, gamma: float, theta: float = 1e-6) -> dict:
    """
    Compute Q^pi(s, a) for all state-action pairs using iterative policy evaluation
    based on the Bellman expectation equation for action-values.

    Args:
        states: List of state identifiers : S: {s1...sn}
        actions: List of action identifiers A: {a1...an}
        transition_probs: Dict mapping (s, a, s') -> P(s'|s,a) : P[s'][a][s] -> prob of moving from s to s' using actiona 
        rewards: Dict mapping (s, a, s') -> R(s,a,s'): R[s'][a][s] reward gained when moving from s to s' using action a
        policy: Dict mapping (s, a) -> pi(a|s) Prob of taking action a standing in s
        gamma: Discount factor y
        theta: Convergence threshold 

    Returns:
        Dict mapping (s, a) -> Q-value rounded to 4 decimal places

        Q(s,a)=E[G(s,a)]
               E[R(s,a)+gamma*G(s'|a,s)]
               R[s,a,s]+ gamma*E[G(s'|a,s)]

    Q(s,a)=E(G|s,a) = E(R(s',a,s)+ yRt+1(s') + r^2(s''))
                    = E[R(s',a,s)]+y E[G(s')]
                    = sum(P(s'|s)*R(s',a,s))+ y V_pi(s') 
    """
    # ---------------------------------------------------------
    # Number of states and actions
    # ---------------------------------------------------------

    num_states = len(states)
    num_actions = len(actions)

    # ---------------------------------------------------------
    # Map state/action identifiers to NumPy array indices
    #
    # Example:
    # states  = ['A', 'B']
    #          -> {'A': 0, 'B': 1}
    #
    # actions = ['L', 'R']
    #          -> {'L': 0, 'R': 1}
    # ---------------------------------------------------------

    state_to_idx = {
        state: i for i, state in enumerate(states)
    }

    action_to_idx = {
        action: i for i, action in enumerate(actions)
    }

    # ---------------------------------------------------------
    # Convert policy dictionary into a matrix
    #
    # policy_matrix.shape = (num_states, num_actions)
    #
    # policy_matrix[s, a] = pi(a | s)
    # ---------------------------------------------------------

    policy_matrix = np.zeros(
        (num_states, num_actions)
    )

    for s in states:
        for a in actions:
            s_idx = state_to_idx[s]
            a_idx = action_to_idx[a]

            policy_matrix[s_idx, a_idx] = policy.get(
                (s, a),
                0.0
            )

    # ---------------------------------------------------------
    # Initialize Q_0(s,a) = 0
    #
    # Q.shape = (num_states, num_actions)
    #
    #             action
    #             a0     a1
    # state s0    Q00    Q01
    #       s1    Q10    Q11
    # ---------------------------------------------------------

    Q = np.zeros(
        (num_states, num_actions)
    )

    # ---------------------------------------------------------
    # Iterative policy evaluation
    # ---------------------------------------------------------

    while True:

        # -----------------------------------------------------
        # Calculate V_k(s')
        #
        # V_k(s') = sum_a pi(a|s') Q_k(s',a)
        #
        # policy_matrix : (S, A)
        # Q             : (S, A)
        #
        # element-wise multiplication:
        #
        # (S,A) * (S,A) -> (S,A)
        #
        # sum over actions:
        #
        # (S,A) -> (S,)
        # -----------------------------------------------------

        V = np.sum(
            policy_matrix * Q,
            axis=1
        )

        # V.shape = (num_states,)

        # -----------------------------------------------------
        # Create Q_{k+1}
        # -----------------------------------------------------

        new_Q = np.zeros_like(Q)

        # -----------------------------------------------------
        # Calculate each Q(s,a)
        # -----------------------------------------------------

        for s in states:

            s_idx = state_to_idx[s]

            for a in actions:

                a_idx = action_to_idx[a]

                q_value = 0.0

                # -------------------------------------------------
                # Bellman equation:
                #
                # Q(s,a) =
                #   sum_s' P(s'|s,a)
                #   [R(s,a,s') + gamma * V(s')]
                #
                # We are calculating ONE scalar Q(s,a) here.
                # -------------------------------------------------

                for s_prime in states:

                    # P(s' | s,a)
                    p = transition_probs.get(
                        (s, a, s_prime),
                        0.0
                    )

                    # R(s,a,s')
                    reward = rewards.get(
                        (s, a, s_prime),
                        0.0
                    )

                    # V(s')
                    s_prime_idx = state_to_idx[s_prime]

                    # Add contribution from this s'
                    q_value += p * (
                        reward
                        + gamma * V[s_prime_idx]
                    )

                # Store Q_{k+1}(s,a)
                new_Q[s_idx, a_idx] = q_value

        # -----------------------------------------------------
        # Check convergence
        #
        # delta =
        # max |Q_{k+1}(s,a) - Q_k(s,a)|
        # -----------------------------------------------------

        delta = np.max(
            np.abs(new_Q - Q)
        )

        # Move to next iteration
        Q = new_Q

        # Stop when maximum change < theta
        if delta < theta:
            break

    # ---------------------------------------------------------
    # Convert Q matrix back to requested dictionary format
    # ---------------------------------------------------------

    q_values = {}

    for s in states:
        for a in actions:

            s_idx = state_to_idx[s]
            a_idx = action_to_idx[a]

            q_values[(s, a)] = round(
                Q[s_idx, a_idx],
                4
            )

    return q_values