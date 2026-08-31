def greedy_policy_improvement(V: dict, transitions: dict, gamma: float) -> tuple:
    """
    Perform greedy policy improvement given a value function and MDP model.
    
    Args:
        V: dict mapping state -> float (state-value function)
        transitions: dict mapping (state, action) -> list of (probability, next_state, reward)
        gamma: float, discount factor
    
    Returns:
        tuple: (policy, Q)
            policy: dict mapping state -> best action
            Q: dict mapping (state, action) -> float
    Greedy Policy improvement 
    value function 
    V[s]=Expected reward
    transitions[(state,action)]=[(transition probability, next state,reaward)]
    gamma

    Q(s,a)=E[y^t*R_t|(s,a)] = max(a(sigma s' (P(s',s,a)[R(s',s,a)+yV)(s')])

    """
    Q = {}
    policy = {}

    # Compute Q(s, a) for every state-action pair
    for (state, action), outcomes in transitions.items():
        q_value = 0.0

        for probability, next_state, reward in outcomes:
            q_value += probability * (
                reward + gamma * V.get(next_state, 0.0)
            )

        Q[(state, action)] = q_value

    # Group available actions by state
    actions_by_state = {}

    for state, action in transitions:
        if state not in actions_by_state:
            actions_by_state[state] = []

        actions_by_state[state].append(action)

    # Choose greedy action
    # sorted() ensures lexicographic tie-breaking
    for state, actions in actions_by_state.items():
        policy[state] = max(
            sorted(actions),
            key=lambda action: Q[(state, action)]
        )

    return policy, Q