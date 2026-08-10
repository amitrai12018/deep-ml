import numpy as np

def simulate_mdp(
    P,
    R,
    policy,
    start_state,
    terminal_states,
    gamma,
    num_episodes,
    max_steps,
    seed=42,
):
    """
    Simulate episodes in an MDP and compute discounted returns.

    Args:
        P: Transition probabilities, shape (n_states, n_actions, n_states)
        R: Rewards, shape (n_states, n_actions, n_states)
        policy: Stochastic policy, shape (n_states, n_actions)
        start_state: Initial state for each episode
        terminal_states: List of terminal state indices
        gamma: Discount factor
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility

    Returns:
        Tuple of (episode_returns, average_return)
    """
    np.random.seed(seed)

    terminal_states = set(terminal_states)
    episode_returns = []

    for _ in range(num_episodes):
        state = start_state
        discounted_return = 0.0

        for step in range(max_steps):
            # Stop immediately if the current state is terminal.
            if state in terminal_states:
                break

            # Sample an action from the stochastic policy.
            action = np.random.choice(
                P.shape[1],
                p=policy[state]
            )

            # Sample the next state from the transition probabilities.
            next_state = np.random.choice(
                P.shape[2],
                p=P[state, action]
            )

            # Accumulate the discounted immediate reward.
            discounted_return += (gamma ** step) * R[state, action, next_state]

            # Move to the next state.
            state = next_state

            # Episode ends immediately upon entering a terminal state.
            if state in terminal_states:
                break

        episode_returns.append(round(float(discounted_return), 4))

    average_return = round(float(np.mean(episode_returns)), 4)

    return episode_returns, average_return