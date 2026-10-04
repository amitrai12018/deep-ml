def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.

    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis

    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """

    # P(H_i) * P(E | H_i)
    weighted = [
        prior * likelihood
        for prior, likelihood in zip(priors, likelihoods)
    ]

    # P(E) = sum P(E | H_i)P(H_i)
    evidence = sum(weighted)

    if evidence == 0:
        raise ValueError("Evidence has probability 0.")

    # P(H_i | E)
    posteriors = [
        value / evidence
        for value in weighted
    ]

    return posteriors