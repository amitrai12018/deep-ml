import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float

    e^((ln(p1*p2*...pn)^(-1/n)))

    = 1/(p1*p2....pn)1/n
    """
    p=1
    for i in range(len(probabilities)):
        p*=probabilities[i]
    
    return 1/p**(1/len(probabilities))


