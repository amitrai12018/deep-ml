
import math


def product_prob(token_probs):
    if len(token_probs) == 0:
        return 1.0

    p = 1.0
    for i in token_probs:
        if not (0 < i <= 1):
            raise ValueError("Invalid probability")
        p *= i

    return p


def raw_log_prob(token_probs):
    p = 0.0

    for i in token_probs:
        if not (0 < i <= 1):
            raise ValueError("Invalid probability")
        p += math.log(i)

    return p


def sum_log_prob(token_probs):
    return round(raw_log_prob(token_probs), 4)


def compare_sequences(probs_a, probs_b):
    a = product_prob(probs_a)
    b = product_prob(probs_b)

    prod = "tie"
    if a > b:
        prod = "a"
    elif a < b:
        prod = "b"

    log_a = raw_log_prob(probs_a)
    log_b = raw_log_prob(probs_b)

    s = "tie"
    if abs(log_a - log_b) > 1e-9:
        if log_a > log_b:
            s = "a"
        else:
            s = "b"

    u = [a == 0.0, b == 0.0]

    return {
        'by_product': prod,
        'by_log': s,
        'underflow': u
    }
