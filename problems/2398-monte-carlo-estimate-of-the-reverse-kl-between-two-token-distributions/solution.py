
import numpy as np


# Provided function
def draw_tokens(probs, n, seed):
    return np.random.default_rng(seed).choice(
        len(probs), size=n, p=probs
    )


def validate(logits_a, logits_b):
    a = np.asarray(logits_a, dtype=float)
    b = np.asarray(logits_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1-D")

    if a.size == 0 or a.shape != b.shape:
        raise ValueError("Invalid input shapes")

    return a, b


def log_softmax(z):
    m = np.max(z)
    shifted = z - m

    return shifted - np.log(np.sum(np.exp(shifted)))


def exact_kl(logits_a, logits_b):
    a, b = validate(logits_a, logits_b)

    log_p = log_softmax(a)
    log_q = log_softmax(b)

    p = np.exp(log_p)

    kl = np.sum(p * (log_p - log_q))

    return round(float(kl), 4)


def mc_kl(logits_a, logits_b, n, seed):
    a, b = validate(logits_a, logits_b)

    if n < 1:
        raise ValueError("n must be >= 1")

    log_p = log_softmax(a)
    log_q = log_softmax(b)

    p = np.exp(log_p)

    selection = draw_tokens(p, n, seed)

    reverse_kl = np.mean(
        log_p[selection] - log_q[selection]
    )

    return round(float(reverse_kl), 4)


def kl_report(policy_logits, ref_logits, n, seed):
    return {
        "reverse_exact": exact_kl(
            policy_logits, ref_logits
        ),
        "reverse_mc": mc_kl(
            policy_logits, ref_logits, n, seed
        ),
        "forward_exact": exact_kl(
            ref_logits, policy_logits
        ),
        "forward_mc": mc_kl(
            ref_logits, policy_logits, n, seed
        )
    }
