import torch

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics using PyTorch.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    import torch

    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)
    correct_answers = torch.as_tensor(
        correct_answers, dtype=torch.long, device=log_probs.device
    )

    # 1. Predictions
    predictions = torch.argmax(log_probs,dim=1)

    # 2. Accuracy
    accuracy = torch.mean((predictions==correct_answers).float())
    # 3. Numerically stable softmax
    m =torch.max(log_probs,dim=1,keepdim=True).values
    shifted = log_probs - m
    exp_logits = torch.exp(shifted)

    # Normalize each question's row independently
    denominator = torch.sum(exp_logits, dim=1, keepdim=True)
    probs = exp_logits / denominator

    # 4. Probability assigned to each correct answer
    row_indices = torch.arange(
        len(correct_answers)
    )
    correct_probs = probs[row_indices, correct_answers]

    # 5. Average correct-answer probability
    avg_correct_prob = correct_probs.mean()

    return {
        "accuracy": accuracy.item(),
        "predictions": predictions.tolist(),
        "avg_correct_prob": avg_correct_prob.item(),
    }