import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
    """
    Compute temperature-scaled softmax probabilities from logits.
    
    Args:
        logits: 1D numpy array of raw model output scores
        temperature: float controlling distribution sharpness
    
    Returns:
        List of probabilities after temperature scaling
    """
    if temperature<=0:
        m=np.argmax(logits)
        ans=np.zeros(logits.shape)
        ans[m]=1
        return ans
    
    else: 
        logits=logits/temperature
        a=logits[np.argmax(logits)]
        probs=np.exp(logits-a)/np.sum(np.exp(logits-a))
        return probs

