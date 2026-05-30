import torch
import math

def ucb_action(counts: torch.Tensor, values: torch.Tensor, t: int, c: float) -> int:
    """
    Choose an action using the UCB1 formula.
    Args:
      counts (torch.Tensor): Number of times each action has been chosen
      values (torch.Tensor): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    # TODO: Implement the UCB action selection
    UCB=values + c*(math.log(t)/counts)**(1/2)
    return torch.argmax(UCB).item()