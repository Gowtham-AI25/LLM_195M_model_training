import math
import torch 
from torch.optim.lr_scheduler import LambdaLR

def Scheduler_4phase(optimizer: torch.optim.Optimizer, train_config) -> LambdaLR:
    """
    Create a 4-phase cosine annealing learning rate scheduler with warmup.

    Phases:
    1. Linear Warmup
    2. Plateau
    3. Cosine Decay
    4. Linear Anneal to Minimum LR

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        train_config: Training configuration containing scheduler parameters.

    Returns:
        LambdaLR: The learning rate scheduler.
    """
    W = train_config.warmup_steps
    P = train_config.plateau_steps
    D = train_config.decay_steps
    A = train_config.anneal_steps

    min_lr_ratio = train_config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        # Phase 1: Warmup
        if step < W:
            return step / max(1, W)

        # Phase 2: Plateau
        if step < W + P:
            return 1.0

        # Phase 3: Cosine decay
        if step < W + P + D:
            u = (step - W - P) / max(1, D)
            return (
                min_lr_ratio
                + (1.0 - min_lr_ratio)
                * 0.5 * (1.0 + math.cos(math.pi * u))
            )

        # Phase 4: Linear anneal to zero
        if step < W + P + D + A:
            k = (step - (W + P + D)) / max(1, A)
            return min_lr_ratio * (1.0 - k)

        # After training
        return 0.0
    
    return LambdaLR(optimizer, lr_lambda)