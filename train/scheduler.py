import math

def get_lr(it, max_lr=6e-4, min_lr=6e-5, warmup_steps=715, max_steps=19073):
    """
    Compute the learning rate for the current training iteration using cosine decay.

    Args:
        it (int): Current training iteration.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Total number of steps.

    Returns:
        float: Computed learning rate for the iteration.
    """
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0, min(1, decay_ratio))  # Clamp between 0 and 1
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine_coeff * (max_lr - min_lr)
