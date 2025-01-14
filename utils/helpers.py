import torch

def validate_tensor(tensor, name="tensor"):
    """
    Validates if a tensor contains NaN or Inf values.

    Args:
        tensor (Tensor): The tensor to validate.
        name (str): The name of the tensor (for error reporting).

    Raises:
        ValueError: If the tensor contains NaN or Inf values.
    """
    if torch.any(torch.isnan(tensor)):
        raise ValueError(f"{name} contains NaN values.")
    if torch.any(torch.isinf(tensor)):
        raise ValueError(f"{name} contains Inf values.")

def calculate_parameters(model):
    """
    Calculate the number of trainable parameters in a model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, step, path):
    """
    Save a model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer state to save.
        step (int): Current training step.
        path (str): File path to save the checkpoint.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }, path)
    print(f"Checkpoint saved at {path}")
