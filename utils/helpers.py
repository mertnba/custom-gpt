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

def load_checkpoint(model, optimizer, path):
    """
    Load a model checkpoint.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (Optimizer): The optimizer to load the state into.
        path (str): File path to the checkpoint.

    Returns:
        int: The training step loaded from the checkpoint.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f"Checkpoint loaded from {path}")
    return step
