import torch
from torch.nn import functional as F

def compute_loss(logits, targets):
    """
    Compute the cross-entropy loss.

    Args:
        logits (Tensor): The model's output logits.
        targets (Tensor): The target labels.

    Returns:
        Tensor: The computed loss.
    """
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def decode_tokens(tokens, mask, logits, tokenizer):
    """
    Decode the most likely token sequence from the logits.

    Args:
        tokens (Tensor): The input tokens.
        mask (Tensor): The attention mask for the tokens.
        logits (Tensor): The model's output logits.
        tokenizer: The tokenizer to decode tokens.

    Returns:
        int: The predicted label.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)

    shift_loss = F.cross_entropy(flat_logits, flat_tokens, reduction="none")
    shift_loss = shift_loss.view(tokens.size(0), -1)

    # Average the loss over the completion region (mask == 1)
    masked_loss = shift_loss * mask[..., 1:].contiguous()
    avg_loss = masked_loss.sum(dim=1) / mask[..., 1:].sum(dim=1)

    # Predicted label is the one with the lowest loss
    return avg_loss.argmin().item()
