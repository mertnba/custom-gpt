import torch
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from evaluation.utils import compute_loss, decode_tokens

def evaluate_model(model, dataloader, device, task="validation", num_steps=20):
    """
    Evaluate the model on a validation dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoaderLite): The dataloader providing validation data.
        device (str): The device to run evaluation on.
        task (str): Task description (default: "validation").
        num_steps (int): Number of steps to evaluate.

    Returns:
        float: The average loss over the evaluation steps.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_steps):
            x, y = dataloader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_steps
    print(f"{task.capitalize()} Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_hellaswag(model, device, tokenizer, num_samples=100):
    """
    Evaluate the model on the HellaSwag dataset.

    Args:
        model (nn.Module): The model to evaluate.
        device (str): The device to run evaluation on.
        tokenizer: The tokenizer to encode and decode tokens.
        num_samples (int): Number of samples to evaluate.

    Returns:
        float: The accuracy on the HellaSwag dataset.
    """
    model.eval()
    num_correct = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        if i >= num_samples:
            break

        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            logits, _ = model(tokens)
            pred = decode_tokens(tokens, mask, logits, tokenizer)

        if pred == label:
            num_correct += 1
        num_total += 1

    accuracy = num_correct / num_total
    print(f"HellaSwag Accuracy: {accuracy:.4f}")
    return accuracy
