import os
import json

class Logger:
    def __init__(self, log_dir, master_process=True):
        """
        Initialize the logger.

        Args:
            log_dir (str): Directory to save logs.
            master_process (bool): Whether this is the main process in distributed training.
        """
        self.master_process = master_process
        self.log_file = os.path.join(log_dir, "log.txt")
        os.makedirs(log_dir, exist_ok=True)
        if master_process:
            with open(self.log_file, "w") as f:
                f.write("")  # Clear the log file

    def log_metrics(self, step, train_loss, val_loss, lr):
        """
        Log training and validation metrics to the log file.

        Args:
            step (int): Current training step.
            train_loss (float): Training loss.
            val_loss (float): Validation loss.
            lr (float): Current learning rate.
        """
        if self.master_process:
            metrics = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(f"Step {step}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {lr:.6e}")

    def close(self):
        """Close the logger (if needed for cleanup)."""
        pass
