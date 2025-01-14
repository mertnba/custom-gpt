import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from train.scheduler import get_lr
from train.logger import Logger
from model.gpt import GPT
from model.config import GPTConfig
from data.dataloader import DataLoaderLite

def train_model():
    # Distributed setup
    ddp_rank = int(os.environ.get('RANK', -1))
    ddp = ddp_rank != -1
    master_process = ddp_rank == 0 if ddp else True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Model and data setup
    config = GPTConfig(vocab_size=50304, block_size=1024)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
    train_loader = DataLoaderLite(B=64, T=1024, process_rank=ddp_rank, num_processes=1, split="train")
    val_loader = DataLoaderLite(B=64, T=1024, process_rank=ddp_rank, num_processes=1, split="val")

    if ddp:
        model = DDP(model, device_ids=[ddp_rank])

    raw_model = model.module if ddp else model

    # Logging setup
    logger = Logger(log_dir="log", master_process=master_process)

    # Training loop
    max_steps = 19073
    grad_accum_steps = 8
    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation and logging
        if step % 250 == 0 or step == max_steps - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(20):  # Validation steps
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= 20

            if master_process:
                logger.log_metrics(step=step, train_loss=loss_accum.item(), val_loss=val_loss, lr=get_lr(step))
    logger.close()

if __name__ == "__main__":
    train_model()
