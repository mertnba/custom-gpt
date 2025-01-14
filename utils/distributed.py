import os
import torch
import torch.distributed as dist

def setup_distributed():
    """
    Set up the environment for distributed training.

    Returns:
        dict: A dictionary containing rank, local_rank, world_size, and is_distributed status.
    """
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = rank != -1

    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        print(f"Distributed training initialized. Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    else:
        print("Running in non-distributed mode.")

    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_distributed': is_distributed
    }

def cleanup_distributed():
    """
    Clean up resources used for distributed training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed training resources cleaned up.")

def synchronize_gradients(loss_accum, op=dist.ReduceOp.SUM):
    """
    Synchronize gradients across all processes.

    Args:
        loss_accum (Tensor): The accumulated loss to synchronize.
        op (ReduceOp): Reduction operation (default: SUM).
    """
    if dist.is_initialized():
        dist.all_reduce(loss_accum, op=op)

def broadcast_parameters(model):
    """
    Broadcast model parameters from rank 0 to all other ranks.

    Args:
        model (nn.Module): The PyTorch model whose parameters are broadcast.
    """
    if dist.is_initialized():
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

def average_gradients(model):
    """
    Averages gradients across all processes.

    Args:
        model (nn.Module): The PyTorch model whose gradients are averaged.
    """
    if dist.is_initialized():
        world_size = dist.get_world_size()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
