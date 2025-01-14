import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """
    Set up the environment for distributed training.

    Returns:
        dict: A dictionary containing rank, local_rank, and world_size.
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
    Clean up distributed training resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed training resources cleaned up.")
