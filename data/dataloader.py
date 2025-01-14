import os
import torch
from data.utils import load_tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        """
        Lightweight data loader for handling tokenized datasets split into shards.

        Args:
            B (int): Batch size.
            T (int): Sequence length.
            process_rank (int): Rank of the current process in distributed training.
            num_processes (int): Total number of processes.
            split (str): Dataset split ('train' or 'val').
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}, "Split must be 'train' or 'val'."

        # Load the shards (files)
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards

        assert len(shards) > 0, f"No shards found for split '{split}'."

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Fetch the next batch of data for training or validation.

        Returns:
            Tuple[Tensor, Tensor]: Input tokens (x) and target tokens (y).
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]

        x = (buf[:-1]).view(B, T)  # Inputs
        y = (buf[1:]).view(B, T)   # Targets

        # Advance position in tensor
        self.current_position += B * T * self.num_processes

        # If out of bounds, load the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y
