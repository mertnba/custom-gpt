## Repository Structure
The repository is organized into the following directories:

- **`model/`**: Defines the transformer model and its components.
  - `gpt.py`: Implements the GPT model and supporting layers like `Block`, `CausalSelfAttention`, and `MLP`.
  - `config.py`: Defines the `GPTConfig` dataclass for configuring model parameters.
  
- **`data/`**: Contains data loading utilities.
  - `dataloader.py`: Implements `DataLoaderLite` for loading and batching tokenized sequences.

- **`train/`**: Includes scripts for training the model.
  - `train.py`: Contains the main training loop, gradient updates, validation, and logging.

- **`utils/`**: Utility functions for common tasks.
  - `helpers.py`: Contains functions like `load_tokens`, `get_most_likely_row`, and other reusable utilities.

- **`distributed/`**: Handles distributed training setup.
  - `distributed_setup.py`: Initializes and manages distributed training.

- **`evaluation/`**: Scripts for evaluating the model on tasks like HellaSwag.
  - `evaluation.py`: Contains HellaSwag evaluation logic and result computation.

## Features
- Full implementation of a GPT model with modular design.
- Support for distributed training using PyTorch DistributedDataParallel (DDP).
- Training loop with gradient accumulation, learning rate scheduling, and gradient clipping.
- Evaluation on HellaSwag with accuracy reporting.
- Flash attention support for improved memory efficiency during training.
