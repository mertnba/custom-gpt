## Repository Structure
The repository is organized into the following directories:

- **`model/`**: Defines the transformer model and its components.
  - `config.py`: Contains the GPTConfig dataclass for configuring model parameters such as the number of layers, heads, and embedding dimensions.
  - `layers.py`: Implements reusable building blocks for the model, including CausalSelfAttention, MLP, and Block.
  - `gpt.py`: Combines the layers and configuration into the full GPT model, including token embeddings, positional encodings, and forward logic.
  
- **`data/`**: Contains data loading utilities.
  - `dataloader.py`: Contains the DataLoaderLite class for managing data shards, batching, and loading sequences for training and validation.
  - `utils.py`: Contains helper functions such as load_tokens for loading preprocessed tokenized data into tensors.

- **`train/`**: Includes scripts for training the model.
  - `train.py`: Contains the main training loop, including gradient updates, loss computation, validation, and logging.
  - `scheduler.py`: Implements the learning rate scheduler logic.
  - `logger.py`: Handles logging of training and validation metrics.

- **`utils/`**: Utility functions for common tasks.
  - `helpers.py`: Contains reusable helper functions, such as tensor manipulation, data validation, or token processing.
  - `distributed.py`: Handles distributed training setup and teardown.

- **`evaluation/`**: Scripts for evaluating the model on tasks like HellaSwag.
  - `evaluation.py`: Contains the core evaluation logic, including loss computation and HellaSwag evaluation.
  - `utils.py`: Utility functions to assist in evaluation, such as calculating metrics or formatting results.

## Features
- Full implementation of a GPT model with modular design.
- Support for distributed training using PyTorch DistributedDataParallel (DDP).
- Training loop with gradient accumulation, learning rate scheduling, and gradient clipping.
- Evaluation on HellaSwag with accuracy reporting.
- Flash attention support for improved memory efficiency during training.
