# Transformers from Scratch for English-to-Arabic Translation

This repository contains an implementation of a Transformer model built from scratch using **PyTorch**. The model is trained for **English-to-Arabic translation** using the [news commentary dataset](https://huggingface.co/datasets/Helsinki-NLP/news_commentary).

## Key Features

- **Custom Transformer Architecture**: Implemented from scratch in PyTorch, inspired by Mr. Umar Jamil's code and his [YouTube video](https://www.youtube.com/watch?v=ISNdQcPhsts&t=7047s).
- **Monitoring and Logging**: Utilized PyTorch Ignite for efficient monitoring, logging, and checkpointing during training.
- **Dataset Handling**: Leveraged Hugging Face's `datasets` library for easy access to the news_commentary dataset and tokenization.

## Repository packages structure

- **`model`**: Contains the architecture of the Transformer model. This directory includes the implementation of the model components.
- **`config`**: Holds configuration files and hyperparameters for training. Adjust these files to modify the training settings and parameters.
- **`train`**: Contains scripts and modules related to training the model. This is where the training process is defined and executed.
- **`utils`**: Includes utility functions and classes, particularly those related to PyTorch Ignite handlers for monitoring, logging, and checkpointing.

## Getting Started

To get started with this repository, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Mo-Ouail-Ocf/transformers-from-scratch.git
   cd transformers-from-scratch
   ```

2. **Set Up Your Environment**:

   Create a Conda environment from the provided `env.yml` file:

   ```bash
   conda env create -f env.yml
   ```

   Activate the Conda environment:

   ```bash
   conda activate transformers-env
   ```

   Note: This code is compatible with Python 3.11.

## Future Tasks

- **Results and Visualization**: Implement visualization tools for attention scores to better understand model performance and behavior.

## Acknowledgements

- Special thanks to Mr. Umar Jamil for his exceptional resources and tutorials, which provided valuable insights into implementing and understanding Transformer models.

- For more information and in-depth tutorials, visit [Mr. Umar Jamil's YouTube channel](https://www.youtube.com/channel/UCzHlPRht58O_vFqX_2OliTQ).

---

