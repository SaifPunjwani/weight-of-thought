# ReasoningResearch

A research project exploring reasoning capabilities in deep learning and natural language processing systems.

## Overview

This project implements several neural architectures to study logical reasoning abilities:

- A hybrid neuro-symbolic reasoner combining GPT-2 with symbolic rule attention
- A deep Q-network based reasoner for logical deduction
- A language reasoning model using LSTM and attention mechanisms
- Dynamic visualization tools for model analysis

## Key Components

- `code/main.py`: Core implementation of the hybrid neuro-symbolic model
- `code/r2.py`: Deep Q-Network based reasoning system
- `language/language.py`: Language reasoning model and visualization tools

## Features

- Prioritized experience replay for efficient learning
- Attention mechanisms for interpretable reasoning
- Real-time visualization of model performance
- Support for both neural and symbolic reasoning approaches

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Matplotlib
- NumPy
- Pygame
- Seaborn

## Usage

The models can be trained on logical reasoning tasks like:
- Syllogistic reasoning
- Conditional logic
- Validity assessment of arguments

See individual model files for training and inference examples.
