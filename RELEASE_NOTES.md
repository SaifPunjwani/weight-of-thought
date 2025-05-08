# Weight-of-Thought (WoT) Release Notes

## Version 0.1.0 - Initial Release

We're excited to release the first version of the Weight-of-Thought (WoT) neural reasoning framework! This release provides the full implementation of the WoT model as described in our [arXiv paper](https://arxiv.org/abs/2504.10646), along with tools for training, evaluation, and visualization.

### Overview

Weight-of-Thought (WoT) is a novel neural reasoning approach that extends beyond traditional Chain-of-Thought (CoT) by representing reasoning as an interconnected web rather than a linear sequence. This architecture allows for more complex reasoning patterns, with information flowing through specialized nodes that exchange messages through multiple rounds.

### Features

- **Core WoT Model**: Complete implementation of the Weight-of-Thought architecture
  - Graph-based reasoning with message passing between nodes
  - Multi-step reasoning process
  - Attention mechanisms at both node and reasoning step levels
  - Task-specific output heads for different reasoning tasks

- **Language Encoders**: Multiple options for encoding text inputs
  - GPT-2 encoder (default)
  - BERT encoder
  - RoBERTa encoder

- **Reasoning Tasks**: Support for diverse reasoning tasks
  - Syllogism (logical deduction)
  - Math Sequence (pattern recognition)
  - Algebra (word problems)
  - Combinatorics (counting problems)
  - Geometry (geometric reasoning)

- **Training & Evaluation**: Complete training and evaluation pipeline
  - Command-line interface for training and evaluation
  - Performance metrics and visualization
  - Data augmentation capabilities

- **Visualization**: Comprehensive visualization tools
  - Node attention visualization
  - Edge attention matrix
  - Reasoning step visualization
  - Weight matrix analysis

- **Examples & Tutorials**: Ready-to-use examples
  - Quick start example
  - Custom task creation
  - Interactive notebooks for exploring the model

### Getting Started

1. Install the package and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the quick start example:
   ```bash
   python examples/quick_start.py
   ```

3. Start exploring the interactive notebooks:
   ```bash
   jupyter notebook notebooks/01_introduction.ipynb
   ```

### Future Work

We are actively working on the following enhancements for future releases:

1. More pre-trained models for specific reasoning domains
2. Support for multi-modal reasoning (text + images)
3. Improved visualization tools
4. Performance optimizations
5. Integration with popular deep learning frameworks

### Contribute

We welcome contributions to the Weight-of-Thought project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

### Citation

If you use Weight-of-Thought in your research, please cite our paper:

```bibtex
@article{punjwani2024weight,
  title={Weight-of-Thought Reasoning: Exploring Neural Network Weights for Enhanced LLM Reasoning},
  author={Punjwani, Saif and Heck, Larry},
  journal={arXiv preprint arXiv:2504.10646},
  year={2024}
}
```

### License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.