## Getting Started

We recommend to start with the [main demo]() to learn how the library works and how to use its features.

Motivation behind this library is to act as a one stop solution for research on Training Dynamics of Neural Networks, which is one of the most important topics in Mechanistic Interpretability and AI Interpretability in general.

This library is heavily insipired by the early work done by [Michael Hu](https://michahu.github.io/) for the paper [Latent State Models of Training Dynamics](https://arxiv.org/abs/2308.09543).

### Installation

```
pip install visualizing-training
```

Import the library using

```
import visualize_training
```

### Key Features

Key features include:

- Flexibility to define your own Neural Network architecture (currently it supports Transformer, MLPs, Resnet, LeNet5)
- Add custom hookpoints to collect data at different stages of the architecture
- Calculation of select metrics on the data collected
- Latent State predictions using [Hidden Markov Models (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze the Training Dynamics
- Interactive Visualization of Training Dynamics & Generalization Strategies in the form of network graphs and line charts.
