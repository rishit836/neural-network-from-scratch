# Neural Network From Scratch (micrograd + MLP example)

This repository contains a small-from-scratch neural network toolkit using a tiny autograd implementation (micrograd-style) and a simple multilayer perceptron (MLP) training example. It is intended for learning and experimentation rather than production use.

---

## What’s in this repo

- `multilayer-perceptron.py` — training script that uses the `micrograd` package to train a small MLP on image data from `dataset/train.csv`.
- `basic-neural-network-from-scratch.ipynb` — notebook with exploratory code and visualizations.
- `micrograd/` — minimal autograd implementation:
	- `micrograd/grad.py` — `Value` class with forward ops and backward pass.
	- `micrograd/nn.py` — `Neuron`, `Layer`, and `MLP` convenience classes built on top of `Value`.
- `dataset/` — contains `train.csv` and `test.csv` expected by the training script.

---

## Requirements

- Python 3.8+ recommended
- pip packages: `numpy`, `pandas`, `matplotlib` (for notebook visualization)

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## How to run the training script

1. Ensure the dataset CSVs are in the `dataset/` folder (the script expects `dataset/train.csv`).
2. Run:

```bash
python multilayer-perceptron.py
```

The script trains for a small number of epochs (default: 5) and prints per-sample progress and the aggregated loss per epoch. The network architecture and hyperparameters are defined inside the script:

- Network: `MLP(nin, [64, 64, 10])`
- Random seed: `150` (for reproducible shuffling)
- Learning rate: `0.01` (applied as `param.data += -0.01 * param.grad`)

---

## Notes and caveats (current implementation)

- The repo purpose is educational: the `micrograd` implementation is intentionally small and explicit.
- The training script currently computes a dataset-level `loss` by summing per-sample scalar `Value` objects; see the script notes below for correctness improvements.
- The script uses `train_label` (integer labels) in the loss expression — it should use one-hot labels (see `y_train`) and a proper cross-entropy loss with a softmax for correct classification training.
- Numerical stability: softmax + log-loss should be implemented carefully (use log-sum-exp trick) when converting scores to probabilities.

Suggested fixes to improve training quality:

- Use the one-hot encoded `y_train` when computing loss for classification.
- Implement `softmax` followed by cross-entropy (or combine into a stable log-softmax + NLL) instead of manually using raw `exp` sums.
- Add a configurable learning rate and a simple training/validation loop with accuracy reporting.

---

## Quick file references

- `multilayer-perceptron.py` — training runner (see top of file for configurable params).
- `micrograd/grad.py` — core `Value` class and `backward()` implementation.
- `micrograd/nn.py` — `Neuron`, `Layer`, and `MLP` classes used by the training script.
- `basic-neural-network-from-scratch.ipynb` — notebook with visualization and exploratory code.

---
Future Plans:
- Currently i use basic mathematical operations in python which is slow for computing but to keep it simple for now so i can actually understand the basics i did that, my aim is to convert that into tensor operations for which `Value` class i have defined will have tensors and everything will be done using numpy 

- I have only implemented MLP yet, once i have optimised the `micrograd` library i will implement based on the research paper in resources folder.
