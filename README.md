# Neural Network From Scratch

A fully connected neural network built completely from scratch using only NumPy, trained on image classification data.  
This project was mainly focused on understanding *how neural networks actually learn internally* rather than relying on high-level frameworks like TensorFlow or PyTorch.

![App Screenshot](https://github.com/rishit836/cnn-from-scratch/blob/basic-neural-network/prediction.png?raw=true)

---

## What This Project Does

- Implements a neural network from scratch using NumPy
- Performs forward propagation manually
- Implements backpropagation and gradient descent without deep learning libraries
- Trains on image classification data
- Predicts random samples after training
- Displays images along with predictions for testing and evaluation

---

## Main Learning Outcomes

This project was less about just “making a model work” and more about deeply understanding the internal mechanics of neural networks.

### Forward Pass

The forward pass was comparatively easier to understand.  
The only initially confusing part was handling matrix dimensions and understanding how data flows through layers using matrix multiplication, but after experimenting with the structures, it became intuitive.

Key concepts learned:
- Matrix multiplication in neural networks
- Weight and bias transformations
- Activation functions
- Layer-wise data flow

---

### Understanding Backpropagation

Backpropagation was the most challenging and important part of the project.

To properly understand it, I spent time going back and forth between:
- theory
- implementation videos
- mathematical derivations
- debugging outputs

This process helped build intuition for:
- gradient flow
- chain rule application
- derivative calculations
- weight updates
- how errors propagate backwards through layers

Resources that helped:

- [Backpropagation Video 1](https://youtu.be/i94OvYb6noo)
- [Backpropagation Video 2](https://youtu.be/VMj-3S1tku0)
- [MIT Backpropagation Notes](https://ocw.mit.edu/courses/9-641j-introduction-to-neural-networks-spring-2005/5d46d3b6e32a5120fe9893193c31e926_lec20_backprop.pdf)
- [Jacobian Matrix Explanation](https://machinelearningmastery.com/a-gentle-introduction-to-the-jacobian/)

A small amount of AI/LLM assistance was also used occasionally for resolving conceptual doubts and debugging issues.

---

## Challenges Faced

During training, several real implementation problems appeared, including:

- Accuracy getting stuck at very low values
- Incorrect gradient calculations
- Activation and loss function mismatches
- Shape and matrix alignment issues
- Debugging exploding/incorrect updates

Fixing these issues helped build a much stronger understanding of:
- why neural networks fail to learn
- how gradients affect optimization
- how activations interact with loss functions
- how small implementation mistakes completely affect training

---

## Additional Features

After training the model:
- Random image prediction testing was implemented
- Image visualization was added inside the notebook
- Predictions could be visually verified against actual samples

This made it easier to evaluate whether the network was genuinely learning or memorizing patterns incorrectly.

---

## Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Project Structure

```bash
basic-neural-network-from-scratch/
│
├── basic-neural-network-from-scratch.ipynb
├── prediction.png
└── README.md
```

---

## How To Run

Clone the repository:

```bash
git clone https://github.com/rishit836/cnn-from-scratch.git
cd cnn-from-scratch
```

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the notebook:

```bash
jupyter notebook
```

---

## Biggest Takeaway

The biggest takeaway from this project was realizing that neural networks become much easier to understand once every operation is implemented manually.

Building everything from scratch helped develop intuition for:
- tensor operations
- gradient flow
- optimization
- debugging deep learning systems
- the actual mathematics behind learning

This project transformed neural networks from something “abstract” into something understandable at an implementation level.
