# AutoGrad

A tiny autodiff and MLP demo built from scratch. This repository contains a minimal reverse-mode automatic differentiation engine (`Value`) and a simple Multi-Layer Perceptron (`MLP`) implementation that uses `Value` objects for operations so you can compute gradients and train models by hand.

## Files
- `backprop_engine.py` — Implements the `Value` class with operator overloads and a `backward()` method for reverse-mode autodiff.
- `mlp.py` — Lightweight neural-network layers (Neuron, Layer, MLP) built on top of `Value`.

## Features ✅
- Scalar autograd with common operations (+, -, *, /, pow) and ReLU activation
- Manual parameter updates (gradient descent) using computed gradients

## Requirements ⚙️
Requirements: Python 3.8+ 

Example usage (in Python or a notebook):

```python
from backprop_engine import Value
from mlp import MLP

# toy dataset: learn product x0*x1 (non-linear)
xs = [[Value(x0), Value(x1)] for (x0, x1) in [(2.0, 3.0), (1.0, -1.0), (0.5, 0.2), (3.0, -2.0)]]
ys = [Value(x[0].data * x[1].data) for x in xs]

model = MLP(2, [8, 1])

# forward pass for first sample
y_pred = model(xs[0])  # should be a Value
loss = (y_pred - ys[0]) ** 2
loss.backward()

# update parameters manually
lr = 0.01
for p in model.parameters():
    p.data -= lr * p.grad
    p.grad = 0.0
```

## Demo Notebook [To Be Added]
There's a demo notebook `AutoGrad_demo.ipynb` that shows a full training loop on a toy dataset and demonstrates forward/backward passes.

---