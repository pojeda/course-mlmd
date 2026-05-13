# Mathematics for Machine Learning

## 1. Linear Algebra

Linear algebra forms the mathematical foundation of machine learning. Data, model parameters, neural 
network activations, and transformations are typically represented using vectors and matrices.

### Topics

- Scalars, vectors, and matrices
- Matrix multiplication
- Dot products
- Norms and distances
- Eigenvalues and eigenvectors
- Tensor operations

### Example: Vector Operations in PyTorch

```python
import torch
import numpy as np

# VECTOR OPERATIONS WITH PYTORCH AND NUMPY

# Create vectors with PyTorch
x_torch = torch.tensor([1.0, 2.0, 3.0])
y_torch = torch.tensor([4.0, 5.0, 6.0])

# Create vectors with NumPy
x_numpy = np.array([1.0, 2.0, 3.0])
y_numpy = np.array([4.0, 5.0, 6.0])

# DOT PRODUCT
# Mathematical expression:
# x · y = Σ_i x_i y_i

# PyTorch version
dot_product_torch = torch.dot(x_torch, y_torch)

# NumPy version
dot_product_numpy = np.dot(x_numpy, y_numpy)

print("PyTorch dot product:", dot_product_torch)
print("NumPy dot product:", dot_product_numpy)

# VECTOR NORM
# Euclidean norm:
# ||x|| = sqrt(Σ_i x_i²)

# PyTorch version
norm_torch = torch.norm(x_torch)

# NumPy version
norm_numpy = np.linalg.norm(x_numpy)

print("\nPyTorch vector norm:", norm_torch)
print("NumPy vector norm:", norm_numpy)
```

One can use NumPy for scientific computing and preprocessing; use PyTorch tensors 
for GPU acceleration, automatic differentiation, and deep learning models.

### Matrix Multiplication Example

```python
import torch
import numpy as np

# MATRIX MULTIPLICATION WITH PYTORCH AND NUMPY

# Create matrices with PyTorch
A_torch = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

B_torch = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

# Create matrices with NumPy
A_numpy = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

B_numpy = np.array([
    [5.0, 6.0],
    [7.0, 8.0]
])

# MATRIX MULTIPLICATION
# Mathematical expression:
# C = AB
# C_ij = Σ_k A_ik B_kj

# PyTorch version
C_torch = torch.matmul(A_torch, B_torch)

# NumPy version
C_numpy = np.matmul(A_numpy, B_numpy)

# Alternative NumPy syntax
C_numpy_alt = A_numpy @ B_numpy

print("PyTorch matrix multiplication:")
print(C_torch)

print("\nNumPy matrix multiplication:")
print(C_numpy)

print("\nNumPy matrix multiplication using @ operator:")
print(C_numpy_alt)
```


## 2. Optimization

Optimization is the process of finding model parameters that minimize a loss function. Most 
machine learning algorithms rely on optimization techniques such as gradient descent.

### Topics

- Functions and derivatives
- Gradients
- Chain rule
- Gradient descent
- Learning rates
- Loss minimization

### Gradient Descent Concept

The goal is to iteratively update parameters:

$$
\theta_{new} = \theta_{old} - \eta \nabla L(\theta)
$$

Where:

- $\theta$ are the model parameters
- $\eta$ is the learning rate
- $L(\theta)$ is the loss function

### Example: Gradient Descent in PyTorch

```python
import torch

# Parameter to optimize
x = torch.tensor([5.0], requires_grad=True)

learning_rate = 0.1

for step in range(20):

    # Example loss function
    loss = (x - 2) ** 2

    # Compute gradients
    loss.backward()

    # Update parameter
    with torch.no_grad():
        x -= learning_rate * x.grad

    # Reset gradients
    x.grad.zero_()

    print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
```

## 3. Probability and Statistics

Probability and statistics are fundamental for understanding uncertainty, noise, 
model evaluation, and probabilistic learning.

### Topics

- Random variables
- Probability distributions
- Gaussian distribution
- Mean and variance
- Covariance and correlation
- Sampling

### Example: Gaussian Distribution

```python
import torch
import matplotlib.pyplot as plt

# Generate samples from a normal distribution
samples = torch.normal(mean=0.0, std=1.0, size=(1000,))

plt.hist(samples.numpy(), bins=30)

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Gaussian Distribution")
plt.savefig("gaussian.png", dpi=300, bbox_inches="tight")
plt.show()
```

### Mean and Standard Deviation

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print("Mean:", torch.mean(x))
print("Standard deviation:", torch.std(x))
```


## 4. Information Theory

Information theory provides mathematical tools for quantifying uncertainty 
and information content in data.

### Topics

- Entropy
- Cross-entropy
- Kullback–Leibler divergence
- Information content

### Entropy

Entropy measures uncertainty:

$$
H(X) = -\sum_i p(x_i) \log p(x_i)
$$

### Example: Entropy Calculation

```python
import torch

# Probability distribution
p = torch.tensor([0.1, 0.3, 0.6])

entropy = -torch.sum(p * torch.log2(p))

print("Entropy:", entropy.item())
```

### Cross-Entropy Loss Example

```python
import torch
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()

# Predicted logits
predictions = torch.tensor([
    [2.0, 0.5, 0.3]
])

# Correct class
labels = torch.tensor([0])

loss = loss_function(predictions, labels)

print("Cross-entropy loss:", loss.item())
```



## 5. Basic Graph Theory

Graphs are widely used in machine learning for representing relationships between 
objects. In chemistry, molecules can naturally be represented as graphs.

### Topics

- Nodes and edges
- Adjacency matrices
- Degree of nodes
- Graph connectivity
- Molecular graphs

### Example: Adjacency Matrix

Suppose we have a graph with 3 nodes:

```text
Node 0 --- Node 1
   \         /
     Node 2
```

All nodes are connected to each other.

```python
import torch

# Simple graph adjacency matrix
adjacency = torch.tensor([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=torch.float32)

print(adjacency)
```

* `1` → nodes are connected
* `0` → no connection

For example:

```text
adjacency[0,1] = 1
```

means:

* Node 0 is connected to Node 1.

### Graph Node Features

Each row corresponds to one node:

```python
import torch

# Example node features
node_features = torch.tensor([
    [1.0, 0.0], #node 0
    [0.0, 1.0], #onde 1
    [1.0, 1.0]  #node 2
])

print(node_features)
```

These features could represent:

* atom types,
* charges,
* molecular descriptors,
* or learned embeddings.


### Aggregate neighbor information

```python
# Matrix multiplication:
# adjacency × node_features

aggregated_features = torch.matmul(
    adjacency,
    node_features
)

print("\nAggregated neighbor features:")
print(aggregated_features)
```

The aggregation collects information from neighboring nodes.

For example:

* Node 0 receives information from Nodes 1 and 2,
* Node 1 receives information from Nodes 0 and 2,
* etc.

## 6. Dimensionality Reduction

Many scientific datasets contain high-dimensional data that can be difficult to 
visualize or analyze directly. Dimensionality reduction techniques project the data into 
lower-dimensional spaces while preserving important structure.

### Topics

- High-dimensional spaces
- Principal Component Analysis (PCA)
- Latent spaces
- Data visualization
- Feature compression

### Example: PCA with PyTorch and scikit-learn

```python
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Random high-dimensional dataset
X = torch.randn(100, 10)

# PCA projection
pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X.numpy())

# Visualization
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")
plt.savefig("pca.png", dpi=300, bbox_inches="tight")
plt.show()
```


## 7. Integrated Neural Network Example

This example combines several mathematical concepts introduced during the course.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic dataset
X = torch.randn(200, 5)

y = (X.sum(dim=1) > 0).long()

# Simple neural network
model = nn.Sequential(
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):

    predictions = model(X)
    loss = loss_function(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

## Summary

This course introduced the core mathematical concepts underlying machine learning and deep learning:

- Linear algebra for data representation
- Optimization for model training
- Probability and statistics for uncertainty and evaluation
- Information theory for measuring information and learning
- Graph theory for relational data
- Dimensionality reduction for visualization and feature compression
