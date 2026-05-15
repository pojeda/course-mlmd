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

You can add the following subsection after the matrix multiplication section and before moving to the next major topic.

### Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is one of the most important matrix factorization techniques in linear algebra and machine learning. It decomposes a matrix into orthogonal components and reveals its intrinsic geometric structure.

Given a matrix $A \in \mathbb{R}^{m \times n}$, the Singular Value Decomposition is:

$$
A = U \Sigma V^T
$$

where:

- $U$ is an orthogonal matrix containing the left singular vectors
- $\Sigma$ is a diagonal matrix containing the singular values
- $V^T$ contains the right singular vectors

The singular values are always non-negative and are usually ordered from largest to smallest:

$$
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0
$$

where $r$ is the rank of the matrix.

#### Geometric Interpretation

SVD can be interpreted as a sequence of geometric transformations:

1. Rotation of the input space by $V^T$
2. Scaling along orthogonal directions by $\Sigma$
3. Rotation into the output space by $U$

This decomposition reveals the most important directions of variation in the data.

#### Applications in Machine Learning

SVD is widely used in machine learning and scientific computing:

- Dimensionality reduction
- Principal Component Analysis (PCA)
- Noise reduction and denoising
- Recommender systems
- Compression of large matrices
- Latent semantic analysis in natural language processing

#### Low-Rank Approximation

A matrix can be approximated using only the largest singular values:

$$
A_k = U_k \Sigma_k V_k^T
$$

where $k < r$.

This produces the best rank-$k$ approximation of the matrix in the least-squares sense.

#### Example: Singular Value Decomposition in Python

```python
import numpy as np
import torch

# MATRIX FOR SVD
A_numpy = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

A_torch = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

# NUMPY SVD
U_numpy, S_numpy, VT_numpy = np.linalg.svd(A_numpy)

print("NumPy SVD")
print("\nU matrix:")
print(U_numpy)

print("\nSingular values:")
print(S_numpy)

print("\nV^T matrix:")
print(VT_numpy)

# PYTORCH SVD
U_torch, S_torch, VT_torch = torch.linalg.svd(A_torch)

print("\n\nPyTorch SVD")
print("\nU matrix:")
print(U_torch)

print("\nSingular values:")
print(S_torch)

print("\nV^T matrix:")
print(VT_torch)

# RECONSTRUCT ORIGINAL MATRIX
Sigma_numpy = np.zeros((3, 2))
np.fill_diagonal(Sigma_numpy, S_numpy)

A_reconstructed = U_numpy @ Sigma_numpy @ VT_numpy

print("\nReconstructed matrix:")
print(A_reconstructed)
```

SVD is a fundamental tool in deep learning, scientific computing, and data analysis because it provides 
a compact and interpretable representation of matrices while preserving the most important information in the data.


## 2. Optimization

Optimization is the process of finding model parameters that minimize a loss function. Most 
machine learning algorithms rely on optimization techniques such as gradient descent.



### Taylor Series and Local Approximations

Taylor expansions provide the mathematical foundation for understanding:

* Gradient-based optimization
* Newton's method
* Second-order optimization
* Local approximations of loss functions
* Curvature and Hessians
* Stability analysis

In machine learning, optimization algorithms often rely on local approximations of functions, 
and Taylor series explain why these approximations work.

Taylor series approximate a function locally around a point using derivatives.
For a function $f(x)$ expanded around $x_0$:

$$
f(x)
=
f(x_0)
+
f'(x_0)(x - x_0)
+
\frac{1}{2}f''(x_0)(x - x_0)^2
+
\cdots
$$

The first-order approximation is:

$$
f(x)
\approx
f(x_0)
+
f'(x_0)(x - x_0)
$$

This approximation forms the basis of gradient descent methods.

The second-order approximation includes curvature information:

$$
f(x)
\approx
f(x_0)
+
f'(x_0)(x - x_0)
+
\frac{1}{2}f''(x_0)(x - x_0)^2
$$

In multiple dimensions, the Taylor expansion becomes:

$$
f(\mathbf{x})
\approx
f(\mathbf{x}_0)
+
\nabla f(\mathbf{x}_0)^T
(\mathbf{x} - \mathbf{x}_0)
+
\frac{1}{2}
(\mathbf{x} - \mathbf{x}_0)^T
H
(\mathbf{x} - \mathbf{x}_0)
$$

where:

- $\nabla f$ is the gradient
- $H$ is the Hessian matrix

These approximations are fundamental in optimization algorithms used in machine learning.


### Gradient Descent Concept

The goal is to iteratively update parameters:

$$
\theta_{new} = \theta_{old} - \eta \nabla L(\theta)
$$

Where:

- $\theta$ are the model parameters
- $\eta$ is the learning rate
- $L(\theta)$ is the loss function

#### Example: Gradient Descent in PyTorch

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

### Automatic Differentiation

Automatic differentiation is a computational technique used to evaluate derivatives 
efficiently and accurately. It is a core component of modern machine learning frameworks 
such as PyTorch, TensorFlow, and JAX.

Unlike symbolic differentiation, automatic differentiation does not manipulate mathematical 
expressions symbolically. Unlike numerical differentiation, it does not rely on finite 
difference approximations.

Instead, automatic differentiation applies the chain rule systematically through a 
sequence of elementary operations.

Given a composite function:

$$
f(x) = f_3(f_2(f_1(x)))
$$

the chain rule states:

$$
\frac{df}{dx}
=
\frac{df_3}{df_2}
\frac{df_2}{df_1}
\frac{df_1}{dx}
$$

Machine learning frameworks construct a computational graph that tracks operations and 
automatically computes gradients during backpropagation.

#### Forward and Reverse Mode Differentiation

There are two main approaches:

#### Forward Mode

Gradients are propagated from inputs to outputs.

Efficient when:
- The number of inputs is small
- The number of outputs is large

#### Reverse Mode

Gradients are propagated backward from outputs to inputs.

Efficient when:
- The number of parameters is very large
- The output is scalar

Reverse-mode automatic differentiation is the foundation of backpropagation in deep learning.

#### Computational Graphs

A computational graph represents mathematical operations as nodes connected by edges.

For example:

$$
y = x^2 + 3x
$$

can be decomposed into elementary operations:
- Multiplication
- Addition

The framework stores intermediate values and computes derivatives automatically.

#### Example: Automatic Differentiation with PyTorch

```python
import torch

# CREATE A TENSOR WITH GRADIENT TRACKING
x = torch.tensor(2.0, requires_grad=True)

# DEFINE A FUNCTION
y = x**2 + 3*x + 1

# COMPUTE DERIVATIVE dy/dx
y.backward()

# PRINT RESULTS
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())
```

#### Mathematical Verification

The function is:

$$
y = x^2 + 3x + 1
$$

Its analytical derivative is:

$$
\frac{dy}{dx} = 2x + 3
$$

For $x = 2$:

$$
\frac{dy}{dx} = 2(2) + 3 = 7
$$

The value computed using automatic differentiation matches the analytical result.

#### Importance in Machine Learning

Automatic differentiation enables efficient training of neural networks by computing 
gradients of loss functions with respect to millions of parameters.

Applications include:

- Backpropagation
- Gradient descent optimization
- Physics-informed neural networks
- Scientific machine learning
- Deep generative models

Without automatic differentiation, training modern deep learning models would be 
computationally impractical.



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
