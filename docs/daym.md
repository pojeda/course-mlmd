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

### Chain Rule

The chain rule is a fundamental concept in calculus and optimization. It allows derivatives 
of composite functions to be computed efficiently and forms the mathematical foundation of 
backpropagation in deep learning.

If a function depends on another function:

$$
y = f(g(x))
$$

then the derivative of $y$ with respect to $x$ is:

$$
\frac{dy}{dx}
=
\frac{dy}{dg}
\frac{dg}{dx}
$$

The chain rule decomposes complex derivatives into products of simpler derivatives.

#### Example

Consider the function:

$$
y = (3x + 1)^2
$$

Define:

$$
u = 3x + 1
$$

Then:

$$
y = u^2
$$

Using the chain rule:

$$
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
$$

Compute each derivative:

$$
\frac{dy}{du} = 2u
$$

$$
\frac{du}{dx} = 3
$$

Substituting:

$$
\frac{dy}{dx}
=
2u \cdot 3
$$

Since $u = 3x + 1$:

$$
\frac{dy}{dx}
=
6(3x + 1)
$$

#### Chain Rule in Multiple Dimensions

In machine learning, functions often depend on many variables. For a multivariable function:

$$
z = f(x, y)
$$

where:

$$
x = g(t)
\quad \text{and} \quad
y = h(t)
$$

the chain rule becomes:

$$
\frac{dz}{dt}
=
\frac{\partial z}{\partial x}
\frac{dx}{dt}
+
\frac{\partial z}{\partial y}
\frac{dy}{dt}
$$

This generalized form is heavily used in neural networks.

#### Importance in Machine Learning

The chain rule is essential for:

- Backpropagation
- Gradient descent
- Automatic differentiation
- Deep neural networks
- Computational graphs

Modern deep learning frameworks compute gradients by repeatedly applying the chain 
rule through layers of a neural network.

#### Example: Chain Rule with PyTorch

```python
import torch

# VARIABLE WITH GRADIENT TRACKING
x = torch.tensor(2.0, requires_grad=True)

# COMPOSITE FUNCTION
y = (3 * x + 1) ** 2

# COMPUTE DERIVATIVE
y.backward()

# RESULTS
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())
```

#### Mathematical Verification

The function is:

$$
y = (3x + 1)^2
$$

Using the chain rule:

$$
\frac{dy}{dx}
=
2(3x + 1)(3)
$$

For $x = 2$:

$$
\frac{dy}{dx}
=
2(7)(3)
=
42
$$

The result computed using automatic differentiation matches the analytical derivative.



## 3. Probability and Statistics

Probability and statistics are fundamental for understanding uncertainty, noise, 
model evaluation, and probabilistic learning.

### Gaussian Distribution

The Gaussian distribution, also called the normal distribution, is one of the most important probability 
distributions in statistics and machine learning. It describes continuous variables that tend to cluster 
around a mean value.

The probability density function of a Gaussian distribution is:

$$
p(x)
=
\frac{1}{
\sqrt{2\pi\sigma^2}
}
\exp
\left(
-\frac{
(x - \mu)^2
}{
2\sigma^2
}
\right)
$$

where:

- $\mu$ is the mean of the distribution
- $\sigma^2$ is the variance
- $\sigma$ is the standard deviation

#### Properties of the Gaussian Distribution

The Gaussian distribution has several important properties:

- Symmetric around the mean
- Bell-shaped curve
- Mean, median, and mode are equal
- Controlled by only two parameters: mean and variance

#### Standard Normal Distribution

A Gaussian distribution with:

$$
\mu = 0
$$

and

$$
\sigma^2 = 1
$$

is called the standard normal distribution.

It is commonly written as:

$$
\mathcal{N}(0, 1)
$$

#### Example: Gaussian Distribution

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


### Bayesian Inference

Bayesian inference is a probabilistic framework for updating beliefs using observed data. It is based on Bayes' theorem, which relates prior knowledge to new evidence.

Bayes' theorem is given by:

$$
P(\theta \mid D)
=
\frac{
P(D \mid \theta) P(\theta)
}{
P(D)
}
$$

where:

- $P(\theta \mid D)$ is the posterior distribution
- $P(D \mid \theta)$ is the likelihood
- $P(\theta)$ is the prior distribution
- $P(D)$ is the evidence or marginal likelihood

#### Interpretation

Bayesian inference updates our belief about parameters $\theta$ after observing data $D$.

The process can be summarized as:

$$
\text{Posterior}
=
\frac{
\text{Likelihood} \times \text{Prior}
}{
\text{Evidence}
}
$$

#### Advantages of Bayesian Methods

Bayesian approaches provide:

- Uncertainty quantification
- Robustness with limited data
- Probabilistic predictions
- Incorporation of prior knowledge

These properties are especially important in scientific machine learning and decision-making under uncertainty.

#### Example: Coin Toss Inference

Suppose we want to estimate the probability of obtaining heads in a coin toss.

If:
- Prior belief: $P(\theta)$
- Observed data: number of heads and tails

then Bayesian inference computes the posterior probability distribution over $\theta$.

As more observations are collected, the posterior becomes more concentrated around the true probability.

#### Applications in Machine Learning

Bayesian methods are widely used in:

- Bayesian neural networks
- Probabilistic graphical models
- Scientific modeling
- Reinforcement learning
- Uncertainty estimation
- Hyperparameter optimization


### Gaussian Processes

Gaussian Processes (GPs) are nonparametric probabilistic models used for regression and uncertainty estimation.

A Gaussian Process defines a probability distribution over functions:

$$
f(x)
\sim
\mathcal{GP}(m(x), k(x, x'))
$$

where:

- $m(x)$ is the mean function
- $k(x, x')$ is the covariance kernel function

A Gaussian Process assumes that any finite collection of function values follows a multivariate Gaussian distribution.

#### Mean Function

The mean function is:

$$
m(x) = \mathbb{E}[f(x)]
$$

In practice, the mean is often assumed to be zero:

$$
m(x) = 0
$$

#### Covariance Kernel

The kernel defines similarity between input points.

One common kernel is the Radial Basis Function (RBF) kernel:

$$
k(x, x')
=
\sigma^2
\exp
\left(
-\frac{
||x - x'||^2
}{
2l^2
}
\right)
$$

where:

- $\sigma^2$ controls the variance
- $l$ is the length scale

#### Intuition

Gaussian Processes model smooth functions by assuming that nearby points have correlated outputs.

Predictions include both:
- A mean estimate
- A predictive uncertainty

This makes Gaussian Processes particularly useful when uncertainty quantification is important.

#### Gaussian Process Regression

Given training data:

$$
X = \{x_1, x_2, \dots, x_n\}
$$

and target values:

$$
\mathbf{y} = [y_1, y_2, \dots, y_n]^T
$$

Gaussian Process regression predicts function values at new points while estimating uncertainty.

#### Advantages

Gaussian Processes provide:

- Uncertainty estimates
- Flexible nonlinear modeling
- Strong performance with small datasets
- Interpretable probabilistic predictions

#### Limitations

Gaussian Processes scale poorly with dataset size because they require inversion of the covariance matrix:

$$
\mathcal{O}(n^3)
$$

where $n$ is the number of training samples.

#### Applications in Machine Learning

Gaussian Processes are commonly used in:

- Bayesian optimization
- Scientific machine learning
- Time series modeling
- Robotics
- Active learning
- Surrogate modeling
- Materials science and molecular modeling

#### Example: Gaussian Process Regression in Python

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# TRAINING DATA
X = np.array([[1.0], [3.0], [5.0], [6.0], [8.0]])
y = np.sin(X).ravel()

# DEFINE KERNEL
kernel = 1.0 * RBF(length_scale=1.0)

# CREATE GAUSSIAN PROCESS MODEL
gp = GaussianProcessRegressor(kernel=kernel)

# TRAIN MODEL
gp.fit(X, y)

# TEST POINTS
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

# PREDICTIONS
y_pred, sigma = gp.predict(X_test, return_std=True)

# PLOT RESULTS
plt.figure(figsize=(8, 5))

plt.plot(X, y, 'o', label='Training Data')
plt.plot(X_test, y_pred, label='GP Mean Prediction')

plt.fill_between(
    X_test.ravel(),
    y_pred - 2 * sigma,
    y_pred + 2 * sigma,
    alpha=0.3,
    label='Uncertainty'
)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()

plt.show()
```

This example demonstrates how Gaussian Processes provide both predictions and 
uncertainty estimates, which is one of their main advantages over standard regression methods.



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
