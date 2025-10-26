# Day 2: Deep Learning for Molecular Systems

**Machine Learning for Molecular Systems - Advanced Course**

*Course Module 2: Neural Networks and Deep Learning Applications*

---

## Table of Contents

1. [Deep Learning Fundamentals](#1-deep-learning-fundamentals)
2. [Feedforward Neural Networks](#2-feedforward-neural-networks)
3. [Multi-Task Learning](#3-multi-task-learning)
4. [Convolutional Neural Networks](#4-convolutional-neural-networks)
5. [Recurrent Neural Networks](#5-recurrent-neural-networks)
6. [Transfer Learning](#6-transfer-learning)
7. [Complete Practical Exercise](#7-complete-practical-exercise)
8. [Model Interpretation](#8-model-interpretation)
9. [Best Practices](#9-best-practices)
10. [Key Takeaways](#10-key-takeaways)
11. [Resources](#11-resources)
12. [Homework Assignment](#12-homework-assignment)
13. [Appendix](#13-appendix)

---

## 1. Deep Learning Fundamentals

### 1.1 Why Deep Learning for Molecules?

Deep learning has revolutionized molecular property prediction by automatically learning complex representations from raw molecular data. Unlike traditional machine learning approaches that rely on hand-crafted descriptors, deep learning models can:

**Key Advantages:**

- **Automatic Feature Learning**: Neural networks learn hierarchical representations directly from molecular structures (SMILES, graphs, 3D coordinates) without manual feature engineering
- **Complex Pattern Recognition**: Capture non-linear relationships and subtle structural patterns that affect molecular properties
- **Transfer Learning**: Pre-trained models on large chemical databases can be fine-tuned for specific tasks with limited data
- **End-to-End Learning**: Direct mapping from molecular representation to property prediction in a single unified model
- **Multi-Task Learning**: Simultaneously predict multiple properties, sharing learned representations across tasks

**Detailed Examples:**

**Example 1: Solubility Prediction**
Traditional ML approach requires calculating molecular descriptors (LogP, molecular weight, H-bond donors/acceptors). Deep learning models can learn these patterns directly from SMILES strings:

```
Input: "CC(=O)OC1=CC=CC=C1C(=O)O" (Aspirin)
Model learns: aromatic rings → non-polar regions
             carboxyl groups → polar regions
             overall balance → solubility estimate
Output: LogS = -1.5 (moderate solubility)
```

**Example 2: Toxicity Prediction**
Deep learning excels at identifying toxic substructures (toxicophores) without explicit rules:

```
Model automatically learns:
- Aromatic amines → potential carcinogens
- Epoxides → DNA reactivity
- Nitro groups → mutagenicity risk
- Combination patterns → synergistic effects
```

**Example 3: Drug-Likeness**
Rather than using Lipinski's Rule of Five, neural networks learn implicit drug-likeness:

```
Training on FDA-approved drugs, the model learns:
- Optimal molecular weight ranges
- Hydrogen bonding patterns
- Lipophilicity balance
- Metabolic stability indicators
- Blood-brain barrier permeability
```

### 1.2 Neural Network Basics

A neural network consists of interconnected layers of artificial neurons that transform input data through learned weights and biases.

**Architecture Components:**

**Input Layer**: Receives molecular features (fingerprints, descriptors, or embeddings)
- Size determined by feature dimension
- No activation function applied

**Hidden Layers**: Transform inputs through non-linear operations
- Each neuron computes: **z = Σ(w_i × x_i) + b**
- Applies activation function: **a = f(z)**
- Multiple layers enable hierarchical feature learning

**Output Layer**: Produces final predictions
- Regression: Single neuron with linear activation
- Binary classification: Single neuron with sigmoid activation
- Multi-class: Multiple neurons with softmax activation

**Mathematical Foundation:**

For a single neuron:
```
Input: x = [x₁, x₂, ..., xₙ]
Weights: w = [w₁, w₂, ..., wₙ]
Bias: b

Linear transformation: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = w^T x + b
Activation: a = f(z)
```

For a layer:
```
Z^[l] = W^[l] × A^[l-1] + b^[l]
A^[l] = f(Z^[l])

Where:
- l = layer index
- W^[l] = weight matrix for layer l
- A^[l-1] = activations from previous layer
- b^[l] = bias vector for layer l
```

**Forward Propagation Example:**

```python
# Simple 3-layer network
import numpy as np

def forward_propagation(X, parameters):
    """
    X: input features (n_features, m_samples)
    parameters: dictionary containing W1, b1, W2, b2, W3, b3
    """
    # Layer 1: Input → Hidden (128 neurons)
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)
    
    # Layer 2: Hidden → Hidden (64 neurons)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = relu(Z2)
    
    # Layer 3: Hidden → Output (1 neuron for regression)
    Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
    A3 = Z3  # Linear activation for regression
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    return A3, cache
```

### 1.3 Activation Functions

Activation functions introduce non-linearity, enabling neural networks to learn complex patterns.

**ReLU (Rectified Linear Unit)**
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

**Advantages:**
- Computationally efficient
- Helps mitigate vanishing gradient problem
- Induces sparsity (many neurons output 0)
- Default choice for hidden layers

**Disadvantages:**
- Dead ReLU problem (neurons stuck at 0)
- Not zero-centered

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

**Visualization Concept:**
```
   |     /
   |    /
   |   /
   |  /
___|/_________
   0
```

**Leaky ReLU**
```
f(x) = max(0.01x, x)
```

**Advantages:**
- Prevents dead neurons
- Small gradient for negative values

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

**Sigmoid**
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```

**Use Cases:**
- Binary classification output layer
- Gate mechanisms in LSTM/GRU

**Disadvantages:**
- Vanishing gradient problem
- Not zero-centered
- Computationally expensive

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

**Tanh (Hyperbolic Tangent)**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - f(x)²
```

**Advantages:**
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid

```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

**Softmax (Multi-Class Output)**
```
f(x_i) = e^(x_i) / Σ(e^(x_j))
```

**Properties:**
- Outputs sum to 1 (probability distribution)
- Used for multi-class classification

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
```

**Activation Function Selection Guide:**

| Layer Type | Recommended Activation | Reason |
|------------|----------------------|--------|
| Hidden layers (general) | ReLU | Fast, effective, prevents vanishing gradients |
| Hidden layers (negative values important) | Leaky ReLU / ELU | Allows negative activations |
| Output (regression) | Linear | Unrestricted output range |
| Output (binary classification) | Sigmoid | Output in [0, 1] range |
| Output (multi-class) | Softmax | Probability distribution |
| Recurrent networks | Tanh | Zero-centered, bounded |

### 1.4 Loss Functions and Backpropagation

**Loss Functions** quantify how well the model's predictions match the true values.

**Mean Squared Error (MSE) - Regression**
```
L(y, ŷ) = (1/n) × Σ(y_i - ŷ_i)²
```

**Characteristics:**
- Penalizes large errors more heavily
- Sensitive to outliers
- Smooth gradient

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]
```

**Mean Absolute Error (MAE) - Robust Regression**
```
L(y, ŷ) = (1/n) × Σ|y_i - ŷ_i|
```

**Advantages:**
- Less sensitive to outliers
- All errors weighted equally

```python
def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

**Binary Cross-Entropy - Binary Classification**
```
L(y, ŷ) = -(1/n) × Σ[y_i × log(ŷ_i) + (1-y_i) × log(1-ŷ_i)]
```

**Used when:**
- Predicting probability of single class
- Output: sigmoid activation

```python
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

**Categorical Cross-Entropy - Multi-Class Classification**
```
L(y, ŷ) = -(1/n) × Σ Σ y_ij × log(ŷ_ij)
```

**Used when:**
- Multiple mutually exclusive classes
- Output: softmax activation

```python
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
```

**Backpropagation Algorithm**

Backpropagation computes gradients of the loss with respect to all parameters using the chain rule.

**Algorithm Steps:**

1. **Forward Pass**: Compute predictions and loss
2. **Output Layer Gradient**: ∂L/∂a^[L]
3. **Backward Pass**: Compute gradients layer by layer
4. **Parameter Update**: Update weights and biases

**Mathematical Derivation:**

For layer l:
```
dZ^[l] = dA^[l] ⊙ f'(Z^[l])
dW^[l] = (1/m) × dZ^[l] × A^[l-1]^T
db^[l] = (1/m) × Σ dZ^[l]
dA^[l-1] = W^[l]^T × dZ^[l]

Where:
- ⊙ represents element-wise multiplication
- f' is the derivative of activation function
- m is the number of training examples
```

**Complete Backpropagation Implementation:**

```python
def backward_propagation(X, Y, cache, parameters):
    """
    Implements backpropagation for 3-layer network
    
    Args:
        X: input features
        Y: true labels
        cache: forward propagation cache
        parameters: model parameters (W1, b1, W2, b2, W3, b3)
    
    Returns:
        gradients: dictionary containing dW1, db1, dW2, db2, dW3, db3
    """
    m = X.shape[1]
    
    # Retrieve cached values
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2 = cache['Z1'], cache['Z2']
    
    # Output layer (Layer 3)
    dZ3 = A3 - Y  # For MSE loss with linear activation
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # Hidden layer 2
    dA2 = np.dot(parameters['W3'].T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Hidden layer 1
    dA1 = np.dot(parameters['W2'].T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {
        'dW1': dW1, 'db1': db1,
        'dW2': dW2, 'db2': db2,
        'dW3': dW3, 'db3': db3
    }
    
    return gradients
```

### 1.5 Optimization Algorithms

Optimization algorithms update model parameters to minimize the loss function.

**Gradient Descent (Batch)**

Updates parameters using the entire dataset:
```
W := W - α × ∂L/∂W
b := b - α × ∂L/∂b

Where α is the learning rate
```

**Pros**: Stable convergence, exact gradient
**Cons**: Slow for large datasets, memory intensive

```python
def gradient_descent(parameters, gradients, learning_rate):
    """
    Update parameters using batch gradient descent
    """
    for key in parameters.keys():
        parameters[key] -= learning_rate * gradients['d' + key]
    return parameters
```

**Stochastic Gradient Descent (SGD)**

Updates parameters using one sample at a time:
```
W := W - α × ∂L_i/∂W  (for single sample i)
```

**Pros**: Fast updates, can escape local minima
**Cons**: Noisy updates, unstable convergence

**Mini-Batch Gradient Descent**

Compromise between batch and stochastic (typically 32-256 samples):

```python
def mini_batch_gradient_descent(X, Y, parameters, batch_size=32, learning_rate=0.01):
    """
    Mini-batch gradient descent implementation
    """
    m = X.shape[1]
    num_batches = m // batch_size
    
    for i in range(num_batches):
        # Get mini-batch
        start = i * batch_size
        end = start + batch_size
        X_batch = X[:, start:end]
        Y_batch = Y[:, start:end]
        
        # Forward propagation
        A3, cache = forward_propagation(X_batch, parameters)
        
        # Backward propagation
        gradients = backward_propagation(X_batch, Y_batch, cache, parameters)
        
        # Update parameters
        parameters = gradient_descent(parameters, gradients, learning_rate)
    
    return parameters
```

**Momentum**

Accelerates SGD by accumulating velocity in relevant direction:
```
v := β × v + (1-β) × ∂L/∂W
W := W - α × v

Common β values: 0.9, 0.99
```

**Benefits:**
- Smooths out oscillations
- Faster convergence
- Better navigation of ravines

```python
def momentum_optimizer(parameters, gradients, velocity, beta=0.9, learning_rate=0.01):
    """
    Parameters update with momentum
    """
    for key in parameters.keys():
        # Update velocity
        velocity['v' + key] = beta * velocity['v' + key] + (1 - beta) * gradients['d' + key]
        # Update parameters
        parameters[key] -= learning_rate * velocity['v' + key]
    
    return parameters, velocity
```

**RMSprop (Root Mean Square Propagation)**

Adapts learning rate per parameter based on recent gradients:
```
s := β × s + (1-β) × (∂L/∂W)²
W := W - α × (∂L/∂W) / √(s + ε)

Where ε is a small constant for numerical stability (typically 1e-8)
```

**Benefits:**
- Adaptive learning rates
- Works well for non-stationary objectives
- Good for RNNs

```python
def rmsprop_optimizer(parameters, gradients, cache, beta=0.999, learning_rate=0.001, epsilon=1e-8):
    """
    RMSprop optimization
    """
    for key in parameters.keys():
        # Update cache (squared gradients)
        cache['s' + key] = beta * cache['s' + key] + (1 - beta) * gradients['d' + key]**2
        # Update parameters
        parameters[key] -= learning_rate * gradients['d' + key] / (np.sqrt(cache['s' + key]) + epsilon)
    
    return parameters, cache
```

**Adam (Adaptive Moment Estimation)**

Combines momentum and RMSprop:
```
m := β₁ × m + (1-β₁) × ∂L/∂W          (momentum)
v := β₂ × v + (1-β₂) × (∂L/∂W)²       (RMSprop)
m̂ := m / (1-β₁^t)                      (bias correction)
v̂ := v / (1-β₂^t)                      (bias correction)
W := W - α × m̂ / (√v̂ + ε)

Common values: β₁=0.9, β₂=0.999, ε=1e-8
```

**Benefits:**
- Most popular optimizer for deep learning
- Works well with sparse gradients
- Combines benefits of momentum and RMSprop
- Bias correction for initialization

```python
def adam_optimizer(parameters, gradients, adam_cache, t, beta1=0.9, beta2=0.999, 
                   learning_rate=0.001, epsilon=1e-8):
    """
    Adam optimization with bias correction
    
    Args:
        t: iteration number (for bias correction)
    """
    for key in parameters.keys():
        # Update momentum
        adam_cache['m' + key] = beta1 * adam_cache['m' + key] + (1 - beta1) * gradients['d' + key]
        
        # Update RMSprop
        adam_cache['v' + key] = beta2 * adam_cache['v' + key] + (1 - beta2) * gradients['d' + key]**2
        
        # Bias correction
        m_corrected = adam_cache['m' + key] / (1 - beta1**t)
        v_corrected = adam_cache['v' + key] / (1 - beta2**t)
        
        # Update parameters
        parameters[key] -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
    
    return parameters, adam_cache
```

**AdamW (Adam with Weight Decay)**

Adam with decoupled weight decay regularization:
```
W := W - α × (m̂ / (√v̂ + ε) + λ × W)

Where λ is the weight decay coefficient
```

**Benefits:**
- Better generalization than Adam
- Proper weight decay implementation
- State-of-the-art for many tasks

```python
# Using PyTorch implementation
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Optimizer Selection Guide:**

| Scenario | Recommended Optimizer | Learning Rate |
|----------|----------------------|---------------|
| General purpose / starting point | Adam | 1e-3 |
| Large dataset, need speed | SGD with momentum | 1e-2 (with decay) |
| RNNs / sequence models | Adam or RMSprop | 1e-3 to 1e-4 |
| Fine-tuning pretrained models | AdamW | 1e-5 to 1e-4 |
| Non-stationary problems | RMSprop | 1e-3 |
| When overfitting occurs | AdamW or SGD | Lower rates |

**Learning Rate Scheduling:**

```python
# Learning rate decay
def lr_decay(initial_lr, epoch, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

# Step decay
def step_decay(initial_lr, epoch, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** np.floor(epoch / epochs_drop))

# Cosine annealing
def cosine_annealing(initial_lr, epoch, total_epochs):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
```

---

## 2. Feedforward Neural Networks

### 2.1 Architecture Design

A feedforward neural network (FNN) is the simplest type of artificial neural network where information moves in only one direction—forward—from input to output.

**Design Principles:**

**Layer Size Guidelines:**
- **Input layer**: Size = number of molecular features (e.g., 2048 for Morgan fingerprints)
- **Hidden layers**: Start with 128-256 neurons, gradually decrease
- **Output layer**: 
  - Regression: 1 neuron
  - Binary classification: 1 neuron
  - Multi-class: Number of classes

**Architecture Patterns:**

**Pattern 1: Pyramid Structure** (Recommended for most molecular tasks)
```
Input (2048) → Hidden1 (512) → Hidden2 (256) → Hidden3 (128) → Output (1)
```
- Progressively reduces dimensionality
- Learns hierarchical features

**Pattern 2: Hourglass Structure**
```
Input (2048) → Hidden1 (256) → Hidden2 (128) → Hidden3 (256) → Output (1)
```
- Creates bottleneck representation
- Useful for learning compressed features

**Pattern 3: Constant Width**
```
Input (2048) → Hidden1 (256) → Hidden2 (256) → Hidden3 (256) → Output (1)
```
- Maintains capacity throughout
- Good for complex non-linear mappings

**Depth vs Width Trade-off:**

| Architecture | Pros | Cons | Best For |
|--------------|------|------|----------|
| Deep & Narrow (5+ layers, 64-128 neurons) | Hierarchical features, fewer parameters | Harder to train, vanishing gradients | Complex patterns, large datasets |
| Shallow & Wide (2-3 layers, 512+ neurons) | Easier to train, stable | More parameters, less hierarchical | Simple patterns, small datasets |
| Balanced (3-4 layers, 128-256 neurons) | Good trade-off | - | General purpose, recommended starting point |

**Complete Architecture Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularFNN(nn.Module):
    """
    Feedforward Neural Network for molecular property prediction
    """
    def __init__(self, input_dim=2048, hidden_dims=[512, 256, 128], 
                 output_dim=1, dropout_rate=0.3):
        """
        Args:
            input_dim: Size of input features (e.g., fingerprint length)
            hidden_dims: List of hidden layer sizes
            output_dim: Number of output neurons (1 for regression)
            dropout_rate: Dropout probability for regularization
        """
        super(MolecularFNN, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_embeddings(self, x):
        """
        Extract learned representations from second-to-last layer
        Useful for visualization and transfer learning
        """
        for layer in self.network[:-1]:  # All layers except final
            x = layer(x)
        return x

# Example instantiation
model = MolecularFNN(
    input_dim=2048,           # Morgan fingerprint size
    hidden_dims=[512, 256, 128],
    output_dim=1,             # Single regression output
    dropout_rate=0.3
)

print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 2.2 Full Training Example

**Complete Training Pipeline:**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

class MolecularDataset(Dataset):
    """Custom Dataset for molecular data"""
    
    def __init__(self, smiles_list, labels, radius=2, n_bits=2048):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: Array of target values
            radius: Morgan fingerprint radius
            n_bits: Fingerprint length
        """
        self.fingerprints = []
        self.labels = []
        
        for smiles, label in zip(smiles_list, labels):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate Morgan fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                self.fingerprints.append(np.array(fp))
                self.labels.append(label)
        
        self.fingerprints = torch.FloatTensor(np.array(self.fingerprints))
        self.labels = torch.FloatTensor(np.array(self.labels)).reshape(-1, 1)
    
    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        return self.fingerprints[idx], self.labels[idx]

# ============================================================================
# 2. TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# ============================================================================
# 3. VALIDATION FUNCTION
# ============================================================================

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return avg_loss, rmse, mae, r2

# ============================================================================
# 4. COMPLETE TRAINING PIPELINE
# ============================================================================

def train_molecular_model(smiles_train, y_train, smiles_val, y_val, 
                          config=None):
    """
    Complete training pipeline for molecular property prediction
    
    Args:
        smiles_train: Training SMILES
        y_train: Training labels
        smiles_val: Validation SMILES
        y_val: Validation labels
        config: Configuration dictionary
    
    Returns:
        trained_model: Best model
        history: Training history
    """
    # Default configuration
    if config is None:
        config = {
            'input_dim': 2048,
            'hidden_dims': [512, 256, 128],
            'output_dim': 1,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 100,
            'patience': 15,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MolecularDataset(smiles_train, y_train)
    val_dataset = MolecularDataset(smiles_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Initialize model
    model = MolecularFNN(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        output_dim=config['output_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_rmse': [], 'val_mae': [], 'val_r2': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_rmse, val_mae, val_r2 = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, "
                  f"MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, history

# ============================================================================
# 5. EXAMPLE USAGE
# ============================================================================

# Generate synthetic data for demonstration
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic molecular data"""
    np.random.seed(42)
    
    # Simple SMILES for demonstration (in practice, use real dataset)
    base_smiles = ['CCO', 'CC(C)O', 'CCCO', 'CC(C)CO', 'CCCCO']
    smiles_list = np.random.choice(base_smiles, n_samples)
    
    # Synthetic labels (in practice, use real property values)
    labels = np.random.randn(n_samples) * 2 + 5
    
    return smiles_list, labels

# Generate data
smiles, labels = generate_synthetic_data(1000)

# Split data
smiles_train, smiles_temp, y_train, y_temp = train_test_split(
    smiles, labels, test_size=0.3, random_state=42
)
smiles_val, smiles_test, y_val, y_test = train_test_split(
    smiles_temp, y_temp, test_size=0.5, random_state=42
)

# Train model
model, history = train_molecular_model(smiles_train, y_train, smiles_val, y_val)

# Visualize training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 3, 2)
plt.plot(history['val_rmse'], label='RMSE')
plt.plot(history['val_mae'], label='MAE')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.title('Validation Errors')

plt.subplot(1, 3, 3)
plt.plot(history['val_r2'])
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Validation R²')

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 Best Practices and Tips

**1. Data Preprocessing**

- **Normalize inputs**: Scale features to similar ranges
  ```python
  from sklearn.preprocessing import StandardScaler
  
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.transform(X_val)
  ```

- **Handle missing values**: Remove or impute before training
- **Remove duplicates**: Ensure no data leakage between train/val/test sets

**2. Architecture Selection**

- **Start simple**: Begin with 2-3 hidden layers
- **Increase gradually**: Add complexity only if needed
- **Monitor overfitting**: If train loss << val loss, model too complex

**3. Hyperparameter Tuning Priority**

| Priority | Hyperparameter | Impact | Typical Range |
|----------|----------------|--------|---------------|
| HIGH | Learning rate | Critical for convergence | 1e-4 to 1e-2 |
| HIGH | Batch size | Memory and convergence speed | 32, 64, 128, 256 |
| MEDIUM | Number of layers | Model capacity | 2-5 |
| MEDIUM | Neurons per layer | Model capacity | 64, 128, 256, 512 |
| MEDIUM | Dropout rate | Regularization | 0.1-0.5 |
| LOW | Optimizer | Usually Adam works | Adam, AdamW |
| LOW | Activation function | ReLU usually best | ReLU, Leaky ReLU |

**4. Regularization Techniques**

```python
# L2 Regularization (Weight Decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Dropout (already in model architecture)
nn.Dropout(p=0.3)

# Batch Normalization (already in model architecture)
nn.BatchNorm1d(hidden_dim)

# Early Stopping (implemented in training loop)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
else:
    patience_counter += 1
```

**5. Debugging Checklist**

- **Problem**: Loss is NaN
  - **Solution**: Reduce learning rate, check for invalid inputs, add gradient clipping
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

- **Problem**: Loss not decreasing
  - **Solution**: Increase learning rate, check data preprocessing, verify labels

- **Problem**: Training loss decreasing but validation loss increasing
  - **Solution**: Overfitting - increase dropout, add L2 regularization, reduce model size

- **Problem**: Both losses high and not improving
  - **Solution**: Underfitting - increase model capacity, decrease regularization, train longer

**6. Monitoring Training**

```python
# Use TensorBoard for real-time monitoring
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# During training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Metrics/RMSE', val_rmse, epoch)
writer.add_scalar('Metrics/R2', val_r2, epoch)

# View with: tensorboard --logdir=runs
```

**7. Model Saving and Loading**

```python
# Save complete model state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_val_loss,
    'history': history
}, 'best_model.pth')

# Load model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**8. Tips for Molecular Data**

- **Use appropriate fingerprints**: Morgan (most common), MACCS, RDKit fingerprints
- **Consider molecular size**: Normalize by molecular weight or atom count if relevant
- **Handle invalid SMILES**: Filter out molecules that RDKit cannot parse
- **Augmentation**: Consider SMILES enumeration for data augmentation

```python
# SMILES enumeration for augmentation
from rdkit import Chem

def enumerate_smiles(smiles, n_variants=5):
    """Generate different SMILES representations of same molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    variants = []
    for _ in range(n_variants):
        variants.append(Chem.MolToSmiles(mol, doRandom=True))
    
    return list(set(variants))
```

---

## 3. Multi-Task Learning

### 3.1 Why and When to Use Multi-Task Learning

Multi-task learning (MTL) is a machine learning paradigm where a model simultaneously learns multiple related tasks, sharing representations between tasks.

**Key Benefits:**

1. **Improved Generalization**: Shared representations act as implicit regularization
2. **Data Efficiency**: Tasks with more data help tasks with less data
3. **Related Features**: Capture common patterns across molecular properties
4. **Transfer Learning**: Learn general molecular representations
5. **Computational Efficiency**: One model predicts multiple properties

**When to Use MTL:**

**Ideal Scenarios:**
- **Related Properties**: Predicting solubility, LogP, and permeability (all related to molecular polarity)
- **Limited Data**: Some tasks have abundant data, others have scarce data
- **Shared Features**: Tasks depend on similar molecular features
- **Multiple Endpoints**: Drug discovery (ADMET properties all relevant)

**Not Recommended:**
- **Unrelated Tasks**: Predicting solubility and catalytic activity (different mechanisms)
- **Conflicting Objectives**: Tasks requiring opposite feature representations
- **Single Task is Sufficient**: When only one property matters

**Examples in Drug Discovery:**

```
Task Group 1: ADMET Properties
├─ Absorption (Caco-2 permeability)
├─ Distribution (BBB permeability, plasma protein binding)
├─ Metabolism (CYP450 inhibition, metabolic stability)
├─ Excretion (clearance, half-life)
└─ Toxicity (hERG inhibition, hepatotoxicity)

Task Group 2: Physical Properties
├─ Solubility (aqueous, LogS)
├─ Lipophilicity (LogP, LogD)
└─ Permeability (PAMPA, Caco-2)

Task Group 3: Biological Activity
├─ Target binding (IC50, Ki)
├─ Cell viability (CC50)
└─ Selectivity across targets
```

### 3.2 Multi-Task Architecture

**Hard Parameter Sharing** (Most Common)

All tasks share hidden layers, separate output heads:

```
              Input (Molecular Features)
                       |
            Shared Hidden Layers
                   /   |   \
           Task 1  Task 2  Task 3
          Output  Output  Output
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn

class MultiTaskMolecularModel(nn.Module):
    """
    Multi-task neural network for molecular property prediction
    """
    def __init__(self, input_dim=2048, shared_dims=[512, 256], 
                 task_configs=None, dropout_rate=0.3):
        """
        Args:
            input_dim: Size of input features
            shared_dims: List of shared hidden layer sizes
            task_configs: List of dicts with task specifications
                         [{'name': 'task1', 'output_dim': 1, 'task_type': 'regression'}, ...]
            dropout_rate: Dropout probability
        """
        super(MultiTaskMolecularModel, self).__init__()
        
        if task_configs is None:
            task_configs = [
                {'name': 'solubility', 'output_dim': 1, 'task_type': 'regression'},
                {'name': 'toxicity', 'output_dim': 1, 'task_type': 'classification'}
            ]
        
        self.task_configs = task_configs
        self.task_names = [task['name'] for task in task_configs]
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task in task_configs:
            task_name = task['name']
            output_dim = task['output_dim']
            
            # Task-specific layers (2 layers)
            task_head = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, output_dim)
            )
            
            self.task_heads[task_name] = task_head
    
    def forward(self, x):
        """
        Forward pass through shared network and all task heads
        
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            Dictionary of predictions for each task
        """
        # Shared representations
        shared_features = self.shared_network(x)
        
        # Task-specific predictions
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs
    
    def get_shared_features(self, x):
        """Extract shared representations for visualization or transfer learning"""
        return self.shared_network(x)

# Example instantiation
task_configs = [
    {'name': 'solubility', 'output_dim': 1, 'task_type': 'regression'},
    {'name': 'bbb_permeability', 'output_dim': 1, 'task_type': 'regression'},
    {'name': 'toxicity', 'output_dim': 1, 'task_type': 'classification'},
    {'name': 'cyp450_inhibition', 'output_dim': 5, 'task_type': 'multi-class'}
]

model = MultiTaskMolecularModel(
    input_dim=2048,
    shared_dims=[512, 256],
    task_configs=task_configs,
    dropout_rate=0.3
)

print(model)
```

**Soft Parameter Sharing** (Advanced)

Each task has its own network, but networks are constrained to be similar:

```python
class SoftParameterSharingModel(nn.Module):
    """
    Soft parameter sharing: separate networks with similarity constraints
    """
    def __init__(self, input_dim, hidden_dims, num_tasks):
        super(SoftParameterSharingModel, self).__init__()
        
        # Create separate networks for each task
        self.task_networks = nn.ModuleList([
            self._build_network(input_dim, hidden_dims)
            for _ in range(num_tasks)
        ])
    
    def _build_network(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        outputs = []
        for network in self.task_networks:
            outputs.append(network(x))
        return torch.cat(outputs, dim=1)
    
    def compute_similarity_loss(self, lambda_reg=0.01):
        """
        Regularization term to keep task networks similar
        """
        similarity_loss = 0
        num_tasks = len(self.task_networks)
        
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                # L2 distance between parameters
                for p1, p2 in zip(self.task_networks[i].parameters(), 
                                 self.task_networks[j].parameters()):
                    similarity_loss += torch.norm(p1 - p2, p=2)
        
        return lambda_reg * similarity_loss
```

### 3.3 Training Strategies

**Multi-Task Loss Function:**

```python
def compute_multitask_loss(outputs, labels, task_configs, task_weights=None):
    """
    Compute weighted combination of task-specific losses
    
    Args:
        outputs: Dict of predictions {task_name: predictions}
        labels: Dict of true labels {task_name: labels}
        task_configs: List of task configurations
        task_weights: Dict of loss weights {task_name: weight}
    
    Returns:
        total_loss: Combined loss
        task_losses: Dict of individual task losses
    """
    if task_weights is None:
        task_weights = {task['name']: 1.0 for task in task_configs}
    
    task_losses = {}
    total_loss = 0
    
    for task in task_configs:
        task_name = task['name']
        task_type = task['task_type']
        
        # Skip if no labels for this task in batch
        if task_name not in labels or labels[task_name] is None:
            continue
        
        pred = outputs[task_name]
        true = labels[task_name]
        
        # Select appropriate loss function
        if task_type == 'regression':
            loss = nn.MSELoss()(pred, true)
        elif task_type == 'classification':
            loss = nn.BCEWithLogitsLoss()(pred, true)
        elif task_type == 'multi-class':
            loss = nn.CrossEntropyLoss()(pred, true)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_losses[task_name] = loss.item()
        total_loss += task_weights[task_name] * loss
    
    return total_loss, task_losses
```

**Complete Training Loop:**

```python
def train_multitask_model(model, train_loader, val_loader, task_configs, 
                         num_epochs=100, device='cuda'):
    """
    Complete multi-task training pipeline
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Initialize task weights (can be learned or fixed)
    task_weights = {task['name']: 1.0 for task in task_configs}
    
    history = {'train_loss': [], 'val_loss': [], 'task_losses': {}}
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_task_losses = {task['name']: 0 for task in task_configs}
        
        for batch_x, batch_labels in train_loader:
            batch_x = batch_x.to(device)
            batch_labels = {k: v.to(device) if v is not None else None 
                          for k, v in batch_labels.items()}
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss, task_losses = compute_multitask_loss(
                outputs, batch_labels, task_configs, task_weights
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for task_name, task_loss in task_losses.items():
                train_task_losses[task_name] += task_loss
        
        # Validation
        model.eval()
        val_loss = 0
        val_task_losses = {task['name']: 0 for task in task_configs}
        
        with torch.no_grad():
            for batch_x, batch_labels in val_loader:
                batch_x = batch_x.to(device)
                batch_labels = {k: v.to(device) if v is not None else None 
                              for k, v in batch_labels.items()}
                
                outputs = model(batch_x)
                loss, task_losses = compute_multitask_loss(
                    outputs, batch_labels, task_configs, task_weights
                )
                
                val_loss += loss.item()
                for task_name, task_loss in task_losses.items():
                    val_task_losses[task_name] += task_loss
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            for task in task_configs:
                task_name = task['name']
                print(f"  {task_name}: {val_task_losses[task_name]/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multitask_model.pth')
    
    return model, history
```

### 3.4 Task Balancing Techniques

Balancing multiple tasks is crucial for effective multi-task learning.

**1. Manual Weight Tuning**

```python
# Simple fixed weights
task_weights = {
    'solubility': 1.0,
    'bbb_permeability': 2.0,  # Prioritize this task
    'toxicity': 1.5,
    'cyp450_inhibition': 1.0
}
```

**2. Uncertainty Weighting** (Kendall et al., 2018)

Learns task weights based on homoscedastic uncertainty:

```python
class MultiTaskLossWithUncertainty(nn.Module):
    """
    Automatically learns task weights based on uncertainty
    """
    def __init__(self, num_tasks):
        super(MultiTaskLossWithUncertainty, self).__init__()
        # Log variance for each task (learned parameters)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        """
        Args:
            losses: List of individual task losses
        
        Returns:
            Weighted total loss
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss

# Usage in training
uncertainty_loss = MultiTaskLossWithUncertainty(num_tasks=4)
optimizer = optim.Adam(list(model.parameters()) + list(uncertainty_loss.parameters()))
```

**3. Gradient Normalization (GradNorm)**

Balances tasks by normalizing gradient magnitudes:

```python
def compute_gradnorm_weights(model, losses, alpha=1.5):
    """
    Compute task weights using GradNorm algorithm
    
    Args:
        model: Neural network model
        losses: List of task losses
        alpha: Hyperparameter for balancing (default: 1.5)
    
    Returns:
        Updated task weights
    """
    # Get gradients for last shared layer
    shared_params = list(model.shared_network.parameters())[-1]
    
    gradients = []
    for loss in losses:
        grad = torch.autograd.grad(loss, shared_params, retain_graph=True)[0]
        gradients.append(torch.norm(grad))
    
    # Compute inverse training rate
    loss_ratios = [loss / losses[0] for loss in losses]
    mean_loss_ratio = sum(loss_ratios) / len(loss_ratios)
    
    # Compute target gradients
    target_grads = [mean_loss_ratio * (ratio ** alpha) for ratio in loss_ratios]
    
    # Compute weights
    weights = [target / grad for target, grad in zip(target_grads, gradients)]
    
    # Normalize
    weights = [w / sum(weights) * len(weights) for w in weights]
    
    return weights
```

**4. Dynamic Task Prioritization**

Adjust weights during training based on task performance:

```python
class DynamicTaskWeighting:
    """
    Dynamically adjust task weights during training
    """
    def __init__(self, num_tasks, initial_weights=None):
        if initial_weights is None:
            self.weights = [1.0] * num_tasks
        else:
            self.weights = initial_weights
        
        self.loss_history = [[] for _ in range(num_tasks)]
    
    def update_weights(self, epoch, task_losses, strategy='inverse_performance'):
        """
        Update weights based on task performance
        
        Strategies:
        - 'inverse_performance': Higher weight for worse-performing tasks
        - 'uncertainty': Higher weight for high-variance tasks
        - 'curriculum': Gradually increase difficulty
        """
        for i, loss in enumerate(task_losses):
            self.loss_history[i].append(loss)
        
        if epoch < 5:  # Wait for some history
            return self.weights
        
        if strategy == 'inverse_performance':
            # Give more weight to tasks with higher recent loss
            recent_losses = [np.mean(history[-5:]) for history in self.loss_history]
            self.weights = [loss / sum(recent_losses) * len(recent_losses) 
                           for loss in recent_losses]
        
        elif strategy == 'uncertainty':
            # Give more weight to high-variance tasks
            variances = [np.var(history[-10:]) for history in self.loss_history]
            self.weights = [var / sum(variances) * len(variances) 
                           for var in variances]
        
        return self.weights
```

### 3.5 Performance Comparison

**Evaluation Metrics for Multi-Task Learning:**

```python
def evaluate_multitask_model(model, test_loader, task_configs, device='cuda'):
    """
    Comprehensive evaluation of multi-task model
    """
    model.eval()
    model = model.to(device)
    
    # Store predictions and labels
    predictions = {task['name']: [] for task in task_configs}
    true_labels = {task['name']: [] for task in task_configs}
    
    with torch.no_grad():
        for batch_x, batch_labels in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            for task in task_configs:
                task_name = task['name']
                if task_name in batch_labels and batch_labels[task_name] is not None:
                    predictions[task_name].extend(
                        outputs[task_name].cpu().numpy()
                    )
                    true_labels[task_name].extend(
                        batch_labels[task_name].cpu().numpy()
                    )
    
    # Compute metrics for each task
    results = {}
    
    for task in task_configs:
        task_name = task['name']
        task_type = task['task_type']
        
        pred = np.array(predictions[task_name]).flatten()
        true = np.array(true_labels[task_name]).flatten()
        
        if task_type == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            results[task_name] = {
                'RMSE': np.sqrt(mean_squared_error(true, pred)),
                'MAE': mean_absolute_error(true, pred),
                'R2': r2_score(true, pred)
            }
        
        elif task_type == 'classification':
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
            
            pred_binary = (pred > 0).astype(int)
            
            results[task_name] = {
                'ROC-AUC': roc_auc_score(true, pred),
                'Accuracy': accuracy_score(true, pred_binary),
                'F1': f1_score(true, pred_binary)
            }
    
    return results

# Compare with single-task models
def compare_single_vs_multitask(smiles_data, labels_dict, task_configs):
    """
    Train single-task models and compare with multi-task model
    """
    results = {'single_task': {}, 'multi_task': {}}
    
    # Train single-task models
    for task in task_configs:
        task_name = task['name']
        print(f"\nTraining single-task model for {task_name}...")
        
        single_model = MolecularFNN(input_dim=2048, output_dim=1)
        # Train single_model (code similar to previous examples)
        # ...
        results['single_task'][task_name] = evaluate_model(single_model, test_data)
    
    # Train multi-task model
    print("\nTraining multi-task model...")
    multitask_model = MultiTaskMolecularModel(task_configs=task_configs)
    # Train multitask_model
    # ...
    results['multi_task'] = evaluate_multitask_model(multitask_model, test_data, task_configs)
    
    # Print comparison
    print("\n" + "="*60)
    print("SINGLE-TASK vs MULTI-TASK COMPARISON")
    print("="*60)
    
    for task in task_configs:
        task_name = task['name']
        print(f"\n{task_name.upper()}:")
        print(f"  Single-task RMSE: {results['single_task'][task_name]['RMSE']:.4f}")
        print(f"  Multi-task RMSE:  {results['multi_task'][task_name]['RMSE']:.4f}")
        
        improvement = ((results['single_task'][task_name]['RMSE'] - 
                       results['multi_task'][task_name]['RMSE']) / 
                       results['single_task'][task_name]['RMSE'] * 100)
        print(f"  Improvement: {improvement:.2f}%")
    
    return results
```

**Expected Results:**

| Task | Single-Task RMSE | Multi-Task RMSE | Improvement |
|------|------------------|-----------------|-------------|
| Solubility | 0.85 | 0.72 | 15.3% |
| BBB Permeability | 0.92 | 0.79 | 14.1% |
| Toxicity (AUC) | 0.78 | 0.84 | 7.7% |
| CYP450 Inhibition | 0.88 | 0.81 | 8.0% |

Multi-task learning typically shows 10-20% improvement, especially for tasks with limited data.

---

## 4. Convolutional Neural Networks

Convolutional Neural Networks (CNNs) excel at extracting local patterns and spatial hierarchies, making them suitable for sequence and image data representations of molecules.

### 4.1 1D CNNs for SMILES

SMILES strings can be treated as sequences where local substructures (like functional groups) are important patterns.

**Architecture Overview:**

```
SMILES: "CCO" → Embedding → Conv1D layers → Pooling → Dense → Output
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SMILES_CNN(nn.Module):
    """
    1D Convolutional Neural Network for SMILES strings
    """
    def __init__(self, vocab_size=50, embedding_dim=128, num_filters=64, 
                 filter_sizes=[3, 5, 7], hidden_dim=256, output_dim=1, dropout_rate=0.3):
        """
        Args:
            vocab_size: Size of character vocabulary (typically 40-60 for SMILES)
            embedding_dim: Dimension of character embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes (kernel sizes)
            hidden_dim: Size of fully connected layer
            output_dim: Output size (1 for regression)
            dropout_rate: Dropout probability
        """
        super(SMILES_CNN, self).__init__()
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Fully connected layers
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(len(filter_sizes) * num_filters)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, max_length)
        
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        # Embedding: (batch_size, max_length) -> (batch_size, max_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Transpose for Conv1d: (batch_size, embedding_dim, max_length)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution + ReLU: (batch_size, num_filters, length - kernel_size + 1)
            conv_out = F.relu(conv(embedded))
            
            # Max pooling: (batch_size, num_filters, 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            
            # Flatten: (batch_size, num_filters)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all filter outputs: (batch_size, len(filter_sizes) * num_filters)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Batch normalization
        normalized = self.batch_norm(concatenated)
        
        # Fully connected layers
        hidden = F.relu(self.fc1(self.dropout(normalized)))
        output = self.fc2(self.dropout(hidden))
        
        return output

# ============================================================================
# SMILES Tokenization
# ============================================================================

class SMILESTokenizer:
    """
    Tokenizer for SMILES strings
    """
    def __init__(self):
        # Common SMILES tokens
        self.tokens = [
            'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P',  # Atoms
            'c', 'n', 'o', 's',  # Aromatic atoms
            '=', '#',  # Bonds
            '(', ')',  # Branches
            '[', ']',  # Atom properties
            '+', '-',  # Charges
            '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Ring numbers
            '@', '@@',  # Chirality
            'H',  # Hydrogen
            '/', '\\'  # Stereochemistry
        ]
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # Create vocabulary
        self.vocab = self.special_tokens + self.tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        self.pad_idx = self.token_to_idx['<PAD>']
        self.unk_idx = self.token_to_idx['<UNK>']
    
    def tokenize(self, smiles):
        """
        Tokenize SMILES string into characters
        """
        tokens = []
        i = 0
        while i < len(smiles):
            # Check for two-character tokens (Cl, Br, @@)
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.tokens:
                    tokens.append(two_char)
                    i += 2
                    continue
            
            # Single character token
            token = smiles[i]
            tokens.append(token if token in self.tokens else '<UNK>')
            i += 1
        
        return tokens
    
    def encode(self, smiles, max_length=100):
        """
        Convert SMILES to integer sequence
        """
        tokens = self.tokenize(smiles)
        
        # Convert to indices
        indices = [self.token_to_idx.get(token, self.unk_idx) for token in tokens]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.pad_idx] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def batch_encode(self, smiles_list, max_length=100):
        """
        Encode batch of SMILES strings
        """
        return [self.encode(smiles, max_length) for smiles in smiles_list]

# ============================================================================
# Dataset and Training
# ============================================================================

class SMILES_Dataset(Dataset):
    """
    Dataset for SMILES strings
    """
    def __init__(self, smiles_list, labels, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.encoded_smiles = []
        self.labels = []
        
        for smiles, label in zip(smiles_list, labels):
            try:
                encoded = torch.LongTensor(tokenizer.encode(smiles, max_length))
                self.encoded_smiles.append(encoded)
                self.labels.append(label)
            except:
                continue
        
        self.labels = torch.FloatTensor(self.labels).reshape(-1, 1)
    
    def __len__(self):
        return len(self.encoded_smiles)
    
    def __getitem__(self, idx):
        return self.encoded_smiles[idx], self.labels[idx]

# Example usage
tokenizer = SMILESTokenizer()
print(f"Vocabulary size: {len(tokenizer.vocab)}")

# Create model
model = SMILES_CNN(
    vocab_size=len(tokenizer.vocab),
    embedding_dim=128,
    num_filters=64,
    filter_sizes=[3, 5, 7],
    hidden_dim=256,
    output_dim=1,
    dropout_rate=0.3
)

# Create dataset
smiles_train = ["CCO", "CC(C)O", "CCCO", "CC(C)CO"]  # Example SMILES
labels_train = [1.5, 1.8, 1.2, 1.6]  # Example labels

train_dataset = SMILES_Dataset(smiles_train, labels_train, tokenizer, max_length=100)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop (similar to previous examples)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):
    model.train()
    for batch_smiles, batch_labels in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_smiles)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
```

**Key Design Choices:**

1. **Multiple Filter Sizes**: Capture patterns of different lengths (3=short, 5=medium, 7=long functional groups)
2. **Embedding Layer**: Learn distributed representations of SMILES characters
3. **Max Pooling**: Extract most important features regardless of position
4. **Concatenation**: Combine features from different filter sizes

### 4.2 2D CNNs for Molecular Images

Molecules can be represented as 2D images (chemical structure diagrams) or as heatmaps of molecular properties.

**Image Representation Approaches:**

1. **Chemical Structure Diagrams**: Rendered 2D molecular structures
2. **3D Conformer Projections**: 2D projections of 3D molecular conformations
3. **Property Heatmaps**: Grids showing electrostatic potential, electron density, etc.

**Implementation:**

```python
import torch
import torch.nn as nn
import torchvision.models as models

class Molecular2DCNN(nn.Module):
    """
    2D CNN for molecular images
    """
    def __init__(self, num_classes=1, pretrained=False):
        """
        Args:
            num_classes: Number of output classes/values
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(Molecular2DCNN, self).__init__()
        
        # Option 1: Custom CNN architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),  # Assuming 224x224 input
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output predictions of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class MolecularResNet(nn.Module):
    """
    ResNet-based model for molecular images (transfer learning)
    """
    def __init__(self, num_classes=1, pretrained=True):
        super(MolecularResNet, self).__init__()
        
        # Load pretrained ResNet
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Generate molecular images from SMILES
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import numpy as np

def smiles_to_image(smiles, size=(224, 224)):
    """
    Convert SMILES to image
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
    
    Returns:
        numpy array of shape (3, height, width)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return blank image if invalid SMILES
        return np.zeros((3, size[1], size[0]))
    
    # Draw molecule
    img = Draw.MolToImage(mol, size=size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Convert to (C, H, W) format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3)
    else:  # RGB
        img_array = img_array.transpose(2, 0, 1)
    
    return img_array / 255.0  # Normalize to [0, 1]

class MolecularImageDataset(Dataset):
    """
    Dataset for molecular images
    """
    def __init__(self, smiles_list, labels, transform=None, size=(224, 224)):
        self.images = []
        self.labels = []
        
        for smiles, label in zip(smiles_list, labels):
            img = smiles_to_image(smiles, size)
            self.images.append(torch.FloatTensor(img))
            self.labels.append(label)
        
        self.labels = torch.FloatTensor(self.labels).reshape(-1, 1)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

### 4.3 When to Use Each Approach

**Comparison Table:**

| Approach | Best For | Pros | Cons |
|----------|----------|------|------|
| **1D CNN on SMILES** | - Sequence patterns<br>- Functional groups<br>- Large datasets | - Fast training<br>- No molecular rendering<br>- Handles variable length | - Limited spatial info<br>- SMILES representation bias |
| **2D CNN on Images** | - Spatial relationships<br>- Stereochemistry<br>- Visual patterns | - Captures 2D structure<br>- Transfer learning from ImageNet<br>- Interpretable | - Slow image generation<br>- Fixed size input<br>- Loss of 3D info |
| **Graph Neural Networks** | - Atom/bond relationships<br>- 3D structure<br>- Small molecules | - Natural molecular representation<br>- Permutation invariant<br>- Interpretable | - More complex implementation<br>- (Covered in Day 3) |

**Decision Guide:**

```
START
  ├─ Need fast training? → 1D CNN on SMILES
  ├─ Have molecular images? → 2D CNN
  ├─ 3D structure important? → GNN (Day 3)
  ├─ Sequence patterns important? → 1D CNN or RNN
  └─ Spatial relationships important? → 2D CNN or GNN
```

**Example Use Cases:**

**Use 1D CNN on SMILES:**
- Toxicity prediction (functional group patterns)
- Synthetic accessibility (molecular complexity)
- Quick property screening

**Use 2D CNN on Images:**
- Structure-activity relationship visualization
- Similarity search with visual features
- Transfer learning from chemical structure databases

**Hybrid Approach:**

```python
class HybridMolecularModel(nn.Module):
    """
    Combine 1D CNN (SMILES) and 2D CNN (images) for robust predictions
    """
    def __init__(self, vocab_size, embedding_dim):
        super(HybridMolecularModel, self).__init__()
        
        # 1D CNN branch for SMILES
        self.smiles_cnn = SMILES_CNN(vocab_size, embedding_dim)
        
        # 2D CNN branch for images
        self.image_cnn = Molecular2DCNN()
        
        # Fusion layer
        self.fusion = nn.Linear(2, 1)  # Combine predictions
    
    def forward(self, smiles, images):
        """
        Args:
            smiles: Encoded SMILES (batch_size, max_length)
            images: Molecular images (batch_size, 3, 224, 224)
        
        Returns:
            Combined predictions
        """
        smiles_pred = self.smiles_cnn(smiles)
        image_pred = self.image_cnn(images)
        
        # Concatenate and fuse
        combined = torch.cat([smiles_pred, image_pred], dim=1)
        final_pred = self.fusion(combined)
        
        return final_pred
```

---

## 5. Recurrent Neural Networks

Recurrent Neural Networks (RNNs) process sequential data by maintaining hidden states that capture information from previous time steps. They are ideal for SMILES strings where the order of tokens matters.

### 5.1 LSTM for SMILES Sequences

Long Short-Term Memory (LSTM) networks address the vanishing gradient problem in traditional RNNs through gating mechanisms.

**LSTM Architecture:**

```
Input → Embedding → LSTM layers → Final hidden state → Dense → Output
```

**Complete Implementation:**

```python
import torch
import torch.nn as nn

class SMILES_LSTM(nn.Module):
    """
    LSTM model for SMILES-based molecular property prediction
    """
    def __init__(self, vocab_size=50, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, output_dim=1, dropout_rate=0.3, bidirectional=True):
        """
        Args:
            vocab_size: Size of SMILES vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Size of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            output_dim: Output size (1 for regression)
            dropout_rate: Dropout between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(SMILES_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
        
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch_size, seq_length, hidden_dim * num_directions)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        
        # Fully connected layers
        output = self.fc(final_hidden)
        
        return output

# Create model
model = SMILES_LSTM(
    vocab_size=50,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    output_dim=1,
    dropout_rate=0.3,
    bidirectional=True
)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**LSTM Internals:**

```python
# Manual LSTM cell implementation for understanding
class LSTMCell(nn.Module):
    """
    Single LSTM cell showing internal gates
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        # Gates: forget, input, output, candidate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)  # Forget gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)  # Input gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # Output gate
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate
    
    def forward(self, x_t, h_prev, c_prev):
        """
        Args:
            x_t: Input at time t (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
        
        Returns:
            h_t: New hidden state
            c_t: New cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Forget gate: decides what to forget from cell state
        f_t = torch.sigmoid(self.W_f(combined))
        
        # Input gate: decides what new information to store
        i_t = torch.sigmoid(self.W_i(combined))
        
        # Candidate: new candidate values for cell state
        c_tilde = torch.tanh(self.W_c(combined))
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Output gate: decides what to output from cell state
        o_t = torch.sigmoid(self.W_o(combined))
        
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
```

**Attention Mechanism for LSTM:**

```python
class SMILES_LSTM_Attention(nn.Module):
    """
    LSTM with attention mechanism for SMILES
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SMILES_LSTM_Attention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        # Embedding and LSTM
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Output
        output = self.fc(context)
        
        return output, attention_weights
```

### 5.2 GRU Alternative

Gated Recurrent Unit (GRU) is a simpler alternative to LSTM with fewer parameters.

**GRU vs LSTM:**

| Feature | LSTM | GRU |
|---------|------|-----|
| **Gates** | 3 (input, forget, output) | 2 (reset, update) |
| **Cell state** | Separate (c_t and h_t) | Unified (h_t only) |
| **Parameters** | More | Fewer (~25% less) |
| **Training speed** | Slower | Faster |
| **Performance** | Slightly better on complex tasks | Comparable on most tasks |
| **Memory** | Higher | Lower |

**GRU Implementation:**

```python
class SMILES_GRU(nn.Module):
    """
    GRU model for SMILES-based molecular property prediction
    """
    def __init__(self, vocab_size=50, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, output_dim=1, dropout_rate=0.3, bidirectional=True):
        super(SMILES_GRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU layers (API similar to LSTM)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Use final hidden state
        if self.gru.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        
        output = self.fc(final_hidden)
        return output
```

**GRU Internals:**

```python
class GRUCell(nn.Module):
    """
    Single GRU cell for understanding
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        
        # Gates: reset and update
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # Reset gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate
    
    def forward(self, x_t, h_prev):
        """
        Args:
            x_t: Input at time t
            h_prev: Previous hidden state
        
        Returns:
            h_t: New hidden state
        """
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # Reset gate: decides how much past to forget
        r_t = torch.sigmoid(self.W_r(combined))
        
        # Update gate: decides how much to update
        z_t = torch.sigmoid(self.W_z(combined))
        
        # Candidate: new candidate hidden state
        combined_reset = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        
        # Final hidden state: interpolation between prev and candidate
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
```

### 5.3 Comparison with CNNs

**Performance Comparison:**

```python
def compare_cnn_lstm_gru(smiles_train, y_train, smiles_val, y_val):
    """
    Compare CNN, LSTM, and GRU on same dataset
    """
    tokenizer = SMILESTokenizer()
    
    # Prepare data
    train_dataset = SMILES_Dataset(smiles_train, y_train, tokenizer)
    val_dataset = SMILES_Dataset(smiles_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    results = {}
    
    # Models to compare
    models = {
        '1D CNN': SMILES_CNN(vocab_size=len(tokenizer.vocab)),
        'LSTM': SMILES_LSTM(vocab_size=len(tokenizer.vocab)),
        'GRU': SMILES_GRU(vocab_size=len(tokenizer.vocab)),
        'LSTM+Attention': SMILES_LSTM_Attention(vocab_size=len(tokenizer.vocab))
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)
        
        # Evaluate
        val_metrics = evaluate_model(model, val_loader, criterion)
        
        results[name] = {
            'RMSE': val_metrics['rmse'],
            'R2': val_metrics['r2'],
            'Parameters': sum(p.numel() for p in model.parameters()),
            'Training time': val_metrics['time']
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'RMSE':<10} {'R2':<10} {'Parameters':<15} {'Time (s)':<10}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['RMSE']:<10.4f} {metrics['R2']:<10.4f} "
              f"{metrics['Parameters']:<15,} {metrics['Training time']:<10.1f}")
    
    return results
```

**Typical Results:**

| Model | RMSE | R² | Parameters | Training Time | Inference Speed |
|-------|------|-----|------------|---------------|-----------------|
| 1D CNN | 0.72 | 0.85 | 1.2M | Fast (1x) | Very Fast |
| LSTM | 0.68 | 0.87 | 2.5M | Slow (3x) | Slow |
| GRU | 0.69 | 0.86 | 1.9M | Medium (2x) | Medium |
| LSTM+Attention | 0.65 | 0.89 | 2.8M | Slowest (3.5x) | Slow |

**Recommendations:**

- **Use CNN** when: Speed is priority, local patterns important, large datasets
- **Use LSTM** when: Long-range dependencies matter, sequential information crucial
- **Use GRU** when: Want RNN benefits with fewer parameters, faster training
- **Use LSTM+Attention** when: Need interpretability, best performance, sufficient data

---

## 6. Transfer Learning

Transfer learning leverages knowledge learned from one task (usually on large datasets) to improve performance on another related task (often with limited data).

### 6.1 Why Transfer Learning for Molecules

**Challenges in Molecular Machine Learning:**

1. **Limited Labeled Data**: Experimental measurements are expensive
   - Example: Only ~10K molecules with measured BBB permeability
   - Contrast with millions of images in ImageNet

2. **Data Imbalance**: Some properties measured more than others
   - Solubility: ~100K datapoints
   - Metabolic stability: ~1K datapoints

3. **Related Tasks**: Many molecular properties share underlying features
   - Lipophilicity and permeability both depend on polarity
   - Multiple ADMET properties relate to molecular shape

**Benefits of Transfer Learning:**

- **Data Efficiency**: Achieve good performance with 10-100x less labeled data
- **Faster Convergence**: Pre-trained models converge in fewer epochs
- **Better Generalization**: Pre-learned features capture general molecular patterns
- **Domain Adaptation**: Adapt models trained on one molecule type to another

**Example Scenario:**

```
Problem: Predict BBB permeability (only 5,000 labeled molecules)

Solution 1 (From Scratch):
├─ Train model on 5,000 BBB molecules
├─ Performance: R² = 0.65
└─ Training time: 50 epochs

Solution 2 (Transfer Learning):
├─ Pre-train on 1M molecules (solubility + LogP + toxicity)
├─ Fine-tune on 5,000 BBB molecules
├─ Performance: R² = 0.82 (+26%)
└─ Training time: 10 epochs (5x faster)
```

### 6.2 Pre-Training Strategies

**1. Self-Supervised Pre-training with Autoencoders**

Learn molecular representations by reconstructing input:

```python
class MolecularAutoencoder(nn.Module):
    """
    Autoencoder for learning molecular representations
    """
    def __init__(self, input_dim=2048, encoding_dim=256):
        super(MolecularAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # Output in [0, 1] for fingerprints
        )
    
    def forward(self, x):
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction
    
    def encode(self, x):
        """Extract learned representations"""
        return self.encoder(x)

# Pre-training on large unlabeled dataset
def pretrain_autoencoder(smiles_list, num_epochs=100):
    """
    Pre-train autoencoder on large molecular dataset
    """
    # Generate fingerprints
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fingerprints.append(np.array(fp))
    
    fingerprints = torch.FloatTensor(np.array(fingerprints))
    dataset = TensorDataset(fingerprints, fingerprints)  # Input = target
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Create and train autoencoder
    autoencoder = MolecularAutoencoder(input_dim=2048, encoding_dim=256)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstruction = autoencoder(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return autoencoder

# Use pre-trained encoder for downstream task
class PretrainedMolecularModel(nn.Module):
    """
    Use pre-trained encoder for property prediction
    """
    def __init__(self, pretrained_encoder, output_dim=1, freeze_encoder=False):
        super(PretrainedMolecularModel, self).__init__()
        
        self.encoder = pretrained_encoder.encoder
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Add prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        predictions = self.prediction_head(features)
        return predictions
```

**2. Multi-Task Pre-training**

Train on multiple related properties simultaneously:

```python
def pretrain_multitask(smiles_list, properties_dict, task_configs):
    """
    Pre-train on multiple molecular properties
    
    Args:
        smiles_list: List of SMILES strings
        properties_dict: Dict of {property_name: values_array}
        task_configs: List of task configurations
    
    Returns:
        Pre-trained model
    """
    # Create multi-task model
    model = MultiTaskMolecularModel(
        input_dim=2048,
        shared_dims=[512, 256],
        task_configs=task_configs
    )
    
    # Prepare dataset
    fingerprints = []
    labels = {task['name']: [] for task in task_configs}
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fingerprints.append(np.array(fp))
            
            for task in task_configs:
                task_name = task['name']
                labels[task_name].append(properties_dict[task_name])
    
    # Train multi-task model
    # (Training code similar to Section 3)
    # ...
    
    return model
```

**3. Masked Language Model (for SMILES)**

Similar to BERT, mask random tokens and predict them:

```python
class MaskedSMILESModel(nn.Module):
    """
    BERT-like masked language model for SMILES
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=6):
        super(MaskedSMILESModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head for masked tokens
        self.mlm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        encoded = self.transformer(embedded, src_key_padding_mask=mask)
        predictions = self.mlm_head(encoded)
        return predictions

def create_masked_data(smiles_list, tokenizer, mask_prob=0.15):
    """
    Create masked SMILES for pre-training
    
    Args:
        smiles_list: List of SMILES strings
        tokenizer: SMILES tokenizer
        mask_prob: Probability of masking each token
    
    Returns:
        masked_inputs, targets
    """
    masked_inputs = []
    targets = []
    
    for smiles in smiles_list:
        encoded = tokenizer.encode(smiles)
        masked = encoded.copy()
        
        for i in range(len(encoded)):
            if np.random.random() < mask_prob:
                masked[i] = tokenizer.token_to_idx['<MASK>']
        
        masked_inputs.append(masked)
        targets.append(encoded)
    
    return torch.LongTensor(masked_inputs), torch.LongTensor(targets)
```

### 6.3 Fine-Tuning Workflow

**Complete Fine-Tuning Pipeline:**

```python
def fine_tune_pretrained_model(pretrained_model, smiles_train, y_train, 
                               smiles_val, y_val, freeze_layers=True):
    """
    Fine-tune pre-trained model on downstream task
    
    Args:
        pretrained_model: Pre-trained neural network
        smiles_train, y_train: Training data for downstream task
        smiles_val, y_val: Validation data
        freeze_layers: Whether to freeze early layers
    
    Returns:
        fine_tuned_model, history
    """
    # Create model with pre-trained weights
    model = PretrainedMolecularModel(
        pretrained_encoder=pretrained_model,
        output_dim=1,
        freeze_encoder=freeze_layers
    )
    
    # Prepare data
    train_dataset = MolecularDataset(smiles_train, y_train)
    val_dataset = MolecularDataset(smiles_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 10x smaller than from-scratch
    criterion = nn.MSELoss()
    
    # Training configuration
    num_epochs = 30  # Fewer epochs needed
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
                all_preds.extend(predictions.numpy())
                all_labels.extend(batch_y.numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, RMSE={val_rmse:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_finetuned_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_finetuned_model.pth'))
    
    return model, history

# Gradual unfreezing strategy
def gradual_unfreezing(model, train_loader, val_loader, num_phases=3):
    """
    Gradually unfreeze layers during fine-tuning
    
    Phase 1: Freeze all, train head only
    Phase 2: Unfreeze top encoder layers
    Phase 3: Unfreeze all layers
    """
    all_layers = list(model.encoder.children())
    criterion = nn.MSELoss()
    
    for phase in range(num_phases):
        print(f"\nPhase {phase+1}/{num_phases}")
        
        # Determine which layers to unfreeze
        if phase == 0:
            # Freeze all encoder layers
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif phase == 1:
            # Unfreeze last 1/3 of encoder layers
            unfreeze_from = len(all_layers) * 2 // 3
            for i, layer in enumerate(all_layers):
                if i >= unfreeze_from:
                    for param in layer.parameters():
                        param.requires_grad = True
        else:
            # Unfreeze all layers
            for param in model.encoder.parameters():
                param.requires_grad = True
        
        # Use decreasing learning rate for each phase
        lr = 1e-3 / (10 ** phase)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # Train for this phase
        for epoch in range(10):
            train_epoch(model, train_loader, criterion, optimizer)
    
    return model
```

### 6.4 Pre-Trained Models (ChemBERTa)

**Using ChemBERTa** (Chemical BERT Architecture):

```python
from transformers import AutoTokenizer, AutoModel
import torch

class ChemBERTaFineTuner(nn.Module):
    """
    Fine-tune ChemBERTa for molecular property prediction
    """
    def __init__(self, output_dim=1, dropout_rate=0.3):
        super(ChemBERTaFineTuner, self).__init__()
        
        # Load pre-trained ChemBERTa
        self.chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        
        # Add prediction head
        hidden_size = self.chemberta.config.hidden_size
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tokenized SMILES (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            Predictions (batch_size, output_dim)
        """
        # Get ChemBERTa embeddings
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Prediction
        predictions = self.prediction_head(cls_embedding)
        
        return predictions

# Usage example
def use_chemberta(smiles_list, labels):
    """
    Fine-tune ChemBERTa on your dataset
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    # Tokenize SMILES
    encoded = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Create model
    model = ChemBERTaFineTuner(output_dim=1)
    
    # Training loop
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        predictions = model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
        
        loss = criterion(predictions.squeeze(), labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return model
```

### 6.5 Results and Comparisons

**Comparison Study:**

```python
def compare_transfer_learning_strategies(smiles_train, y_train, smiles_val, y_val):
    """
    Compare different transfer learning approaches
    """
    results = {}
    
    # Strategy 1: Train from scratch (baseline)
    print("\n1. Training from scratch...")
    scratch_model = MolecularFNN(input_dim=2048)
    scratch_history = train_model(scratch_model, smiles_train, y_train, smiles_val, y_val)
    results['From Scratch'] = evaluate_model(scratch_model, smiles_val, y_val)
    
    # Strategy 2: Pre-trained autoencoder
    print("\n2. Pre-trained autoencoder + fine-tuning...")
    # Pre-train on large unlabeled dataset (e.g., ZINC database)
    large_smiles = load_large_dataset()  # Assume 1M molecules
    autoencoder = pretrain_autoencoder(large_smiles, num_epochs=100)
    
    transfer_model = PretrainedMolecularModel(autoencoder, freeze_encoder=True)
    transfer_history = fine_tune_pretrained_model(transfer_model, smiles_train, y_train, 
                                                   smiles_val, y_val)
    results['Autoencoder Transfer'] = evaluate_model(transfer_model, smiles_val, y_val)
    
    # Strategy 3: Multi-task pre-training
    print("\n3. Multi-task pre-training + fine-tuning...")
    multitask_model = pretrain_multitask(large_smiles, properties_dict, task_configs)
    multitask_transfer = fine_tune_pretrained_model(multitask_model, smiles_train, y_train,
                                                    smiles_val, y_val)
    results['Multi-Task Transfer'] = evaluate_model(multitask_transfer, smiles_val, y_val)
    
    # Strategy 4: ChemBERTa
    print("\n4. ChemBERTa fine-tuning...")
    chemberta_model = ChemBERTaFineTuner()
    chemberta_history = fine_tune_chemberta(chemberta_model, smiles_train, y_train,
                                           smiles_val, y_val)
    results['ChemBERTa'] = evaluate_model(chemberta_model, smiles_val, y_val)
    
    # Print comparison
    print("\n" + "="*70)
    print("TRANSFER LEARNING COMPARISON")
    print("="*70)
    print(f"{'Strategy':<30} {'RMSE':<10} {'R²':<10} {'Training Time':<15}")
    print("-"*70)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<30} {metrics['RMSE']:<10.4f} {metrics['R²']:<10.4f} "
              f"{metrics['time']:<15.1f}s")
    
    return results
```

**Expected Results (BBB Permeability Example):**

| Strategy | RMSE | R² | Data Required | Training Time | Improvement |
|----------|------|-----|---------------|---------------|-------------|
| From Scratch | 0.95 | 0.68 | 5,000 | 120 min | Baseline |
| Autoencoder Transfer | 0.78 | 0.79 | 5,000 | 45 min | +16% |
| Multi-Task Transfer | 0.72 | 0.82 | 5,000 | 35 min | +20% |
| ChemBERTa | 0.65 | 0.87 | 5,000 | 25 min | +28% |
| ChemBERTa (Low Data) | 0.82 | 0.75 | 500 | 15 min | Still viable |

**Key Insights:**

1. Transfer learning provides 15-30% improvement in performance
2. Training time reduced by 60-80%
3. Most effective when target task has limited data (<10K samples)
4. ChemBERTa performs best due to massive pre-training on 77M molecules
5. Multi-task pre-training excellent when related properties available

---

## 7. Complete Practical Exercise

### 7.1 Problem: BBB Permeability Prediction

**Background:**

Blood-Brain Barrier (BBB) permeability is crucial for CNS drugs. We'll predict log(BB), the logarithm of the brain-to-blood concentration ratio.

**Dataset:**
- Training: 1,500 molecules
- Validation: 300 molecules
- Test: 200 molecules
- Features: SMILES strings
- Target: log(BB) values (continuous, range: -3 to 2)

### 7.2 Full Pipeline

```python
# ============================================================================
# COMPLETE BBB PERMEABILITY PREDICTION PIPELINE
# ============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_bbb_data(filepath='bbb_permeability.csv'):
    """
    Load BBB permeability dataset
    
    Expected columns: SMILES, logBB
    """
    try:
        df = pd.read_csv(filepath)
    except:
        # Generate synthetic data for demonstration
        print("Generating synthetic BBB data...")
        df = generate_synthetic_bbb_data(2000)
    
    # Remove invalid SMILES
    valid_indices = []
    for idx, smiles in enumerate(df['SMILES']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_indices.append(idx)
    
    df = df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"Loaded {len(df)} valid molecules")
    print(f"logBB range: [{df['logBB'].min():.2f}, {df['logBB'].max():.2f}]")
    
    return df

def generate_synthetic_bbb_data(n_samples=2000):
    """
    Generate synthetic BBB data for demonstration
    """
    # Common drug-like SMILES templates
    templates = [
        "CCO", "CC(C)O", "CCCO", "CC(C)CO", "CCCCO",
        "c1ccccc1", "c1ccccc1C", "c1ccccc1O", "c1ccccc1N",
        "CC(=O)O", "CC(=O)N", "CCNC", "CCN(C)C",
        "c1ccc(cc1)C(=O)O", "c1ccc(cc1)N"
    ]
    
    smiles_list = []
    logbb_list = []
    
    for _ in range(n_samples):
        # Random combination of templates
        smiles = np.random.choice(templates)
        
        # Calculate simple features for synthetic logBB
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # Synthetic logBB based on known correlations
            logbb = 0.1 * logp - 0.01 * tpsa - 0.003 * mw + np.random.normal(0, 0.3)
            logbb = np.clip(logbb, -3, 2)
            
            smiles_list.append(smiles)
            logbb_list.append(logbb)
    
    df = pd.DataFrame({'SMILES': smiles_list, 'logBB': logbb_list})
    return df

# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

def compute_molecular_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    Compute Morgan fingerprints for molecules
    """
    fingerprints = []
    
    for smiles in tqdm(smiles_list, desc="Computing fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(np.array(fp))
        else:
            fingerprints.append(np.zeros(n_bits))
    
    return np.array(fingerprints)

def compute_molecular_descriptors(smiles_list):
    """
    Compute RDKit molecular descriptors
    """
    descriptors_list = []
    
    descriptor_functions = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.TPSA,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
        Descriptors.FractionCsp3
    ]
    
    for smiles in tqdm(smiles_list, desc="Computing descriptors"):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = [func(mol) for func in descriptor_functions]
            descriptors_list.append(desc)
        else:
            descriptors_list.append([0] * len(descriptor_functions))
    
    return np.array(descriptors_list)

# ============================================================================
# 3. DEEP LEARNING MODEL
# ============================================================================

class BBBPermeabilityModel(nn.Module):
    """
    Neural network for BBB permeability prediction
    """
    def __init__(self, input_dim=2048):
        super(BBBPermeabilityModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class BBBDataset(Dataset):
    """Dataset for BBB permeability"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_deep_learning_model(X_train, y_train, X_val, y_val, 
                              num_epochs=100, batch_size=64, lr=0.001):
    """
    Train deep learning model for BBB permeability
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = BBBDataset(X_train, y_train)
    val_dataset = BBBDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BBBPermeabilityModel(input_dim=X_train.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_rmse': [], 'val_mae': [], 'val_r2': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        val_mae = mean_absolute_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, "
                  f"MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_bbb_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(torch.load('best_bbb_model.pth'))
    
    return model, history, training_time

# ============================================================================
# 5. RANDOM FOREST BASELINE
# ============================================================================

def train_random_forest_model(X_train, y_train, X_val, y_val):
    """
    Train Random Forest baseline model
    """
    start_time = time.time()
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    train_pred = rf_model.predict(X_train)
    val_pred = rf_model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    training_time = time.time() - start_time
    
    print("\nRandom Forest Results:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")
    print(f"  Val MAE: {val_mae:.4f}")
    print(f"  Val R²: {val_r2:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return rf_model, {
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'time': training_time
    }

# ============================================================================
# 6. EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_type='dl'):
    """
    Comprehensive model evaluation
    """
    if model_type == 'dl':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        test_dataset = BBBDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                predictions.extend(pred.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
    else:
        predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nTest Set Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}

def visualize_results(y_true, y_pred_dl, y_pred_rf, history):
    """
    Create comprehensive visualization of results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Training history
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Metrics evolution
    axes[0, 1].plot(history['val_rmse'], label='RMSE', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].grid(True)
    
    ax2 = axes[0, 1].twinx()
    ax2.plot(history['val_r2'], label='R²', color='blue')
    ax2.set_ylabel('R²')
    
    # 3. R² evolution
    axes[0, 2].plot(history['val_r2'], color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].set_title('Validation R²')
    axes[0, 2].grid(True)
    
    # 4. DL predictions scatter plot
    axes[1, 0].scatter(y_true, y_pred_dl, alpha=0.5)
    axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2)
    axes[1, 0].set_xlabel('True logBB')
    axes[1, 0].set_ylabel('Predicted logBB')
    axes[1, 0].set_title('Deep Learning Predictions')
    axes[1, 0].grid(True)
    
    # 5. RF predictions scatter plot
    axes[1, 1].scatter(y_true, y_pred_rf, alpha=0.5, color='orange')
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                     'r--', lw=2)
    axes[1, 1].set_xlabel('True logBB')
    axes[1, 1].set_ylabel('Predicted logBB')
    axes[1, 1].set_title('Random Forest Predictions')
    axes[1, 1].grid(True)
    
    # 6. Residuals comparison
    residuals_dl = y_true - y_pred_dl
    residuals_rf = y_true - y_pred_rf
    
    axes[1, 2].hist(residuals_dl, bins=30, alpha=0.5, label='Deep Learning')
    axes[1, 2].hist(residuals_rf, bins=30, alpha=0.5, label='Random Forest')
    axes[1, 2].set_xlabel('Residuals')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Residuals Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('bbb_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("="*70)
    print("BBB PERMEABILITY PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_bbb_data()
    
    # 2. Split data
    print("\n2. Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"  Train: {len(train_df)} molecules")
    print(f"  Val: {len(val_df)} molecules")
    print(f"  Test: {len(test_df)} molecules")
    
    # 3. Extract features
    print("\n3. Extracting features...")
    X_train = compute_molecular_fingerprints(train_df['SMILES'].values)
    X_val = compute_molecular_fingerprints(val_df['SMILES'].values)
    X_test = compute_molecular_fingerprints(test_df['SMILES'].values)
    
    y_train = train_df['logBB'].values
    y_val = val_df['logBB'].values
    y_test = test_df['logBB'].values
    
    # 4. Train Deep Learning model
    print("\n4. Training Deep Learning model...")
    dl_model, history, dl_time = train_deep_learning_model(
        X_train, y_train, X_val, y_val,
        num_epochs=100, batch_size=64, lr=0.001
    )
    print(f"  Training time: {dl_time:.2f}s")
    
    # 5. Train Random Forest baseline
    print("\n5. Training Random Forest baseline...")
    rf_model, rf_results = train_random_forest_model(X_train, y_train, X_val, y_val)
    
    # 6. Evaluate on test set
    print("\n6. Evaluating models on test set...")
    
    print("\nDeep Learning Model:")
    dl_predictions, dl_metrics = evaluate_model(dl_model, X_test, y_test, 'dl')
    
    print("\nRandom Forest Model:")
    rf_predictions, rf_metrics = evaluate_model(rf_model, X_test, y_test, 'rf')
    
    # 7. Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Metric':<15} {'Deep Learning':<20} {'Random Forest':<20} {'Improvement'}")
    print("-"*70)
    print(f"{'RMSE':<15} {dl_metrics['rmse']:<20.4f} {rf_metrics['rmse']:<20.4f} "
          f"{(rf_metrics['rmse']-dl_metrics['rmse'])/rf_metrics['rmse']*100:>6.1f}%")
    print(f"{'MAE':<15} {dl_metrics['mae']:<20.4f} {rf_metrics['mae']:<20.4f} "
          f"{(rf_metrics['mae']-dl_metrics['mae'])/rf_metrics['mae']*100:>6.1f}%")
    print(f"{'R²':<15} {dl_metrics['r2']:<20.4f} {rf_metrics['r2']:<20.4f} "
          f"{(dl_metrics['r2']-rf_metrics['r2'])/rf_metrics['r2']*100:>6.1f}%")
    print(f"{'Training Time':<15} {dl_time:<20.1f}s {rf_results['time']:<20.1f}s")
    
    # 8. Visualize results
    print("\n8. Generating visualizations...")
    visualize_results(y_test, dl_predictions, rf_predictions, history)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    return {
        'dl_model': dl_model,
        'rf_model': rf_model,
        'dl_metrics': dl_metrics,
        'rf_metrics': rf_metrics,
        'history': history
    }

# Run the pipeline
if __name__ == "__main__":
    results = main()
```

### 7.3 Expected Output

```
======================================================================
BBB PERMEABILITY PREDICTION - COMPLETE PIPELINE
======================================================================

1. Loading data...
Loaded 2000 valid molecules
logBB range: [-2.85, 1.92]

2. Splitting data...
  Train: 1400 molecules
  Val: 300 molecules
  Test: 300 molecules

3. Extracting features...
Computing fingerprints: 100%|██████████| 1400/1400 [00:05<00:00]
Computing fingerprints: 100%|██████████| 300/300 [00:01<00:00]
Computing fingerprints: 100%|██████████| 300/300 [00:01<00:00]

4. Training Deep Learning model...
Using device: cuda
Epoch 10/100
  Train Loss: 0.3245
  Val Loss: 0.3678, RMSE: 0.6065, MAE: 0.4532, R²: 0.7234

Epoch 20/100
  Train Loss: 0.2156
  Val Loss: 0.2987, RMSE: 0.5465, MAE: 0.4012, R²: 0.7789

... (training continues)

Early stopping at epoch 68
  Training time: 142.35s

5. Training Random Forest baseline...

Random Forest Results:
  Train RMSE: 0.1234
  Val RMSE: 0.6234
  Val MAE: 0.4689
  Val R²: 0.7456
  Training time: 56.78s

6. Evaluating models on test set...

Deep Learning Model:
Test Set Evaluation:
  RMSE: 0.5234
  MAE: 0.3876
  R²: 0.7923

Random Forest Model:
Test Set Evaluation:
  RMSE: 0.5987
  MAE: 0.4456
  R²: 0.7534

======================================================================
FINAL COMPARISON
======================================================================
Metric          Deep Learning        Random Forest        Improvement
----------------------------------------------------------------------
RMSE            0.5234               0.5987                12.6%
MAE             0.3876               0.4456                13.0%
R²              0.7923               0.7534                 5.2%
Training Time   142.4s               56.8s                

8. Generating visualizations...

======================================================================
PIPELINE COMPLETE!
======================================================================
```

---

## 8. Model Interpretation

Understanding what your model has learned is crucial for trust, debugging, and scientific insight.

### 8.1 Gradient-Based Importance

Calculate feature importance by examining gradients:

```python
def compute_gradient_importance(model, X, device='cuda'):
    """
    Compute feature importance using gradients
    
    Args:
        model: Trained neural network
        X: Input features (numpy array or tensor)
    
    Returns:
        importance_scores: Feature importance for each sample
    """
    model.eval()
    
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    
    X = X.to(device)
    X.requires_grad = True
    
    # Forward pass
    output = model(X)
    
    # Compute gradients
    gradients = []
    for i in range(output.shape[0]):
        model.zero_grad()
        if X.grad is not None:
            X.grad.zero_()
        
        output[i].backward(retain_graph=True)
        gradients.append(X.grad[i].cpu().detach().numpy().copy())
    
    gradients = np.array(gradients)
    
    # Importance = absolute gradient * input value
    importance = np.abs(gradients) * X.cpu().detach().numpy()
    
    return importance

def visualize_feature_importance(importance, feature_names=None, top_k=20):
    """
    Visualize feature importance
    """
    # Average importance across samples
    avg_importance = np.mean(importance, axis=0)
    
    # Get top-k features
    top_indices = np.argsort(avg_importance)[-top_k:][::-1]
    top_importance = avg_importance[top_indices]
    
    if feature_names is not None:
        top_features = [feature_names[i] for i in top_indices]
    else:
        top_features = [f"Feature {i}" for i in top_indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), top_importance)
    plt.yticks(range(top_k), top_features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

# Usage
importance_scores = compute_gradient_importance(model, X_test)
visualize_feature_importance(importance_scores, top_k=20)
```

### 8.2 Integrated Gradients

More accurate attribution method that accounts for baseline:

```python
def integrated_gradients(model, X, baseline=None, steps=50, device='cuda'):
    """
    Compute integrated gradients for feature attribution
    
    Args:
        model: Trained neural network
        X: Input features
        baseline: Baseline input (default: zero)
        steps: Number of integration steps
    
    Returns:
        attributions: Feature attributions
    """
    model.eval()
    
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    
    X = X.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(X)
    else:
        baseline = torch.FloatTensor(baseline).to(device)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps).to(device)
    interpolated_inputs = []
    
    for alpha in alphas:
        interpolated = baseline + alpha * (X - baseline)
        interpolated_inputs.append(interpolated)
    
    interpolated_inputs = torch.stack(interpolated_inputs)
    
    # Compute gradients for each interpolated input
    gradients = []
    
    for interp_input in interpolated_inputs:
        interp_input.requires_grad = True
        output = model(interp_input)
        
        model.zero_grad()
        output.sum().backward()
        
        gradients.append(interp_input.grad.cpu().detach())
    
    gradients = torch.stack(gradients)
    
    # Average gradients and multiply by input difference
    avg_gradients = torch.mean(gradients, dim=0)
    attributions = (X.cpu() - baseline.cpu()) * avg_gradients
    
    return attributions.numpy()

def explain_prediction(model, smiles, tokenizer, importance_method='integrated_gradients'):
    """
    Explain prediction for a single molecule
    """
    # Encode SMILES
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    X = np.array(fp).reshape(1, -1)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(torch.FloatTensor(X)).item()
    
    # Get importance
    if importance_method == 'integrated_gradients':
        importance = integrated_gradients(model, X)
    else:
        importance = compute_gradient_importance(model, X)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Draw molecule
    img = Draw.MolToImage(mol, size=(400, 400))
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Predicted logBB: {prediction:.2f}')
    
    # Plot importance
    top_k = 50
    top_indices = np.argsort(np.abs(importance[0]))[-top_k:]
    ax2.barh(range(top_k), importance[0, top_indices])
    ax2.set_xlabel('Attribution Score')
    ax2.set_ylabel('Fingerprint Bit')
    ax2.set_title('Feature Attributions (Integrated Gradients)')
    
    plt.tight_layout()
    plt.savefig(f'explanation_{smiles[:10]}.png', dpi=150)
    plt.show()
    
    return prediction, importance
```

### 8.3 Activation Maximization

Find input patterns that maximally activate specific neurons:

```python
def activation_maximization(model, layer_name, neuron_idx, 
                           input_shape=(1, 2048), iterations=1000, lr=0.1):
    """
    Find input that maximally activates a specific neuron
    
    Args:
        model: Neural network
        layer_name: Name of layer to analyze
        neuron_idx: Index of neuron in that layer
        input_shape: Shape of input
        iterations: Number of optimization steps
        lr: Learning rate
    
    Returns:
        optimal_input: Input that maximizes activation
    """
    model.eval()
    
    # Initialize random input
    optimal_input = torch.randn(input_shape, requires_grad=True)
    optimizer = optim.Adam([optimal_input], lr=lr)
    
    # Get layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))
            break
    
    # Optimize
    for i in range(iterations):
        optimizer.zero_grad()
        
        _ = model(optimal_input)
        
        # Loss = negative activation (we want to maximize)
        loss = -activation[layer_name][0, neuron_idx]
        
        loss.backward()
        optimizer.step()
        
        # Clip to valid range [0, 1] for fingerprints
        with torch.no_grad():
            optimal_input.clamp_(0, 1)
        
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}, Activation: {-loss.item():.4f}")
    
    return optimal_input.detach().numpy()

# Usage
optimal_fp = activation_maximization(
    model, 
    layer_name='network.4',  # Second hidden layer
    neuron_idx=42,
    iterations=1000
)

print(f"Optimal fingerprint pattern: {optimal_fp[0][:20]}...")
print(f"Number of active bits: {np.sum(optimal_fp > 0.5)}")
```

**Attention Visualization for LSTM:**

```python
def visualize_attention(model, smiles, tokenizer):
    """
    Visualize attention weights for LSTM model with attention
    """
    # Encode SMILES
    encoded = tokenizer.encode(smiles)
    X = torch.LongTensor([encoded])
    
    # Get prediction and attention weights
    model.eval()
    with torch.no_grad():
        prediction, attention_weights = model(X)
    
    # Plot attention
    attention = attention_weights[0].squeeze().numpy()
    tokens = tokenizer.tokenize(smiles)
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(tokens)), attention[:len(tokens)])
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.xlabel('SMILES Token')
    plt.ylabel('Attention Weight')
    plt.title(f'Attention Weights for: {smiles}\nPrediction: {prediction.item():.2f}')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()
```

---

## 9. Best Practices

### 9.1 Hyperparameter Tuning with Optuna

Automated hyperparameter optimization:

```python
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter optimization
    """
    # Suggest hyperparameters
    config = {
        'hidden_dim_1': trial.suggest_int('hidden_dim_1', 256, 1024, step=128),
        'hidden_dim_2': trial.suggest_int('hidden_dim_2', 128, 512, step=64),
        'hidden_dim_3': trial.suggest_int('hidden_dim_3', 64, 256, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    }
    
    # Create model
    model = MolecularFNN(
        input_dim=X_train.shape[1],
        hidden_dims=[config['hidden_dim_1'], config['hidden_dim_2'], config['hidden_dim_3']],
        output_dim=1,
        dropout_rate=config['dropout_rate']
    )
    
    # Create dataloaders
    train_dataset = BBBDataset(X_train, y_train)
    val_dataset = BBBDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Train
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    
    num_epochs = 50  # Reduced for faster tuning
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

# Run optimization
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

study.optimize(
    lambda trial: objective(trial, X_train, y_train, X_val, y_val),
    n_trials=50,
    timeout=7200  # 2 hours
)

# Print results
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print(f"\nBest validation loss: {study.best_value:.4f}")

# Visualize
fig1 = plot_optimization_history(study)
fig1.write_image('optimization_history.png')

fig2 = plot_param_importances(study)
fig2.write_image('param_importances.png')

# Train final model with best parameters
best_config = study.best_params
final_model = MolecularFNN(
    input_dim=X_train.shape[1],
    hidden_dims=[best_config['hidden_dim_1'], 
                best_config['hidden_dim_2'], 
                best_config['hidden_dim_3']],
    output_dim=1,
    dropout_rate=best_config['dropout_rate']
)
```

### 9.2 Complete Debugging Checklist

**Pre-Training Checks:**

```python
def pre_training_diagnostics(model, train_loader, device='cpu'):
    """
    Run diagnostics before training
    """
    print("="*70)
    print("PRE-TRAINING DIAGNOSTICS")
    print("="*70)
    
    model = model.to(device)
    
    # 1. Check data loading
    print("\n1. Data Loading Check:")
    try:
        batch_x, batch_y = next(iter(train_loader))
        print(f"  ✓ Batch shapes: X={batch_x.shape}, Y={batch_y.shape}")
        print(f"  ✓ X range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"  ✓ Y range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
        print(f"  ✓ No NaN in X: {not torch.isnan(batch_x).any()}")
        print(f"  ✓ No NaN in Y: {not torch.isnan(batch_y).any()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # 2. Check forward pass
    print("\n2. Forward Pass Check:")
    try:
        model.eval()
        with torch.no_grad():
            batch_x = batch_x.to(device)
            output = model(batch_x)
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  ✓ No NaN in output: {not torch.isnan(output).any()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # 3. Check backward pass
    print("\n3. Backward Pass Check:")
    try:
        model.train()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        
        print(f"  ✓ Loss value: {loss.item():.4f}")
        print(f"  ✓ Loss is finite: {torch.isfinite(loss)}")
        
        # Check gradients
        has_gradients = False
        max_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        print(f"  ✓ Gradients computed: {has_gradients}")
        print(f"  ✓ Max gradient: {max_grad:.6f}")
        
        optimizer.step()
        print(f"  ✓ Optimizer step successful")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    
    # 4. Check model parameters
    print("\n4. Model Parameters Check:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    
    # 5. Check learning rate
    print("\n5. Optimizer Check:")
    for param_group in optimizer.param_groups:
        print(f"  ✓ Learning rate: {param_group['lr']}")
    
    print("\n" + "="*70)
    print("ALL CHECKS PASSED - READY TO TRAIN")
    print("="*70)
    
    return True

# Run diagnostics
if pre_training_diagnostics(model, train_loader):
    # Start training
    train_model(...)
```

### 9.3 Common Issues and Solutions

**Comprehensive Troubleshooting Table:**

| Issue | Symptoms | Possible Causes | Solutions |
|-------|----------|-----------------|-----------|
| **Loss is NaN** | Loss becomes NaN during training | - Learning rate too high<br>- Gradient explosion<br>- Invalid inputs | - Reduce learning rate by 10x<br>- Add gradient clipping<br>- Check for NaN/Inf in data<br>- Use mixed precision training |
| **Loss not decreasing** | Loss stays constant or increases | - Learning rate too low<br>- Wrong loss function<br>- Data not normalized<br>- Model too simple | - Increase learning rate<br>- Verify loss function matches task<br>- Normalize inputs<br>- Increase model capacity |
| **Overfitting** | Train loss << Val loss | - Model too complex<br>- Too little data<br>- Insufficient regularization | - Add dropout (0.3-0.5)<br>- Add L2 regularization<br>- Use data augmentation<br>- Reduce model size<br>- Early stopping |
| **Underfitting** | Both losses high | - Model too simple<br>- Training time insufficient<br>- Learning rate issues | - Increase model capacity<br>- Train longer<br>- Tune learning rate<br>- Remove excessive regularization |
| **Slow training** | Training takes too long | - Batch size too small<br>- Model too large<br>- Inefficient data loading | - Increase batch size<br>- Use GPU<br>- Enable data loader workers<br>- Use mixed precision |
| **Unstable training** | Loss oscillates wildly | - Learning rate too high<br>- Batch size too small<br>- Poor initialization | - Reduce learning rate<br>- Increase batch size<br>- Use learning rate scheduler<br>- Use batch normalization |
| **Poor test performance** | Test worse than validation | - Data leakage<br>- Different distribution<br>- Overfitting to val set | - Check train/val/test splits<br>- Verify data preprocessing<br>- Use stratified splitting |
| **Gradient vanishing** | Gradients become very small | - Too many layers<br>- Wrong activation<br>- Poor initialization | - Use ReLU/Leaky ReLU<br>- Reduce number of layers<br>- Use skip connections<br>- Use batch normalization |
| **Out of memory** | CUDA out of memory | - Batch size too large<br>- Model too large<br>- Gradient accumulation needed | - Reduce batch size<br>- Use gradient checkpointing<br>- Use mixed precision<br>- Clear cache regularly |

**Debugging Code:**

```python
def debug_training_step(model, batch_x, batch_y, criterion, optimizer):
    """
    Detailed debugging of single training step
    """
    print("\n" + "="*70)
    print("DEBUGGING TRAINING STEP")
    print("="*70)
    
    # 1. Input check
    print("\n1. Input Check:")
    print(f"  X shape: {batch_x.shape}, dtype: {batch_x.dtype}")
    print(f"  Y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
    print(f"  X range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
    print(f"  Y range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
    print(f"  X has NaN: {torch.isnan(batch_x).any()}")
    print(f"  Y has NaN: {torch.isnan(batch_y).any()}")
    
    # 2. Forward pass
    print("\n2. Forward Pass:")
    model.train()
    optimizer.zero_grad()
    
    output = model(batch_x)
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output has NaN: {torch.isnan(output).any()}")
    
    # 3. Loss computation
    print("\n3. Loss Computation:")
    loss = criterion(output, batch_y)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss is finite: {torch.isfinite(loss)}")
    
    # 4. Backward pass
    print("\n4. Backward Pass:")
    loss.backward()
    
    # Check gradients
    print("\n  Gradient Statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()
            grad_mean = param.grad.mean().item()
            grad_norm = param.grad.norm().item()
            
            print(f"    {name}:")
            print(f"      Range: [{grad_min:.6f}, {grad_max:.6f}]")
            print(f"      Mean: {grad_mean:.6f}, Norm: {grad_norm:.6f}")
            print(f"      Has NaN: {torch.isnan(param.grad).any()}")
    
    # 5. Optimizer step
    print("\n5. Optimizer Step:")
    optimizer.step()
    print("  ✓ Step completed")
    
    # 6. Parameter update check
    print("\n6. Parameter Update Check:")
    new_output = model(batch_x)
    new_loss = criterion(new_output, batch_y)
    print(f"  New loss: {new_loss.item():.4f}")
    print(f"  Loss change: {new_loss.item() - loss.item():.6f}")
    
    print("\n" + "="*70)

# Usage: Run on first batch to debug
batch_x, batch_y = next(iter(train_loader))
debug_training_step(model, batch_x, batch_y, criterion, optimizer)
```

---

## 10. Key Takeaways

**Core Concepts:**

1. **Deep Learning Advantages for Molecules**
   - Automatic feature learning from raw molecular representations
   - Captures complex non-linear relationships
   - Enables transfer learning and multi-task learning
   - State-of-the-art performance on many molecular property prediction tasks

2. **Model Selection Guidelines**
   - **Feedforward NN**: Best for general-purpose molecular property prediction with fingerprints
   - **1D CNN**: Excellent for SMILES when local patterns (functional groups) matter
   - **LSTM/GRU**: Best for sequence modeling and when order matters in SMILES
   - **Multi-Task**: Use when predicting multiple related properties
   - **Transfer Learning**: Critical when data is limited (<5K samples)

3. **Architecture Best Practices**
   - Start simple (2-3 layers) and add complexity only if needed
   - Use ReLU activation for hidden layers
   - Apply batch normalization and dropout for regularization
   - Use Adam optimizer with learning rate 1e-3 as default
   - Implement early stopping (patience=15-20)

4. **Training Strategies**
   - Always split data into train/val/test (70/15/15)
   - Normalize inputs for faster convergence
   - Use learning rate scheduling (ReduceLROnPlateau)
   - Monitor multiple metrics (RMSE, MAE, R²)
   - Save best model based on validation loss

5. **Performance Optimization**
   - Deep learning typically provides 10-20% improvement over Random Forest
   - Transfer learning can improve performance by 15-30% with limited data
   - Multi-task learning helps when tasks are related
   - Ensemble methods (combining multiple models) can add another 5-10%

6. **Hyperparameter Importance Ranking**
   1. Learning rate (most critical)
   2. Batch size
   3. Number of layers and neurons
   4. Dropout rate
   5. Optimizer choice (Adam usually best)

7. **Common Pitfalls to Avoid**
   - Training without validation set
   - Not normalizing inputs
   - Ignoring overfitting signs
   - Using too large learning rate
   - Not checking for data leakage
   - Forgetting to set model.eval() during inference

8. **Model Interpretation Matters**
   - Use gradient-based methods for feature importance
   - Integrated gradients provide better attributions
   - Attention mechanisms add interpretability
   - Always validate interpretations with domain knowledge

9. **Production Considerations**
   - Save model architecture and weights separately
   - Version control for models and data
   - Monitor model performance degradation over time
   - Implement confidence scoring for predictions
   - Document training procedures and hyperparameters

10. **Research Directions**
    - Graph Neural Networks for molecular graphs (Day 3)
    - 3D conformation-based models
    - Active learning for efficient data collection
    - Uncertainty quantification
    - Explainable AI for drug discovery

**Performance Benchmarks (BBB Permeability):**

| Approach | R² Score | RMSE | Training Time | Data Required |
|----------|----------|------|---------------|---------------|
| Random Forest | 0.75 | 0.60 | Fast (1 min) | 1K+ |
| Feedforward NN | 0.79 | 0.52 | Medium (15 min) | 1K+ |
| 1D CNN (SMILES) | 0.82 | 0.48 | Medium (20 min) | 2K+ |
| LSTM (SMILES) | 0.84 | 0.45 | Slow (45 min) | 3K+ |
| Transfer Learning | 0.87 | 0.41 | Fast (10 min) | 500+ |
| Multi-Task | 0.86 | 0.42 | Medium (25 min) | 1K+ per task |

---

## 11. Resources

### 11.1 Essential Papers

**Deep Learning Fundamentals:**
1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553), 436-444.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**Molecular Deep Learning:**
3. Wu, Z., et al. (2018). "MoleculeNet: A benchmark for molecular machine learning." *Chemical Science*, 9(2), 513-530.
4. Wen, M., et al. (2020). "Deep learning for molecular property prediction." *arXiv preprint* arXiv:2003.03167.
5. Goh, G. B., et al. (2017). "Deep learning for computational chemistry." *Journal of Computational Chemistry*, 38(16), 1291-1307.

**Architecture-Specific:**
6. Gómez-Bombarelli, R., et al. (2018). "Automatic chemical design using a data-driven continuous representation of molecules." *ACS Central Science*, 4(2), 268-276.
7. Zheng, S., et al. (2020). "Identifying structure–property relationships through SMILES syntax analysis with self-attention mechanism." *Journal of Chemical Information and Modeling*, 59(2), 914-923.
8. Chithrananda, S., et al. (2020). "ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction." *arXiv preprint* arXiv:2010.09885.

**Transfer Learning:**
9. Hu, W., et al. (2020). "Strategies for pre-training graph neural networks." *ICLR 2020*.
10. Ramsundar, B., et al. (2015). "Massively multitask networks for drug discovery." *arXiv preprint* arXiv:1502.02072.

**Model Interpretation:**
11. Sundararajan, M., et al. (2017). "Axiomatic attribution for deep networks." *ICML 2017*.
12. Jiménez-Luna, J., et al. (2020). "Drug discovery with explainable artificial intelligence." *Nature Machine Intelligence*, 2(10), 573-584.

### 11.2 Software Libraries

**Deep Learning Frameworks:**
- **PyTorch**: https://pytorch.org/ (Recommended for research)
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **JAX**: https://github.com/google/jax (For advanced users)

**Molecular Machine Learning:**
- **DeepChem**: https://deepchem.io/ (Comprehensive molecular ML library)
- **RDKit**: https://www.rdkit.org/ (Cheminformatics toolkit)
- **Mordred**: https://github.com/mordred-descriptor/mordred (Molecular descriptors)
- **ChemProp**: https://github.com/chemprop/chemprop (Message passing neural networks)

**Model Interpretation:**
- **Captum**: https://captum.ai/ (PyTorch interpretability)
- **SHAP**: https://github.com/slundberg/shap (SHapley Additive exPlanations)
- **LIME**: https://github.com/marcotcr/lime (Local interpretable model-agnostic explanations)

**Hyperparameter Tuning:**
- **Optuna**: https://optuna.org/ (Automated hyperparameter optimization)
- **Ray Tune**: https://docs.ray.io/en/latest/tune/index.html (Scalable tuning)
- **Weights & Biases**: https://wandb.ai/ (Experiment tracking)

### 11.3 Datasets

**Public Molecular Datasets:**
1. **MoleculeNet**: Collection of datasets for molecular property prediction
   - http://moleculenet.ai/

2. **ZINC Database**: 230M purchasable compounds
   - https://zinc.docking.org/

3. **PubChem**: 100M+ compounds with biological activities
   - https://pubchem.ncbi.nlm.nih.gov/

4. **ChEMBL**: 2M+ bioactive molecules
   - https://www.ebi.ac.uk/chembl/

5. **Tox21**: Toxicity data for 12K compounds
   - https://tripod.nih.gov/tox21/

6. **BBBP (Blood-Brain Barrier Penetration)**: 2K molecules
   - Part of MoleculeNet

7. **ESOL (Solubility)**: 1K molecules with measured solubility
   - https://pubs.acs.org/doi/10.1021/ci034243x

### 11.4 Online Courses and Tutorials

**Deep Learning:**
1. **Deep Learning Specialization** (Coursera - Andrew Ng)
   - https://www.coursera.org/specializations/deep-learning

2. **Fast.ai Practical Deep Learning**
   - https://course.fast.ai/

**Molecular Machine Learning:**
3. **DeepChem Tutorials**
   - https://deepchem.readthedocs.io/en/latest/get_started/tutorials.html

4. **Molecular Machine Learning** (MIT)
   - http://molecularml.github.io/

5. **Machine Learning for Drug Discovery** (Stanford)
   - http://cs229.stanford.edu/proj2019aut/

### 11.5 Useful Blogs and Articles

1. **Distill.pub**: Interactive ML visualizations
   - https://distill.pub/

2. **Pat Walters' Blog**: Practical cheminformatics
   - http://practicalcheminformatics.blogspot.com/

3. **Is Life Worth Living?**: Deep learning for chemistry
   - https://iwatobipen.wordpress.com/

4. **DeepMind Research**: Latest AI research
   - https://deepmind.com/research

### 11.6 Community and Forums

1. **RDKit Discussions**: https://github.com/rdkit/rdkit/discussions
2. **DeepChem Gitter**: https://gitter.im/deepchem/Lobby
3. **r/MachineLearning**: https://www.reddit.com/r/MachineLearning/
4. **r/cheminformatics**: https://www.reddit.com/r/cheminformatics/

---

## 12. Homework Assignment

### Instructions

Complete the following 10 exercises to reinforce your understanding of Day 2 concepts. Submit a Jupyter notebook with code, results, and brief explanations.

**Evaluation Criteria:**
- Code correctness and clarity (40%)
- Results and analysis quality (30%)
- Explanations and insights (20%)
- Code documentation (10%)

**Submission Format:**
- Jupyter notebook (.ipynb)
- Include all outputs and visualizations
- Add markdown cells with explanations
- Ensure code is reproducible

---

### Exercise 1: Implement a Custom Neural Network (10 points)

**Task:** Build a feedforward neural network from scratch using only NumPy (no PyTorch/TensorFlow).

**Requirements:**
- Implement forward propagation
- Implement backward propagation
- Train on a simple molecular dataset (e.g., solubility)
- Compare performance with PyTorch implementation

**Deliverables:**
- Complete implementation with comments
- Training loss plot
- Comparison table

---

### Exercise 2: Activation Function Comparison (8 points)

**Task:** Compare different activation functions on molecular property prediction.

**Requirements:**
- Test: ReLU, Leaky ReLU, ELU, SELU, Tanh
- Use same architecture (3 hidden layers)
- Plot training curves for each
- Analyze convergence speed and final performance

**Deliverables:**
- Training curves for all activations
- Performance comparison table
- Analysis of results (2-3 paragraphs)

---

### Exercise 3: Implement Multi-Task Learning (12 points)

**Task:** Build a multi-task model to predict multiple molecular properties.

**Requirements:**
- Predict at least 3 properties (e.g., solubility, LogP, TPSA)
- Implement task weighting (try 3 different strategies)
- Compare with single-task models
- Visualize shared representations (t-SNE)

**Deliverables:**
- Multi-task model implementation
- Performance comparison
- t-SNE visualization
- Analysis of benefits

---

### Exercise 4: SMILES CNN Implementation (10 points)

**Task:** Implement and optimize a 1D CNN for SMILES.

**Requirements:**
- Test different filter sizes (3, 5, 7, 9)
- Test different numbers of filters (32, 64, 128)
- Implement data augmentation (SMILES enumeration)
- Compare with fingerprint-based model

**Deliverables:**
- CNN implementation
- Hyperparameter search results
- Performance comparison
- Best model configuration

---

### Exercise 5: LSTM vs GRU Comparison (10 points)

**Task:** Compare LSTM and GRU architectures for SMILES processing.

**Requirements:**
- Implement both LSTM and GRU
- Add attention mechanism to both
- Compare training time, memory usage, performance
- Test on sequences of different lengths

**Deliverables:**
- Both implementations
- Performance metrics table
- Training time comparison
- Recommendations

---

### Exercise 6: Transfer Learning Experiment (12 points)

**Task:** Implement transfer learning for a low-data regime task.

**Requirements:**
- Pre-train on large dataset (e.g., solubility, 10K molecules)
- Fine-tune on small dataset (BBB permeability, 500 molecules)
- Try different freezing strategies
- Compare with training from scratch

**Deliverables:**
- Pre-training code
- Fine-tuning code
- Learning curves comparison
- Analysis of data efficiency

---

### Exercise 7: Model Interpretation (10 points)

**Task:** Implement and compare interpretation methods.

**Requirements:**
- Implement gradient-based importance
- Implement integrated gradients
- Apply to 10 test molecules
- Visualize important features
- Validate interpretations

**Deliverables:**
- Implementation of both methods
- Visualizations for 5 molecules
- Comparison of methods
- Validation with domain knowledge

---

### Exercise 8: Hyperparameter Tuning with Optuna (10 points)

**Task:** Use Optuna to find optimal hyperparameters.

**Requirements:**
- Define search space for 6+ hyperparameters
- Run at least 50 trials
- Visualize optimization history
- Analyze parameter importance
- Train final model with best parameters

**Deliverables:**
- Optuna code
- Optimization visualizations
- Best hyperparameters
- Final model performance

---

### Exercise 9: Complete Pipeline Development (15 points)

**Task:** Build a complete end-to-end pipeline for a novel dataset.

**Requirements:**
- Choose a dataset from MoleculeNet (not used in class)
- Data preprocessing and splitting
- Feature engineering
- Model selection (try 3+ architectures)
- Hyperparameter tuning
- Final evaluation and error analysis

**Deliverables:**
- Complete pipeline code
- Detailed README
- Results report (1-2 pages)
- Error analysis

---

### Exercise 10: Research Paper Implementation (13 points)

**Task:** Implement a model from a recent research paper.

**Suggested Papers:**
1. "Molecular graph convolutions: moving beyond fingerprints" (Duvenaud et al., 2015)
2. "Analyzing learned molecular representations for property prediction" (Yang et al., 2019)
3. "Self-Attention-Based Molecular Representation" (Zheng et al., 2019)

**Requirements:**
- Implement core architecture from paper
- Reproduce key results (within 5% of reported performance)
- Apply to new dataset
- Write implementation notes

**Deliverables:**
- Implementation code
- Results comparison with paper
- Application to new dataset
- Implementation notes (1 page)

---

### Bonus Exercise: Ensemble Methods (+5 points)

**Task:** Implement ensemble methods to improve predictions.

**Requirements:**
- Create ensemble of 5+ models (different architectures)
- Try different ensemble strategies (averaging, stacking, voting)
- Compare with individual models
- Analyze diversity of predictions

**Deliverables:**
- Ensemble implementation
- Performance comparison
- Diversity analysis

---

### Submission Guidelines

**File Structure:**
```
homework_day2/
├── README.md
├── exercise_1_custom_nn.ipynb
├── exercise_2_activation_comparison.ipynb
├── exercise_3_multitask.ipynb
├── exercise_4_smiles_cnn.ipynb
├── exercise_5_lstm_gru.ipynb
├── exercise_6_transfer_learning.ipynb
├── exercise_7_interpretation.ipynb
├── exercise_8_optuna.ipynb
├── exercise_9_complete_pipeline.ipynb
├── exercise_10_paper_implementation.ipynb
├── bonus_ensemble.ipynb (optional)
├── data/ (if applicable)
└── models/ (saved models)
```

**README.md should include:**
- Student name and ID
- Brief description of each exercise
- Key findings and insights
- Challenges encountered
- Total time spent

**Deadline:** 7 days from course date

**Grading:** Total 100 points (+ 5 bonus)

---

## 13. Appendix

### Appendix A: Complete Feedforward NN Template

```python
"""
Complete Feedforward Neural Network Template
For Molecular Property Prediction

This template provides a production-ready implementation
with all best practices included.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    """Configuration for model training"""
    
    # Data
    data_path = 'molecular_data.csv'
    smiles_column = 'SMILES'
    target_column = 'property'
    
    # Features
    fingerprint_radius = 2
    fingerprint_bits = 2048
    
    # Model
    input_dim = 2048
    hidden_dims = [512, 256, 128]
    output_dim = 1
    dropout_rate = 0.3
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1e-5
    num_epochs = 100
    patience = 20
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    model_dir = 'models'
    results_dir = 'results'
    
    # Random seed
    random_seed = 42

# ============================================================================
# 2. DATA HANDLING
# ============================================================================

class MolecularDataset(Dataset):
    """Dataset for molecular property prediction"""
    
    def __init__(self, smiles_list, targets, config, scaler=None):
        self.config = config
        self.fingerprints = []
        self.targets = []
        
        # Generate fingerprints
        for smiles, target in tqdm(zip(smiles_list, targets), 
                                   desc="Processing molecules"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, config.fingerprint_radius, nBits=config.fingerprint_bits
                )
                self.fingerprints.append(np.array(fp))
                self.targets.append(target)
        
        # Convert to tensors
        self.fingerprints = np.array(self.fingerprints)
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.fingerprints = self.scaler.fit_transform(self.fingerprints)
        else:
            self.scaler = scaler
            self.fingerprints = self.scaler.transform(self.fingerprints)
        
        self.fingerprints = torch.FloatTensor(self.fingerprints)
        self.targets = torch.FloatTensor(self.targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.fingerprints)
    
    def __getitem__(self, idx):
        return self.fingerprints[idx], self.targets[idx]

# ============================================================================
# 3. MODEL DEFINITION
# ============================================================================

class MolecularNN(nn.Module):
    """Feedforward neural network for molecular property prediction"""
    
    def __init__(self, config):
        super(MolecularNN, self).__init__()
        
        layers = []
        prev_dim = config.input_dim
        
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# 4. TRAINING
# ============================================================================

class Trainer:
    """Training manager"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer and criterion
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_rmse': [], 'val_mae': [], 'val_r2': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = np.array(all_predictions).flatten()
        all_targets = np.array(all_targets).flatten()
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return avg_loss, rmse, mae, r2
    
    def train(self, train_loader, val_loader):
        """Complete training loop"""
        print(f"Training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_rmse, val_mae, val_r2 = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_rmse)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, "
                      f"MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")
    
    def save_model(self, filepath):
        """Save model and training info"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")

# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print("\nTest Set Evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return all_predictions, {'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_results(history, predictions, targets, save_path='results.png'):
    """Plot training history and predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training history
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training History')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Metrics
    axes[0, 1].plot(history['val_rmse'], label='RMSE')
    axes[0, 1].plot(history['val_mae'], label='MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].set_title('Validation Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Predictions
    axes[1, 0].scatter(targets, predictions, alpha=0.5)
    axes[1, 0].plot([targets.min(), targets.max()], 
                     [targets.min(), targets.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predictions')
    axes[1, 0].set_title('Predictions vs True Values')
    axes[1, 0].grid(True)
    
    # Residuals
    residuals = targets - predictions
    axes[1, 1].hist(residuals, bins=30)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {save_path}")

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Set random seeds
    np.random.seed(Config.random_seed)
    torch.manual_seed(Config.random_seed)
    
    # Create directories
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.results_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(Config.data_path)
    smiles = df[Config.smiles_column].values
    targets = df[Config.target_column].values
    
    # Split data
    print("Splitting data...")
    smiles_train, smiles_temp, y_train, y_temp = train_test_split(
        smiles, targets, test_size=0.3, random_state=Config.random_seed
    )
    smiles_val, smiles_test, y_val, y_test = train_test_split(
        smiles_temp, y_temp, test_size=0.5, random_state=Config.random_seed
    )
    
    print(f"Train: {len(smiles_train)}, Val: {len(smiles_val)}, Test: {len(smiles_test)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MolecularDataset(smiles_train, y_train, Config)
    val_dataset = MolecularDataset(smiles_val, y_val, Config, 
                                   scaler=train_dataset.scaler)
    test_dataset = MolecularDataset(smiles_test, y_test, Config,
                                   scaler=train_dataset.scaler)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    # Create model
    print("Creating model...")
    model = MolecularNN(Config)
    
    # Create trainer
    trainer = Trainer(model, Config)
    
    # Train
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(Config.model_dir, f'model_{timestamp}.pth')
    trainer.save_model(model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, metrics = evaluate_model(model, test_loader, trainer.device)
    
    # Plot results
    results_path = os.path.join(Config.results_dir, f'results_{timestamp}.png')
    plot_results(trainer.history, predictions, y_test, results_path)
    
    # Save metrics
    metrics_path = os.path.join(Config.results_dir, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
```

### Appendix B: SMILES Processing Utilities

```python
"""
SMILES Processing Utilities
Complete toolkit for SMILES handling, tokenization, and augmentation
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# ============================================================================
# SMILES TOKENIZATION
# ============================================================================

class SMILESTokenizer:
    """Advanced SMILES tokenizer with comprehensive vocabulary"""
    
    def __init__(self):
        # Define token patterns (order matters!)
        self.token_patterns = [
            r'Br',  # Bromine (must come before 'r')
            r'Cl',  # Chlorine (must come before 'l')
            r'@@',  # Chirality
            r'@',   # Chirality
            r'\[([^\]]+)\]',  # Bracketed atoms
            r'[A-Z][a-z]?',   # Elements
            r'[#%\(\)\+\-0-9=\\/]',  # Other tokens
        ]
        
        self.regex = re.compile('|'.join(self.token_patterns))
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.mask_token = '<MASK>'
        
        # Build vocabulary from common SMILES
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build comprehensive vocabulary"""
        # Common SMILES tokens
        common_tokens = [
            # Elements
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'B', 'Si', 'Se', 'As',
            # Aromatic
            'c', 'n', 'o', 's', 'p',
            # Bonds
            '-', '=', '#', '/', '\\',
            # Branches
            '(', ')',
            # Rings
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '%10', '%11',
            # Chirality
            '@', '@@',
            # Brackets
            '[', ']',
            # Charges
            '+', '-', '++', '--',
            # Hydrogens
            'H',
        ]
        
        # Special tokens
        special_tokens = [
            self.pad_token, self.unk_token,
            self.start_token, self.end_token, self.mask_token
        ]
        
        # Combined vocabulary
        self.vocab = special_tokens + common_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
    
    def tokenize(self, smiles):
        """Tokenize SMILES string"""
        tokens = self.regex.findall(smiles)
        return tokens
    
    def encode(self, smiles, max_length=None, add_special_tokens=True):
        """Convert SMILES to token indices"""
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
        
        # Convert to indices
        indices = [
            self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
            for token in tokens
        ]
        
        # Pad or truncate
        if max_length is not None:
            if len(indices) < max_length:
                indices += [self.token_to_idx[self.pad_token]] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices):
        """Convert token indices back to SMILES"""
        tokens = [self.idx_to_token.get(idx, self.unk_token) for idx in indices]
        
        # Remove special tokens and padding
        tokens = [t for t in tokens if t not in 
                 [self.pad_token, self.start_token, self.end_token]]
        
        return ''.join(tokens)
    
    def batch_encode(self, smiles_list, max_length=None, add_special_tokens=True):
        """Encode batch of SMILES"""
        return [self.encode(s, max_length, add_special_tokens) for s in smiles_list]

# ============================================================================
# SMILES VALIDATION AND CANONICALIZATION
# ============================================================================

def validate_smiles(smiles):
    """Check if SMILES is valid"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def canonicalize_smiles(smiles):
    """Convert to canonical SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def remove_stereochemistry(smiles):
    """Remove stereochemistry information"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except:
        return None

# ============================================================================
# SMILES AUGMENTATION
# ============================================================================

def enumerate_smiles(smiles, n_variants=10):
    """Generate different SMILES representations of same molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        
        variants = set()
        for _ in range(n_variants * 2):  # Generate more, then sample
            variant = Chem.MolToSmiles(mol, doRandom=True)
            variants.add(variant)
        
        variants = list(variants)
        
        # Ensure original is included
        if smiles not in variants:
            variants.append(smiles)
        
        return variants[:n_variants]
    except:
        return [smiles]

def augment_smiles_dataset(smiles_list, labels, augmentation_factor=5):
    """Augment SMILES dataset with enumeration"""
    augmented_smiles = []
    augmented_labels = []
    
    for smiles, label in zip(smiles_list, labels):
        variants = enumerate_smiles(smiles, n_variants=augmentation_factor)
        augmented_smiles.extend(variants)
        augmented_labels.extend([label] * len(variants))
    
    return augmented_smiles, augmented_labels

# ============================================================================
# SMILES FILTERING
# ============================================================================

def filter_smiles_dataset(smiles_list, labels, 
                         remove_invalid=True,
                         remove_duplicates=True,
                         max_length=None,
                         min_atoms=None,
                         max_atoms=None):
    """Filter SMILES dataset based on criteria"""
    filtered_smiles = []
    filtered_labels = []
    seen_canonical = set()
    
    for smiles, label in zip(smiles_list, labels):
        # Check validity
        if remove_invalid:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
        
        # Check duplicates
        if remove_duplicates:
            canonical = canonicalize_smiles(smiles)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
        
        # Check length
        if max_length is not None and len(smiles) > max_length:
            continue
        
        # Check atom count
        if min_atoms is not None or max_atoms is not None:
            mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumHeavyAtoms()
            
            if min_atoms is not None and n_atoms < min_atoms:
                continue
            if max_atoms is not None and n_atoms > max_atoms:
                continue
        
        filtered_smiles.append(smiles)
        filtered_labels.append(label)
    
    return filtered_smiles, filtered_labels

# ============================================================================
# SMILES STATISTICS
# ============================================================================

def compute_smiles_statistics(smiles_list):
    """Compute statistics about SMILES dataset"""
    stats = {
        'total_molecules': len(smiles_list),
        'valid_molecules': 0,
        'length_stats': {},
        'atom_stats': {},
        'ring_stats': {},
    }
    
    lengths = []
    atom_counts = []
    ring_counts = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            stats['valid_molecules'] += 1
            lengths.append(len(smiles))
            atom_counts.append(mol.GetNumHeavyAtoms())
            ring_counts.append(Chem.GetSSSR(mol))
    
    # Length statistics
    stats['length_stats'] = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'median': np.median(lengths)
    }
    
    # Atom statistics
    stats['atom_stats'] = {
        'mean': np.mean(atom_counts),
        'std': np.std(atom_counts),
        'min': np.min(atom_counts),
        'max': np.max(atom_counts),
        'median': np.median(atom_counts)
    }
    
    # Ring statistics
    stats['ring_stats'] = {
        'mean': np.mean(ring_counts),
        'std': np.std(ring_counts),
        'min': np.min(ring_counts),
        'max': np.max(ring_counts)
    }
    
    return stats

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example SMILES
    smiles_examples = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    # Initialize tokenizer
    tokenizer = SMILESTokenizer()
    
    # Tokenize
    for smiles in smiles_examples:
        tokens = tokenizer.tokenize(smiles)
        encoded = tokenizer.encode(smiles, max_length=50)
        decoded = tokenizer.decode(encoded)
        
        print(f"\nSMILES: {smiles}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded[:10]}...")
        print(f"Decoded: {decoded}")
    
    # Augmentation
    print("\n\nSMILES Enumeration:")
    variants = enumerate_smiles(smiles_examples[1], n_variants=5)
    for i, variant in enumerate(variants, 1):
        print(f"{i}. {variant}")
    
    # Statistics
    print("\n\nDataset Statistics:")
    stats = compute_smiles_statistics(smiles_examples)
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Valid molecules: {stats['valid_molecules']}")
    print(f"Average length: {stats['length_stats']['mean']:.1f}")
    print(f"Average atoms: {stats['atom_stats']['mean']:.1f}")
```

### Appendix C: Model Evaluation Suite

```python
"""
Comprehensive Model Evaluation Suite
Tools for thorough model performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, classification_report
)
from scipy import stats
import pandas as pd

# ============================================================================
# REGRESSION METRICS
# ============================================================================

class RegressionEvaluator:
    """Comprehensive regression model evaluation"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()
        self.residuals = self.y_true - self.y_pred
    
    def compute_metrics(self):
        """Compute all regression metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['MSE'] = mean_squared_error(self.y_true, self.y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = mean_absolute_error(self.y_true, self.y_pred)
        metrics['R2'] = r2_score(self.y_true, self.y_pred)
        
        # Additional metrics
        metrics['Max_Error'] = np.max(np.abs(self.residuals))
        metrics['Mean_Residual'] = np.mean(self.residuals)
        metrics['Std_Residual'] = np.std(self.residuals)
        
        # Relative metrics
        metrics['MAPE'] = np.mean(np.abs(self.residuals / (self.y_true + 1e-10))) * 100
        
        # Correlation
        metrics['Pearson_r'], metrics['Pearson_p'] = stats.pearsonr(
            self.y_true, self.y_pred
        )
        metrics['Spearman_r'], metrics['Spearman_p'] = stats.spearmanr(
            self.y_true, self.y_pred
        )
        
        return metrics
    
    def plot_results(self, save_path='regression_evaluation.png'):
        """Create comprehensive visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Predictions vs True
        axes[0, 0].scatter(self.y_true, self.y_pred, alpha=0.5)
        axes[0, 0].plot([self.y_true.min(), self.y_true.max()],
                        [self.y_true.min(), self.y_true.max()],
                        'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predictions vs True Values')
        axes[0, 0].grid(True)
        
        # Add R² to plot
        r2 = r2_score(self.y_true, self.y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}',
                        transform=axes[0, 0].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Residuals vs Predicted
        axes[0, 1].scatter(self.y_pred, self.residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True)
        
        # 3. Residuals Distribution
        axes[0, 2].hist(self.residuals, bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residual Distribution')
        axes[0, 2].axvline(x=0, color='r', linestyle='--')
        axes[0, 2].grid(True)
        
        # 4. Q-Q Plot
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True)
        
        # 5. Absolute Error vs True
        abs_errors = np.abs(self.residuals)
        axes[1, 1].scatter(self.y_true, abs_errors, alpha=0.5)
        axes[1, 1].set_xlabel('True Values')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Error vs True Values')
        axes[1, 1].grid(True)
        
        # 6. Error Distribution by Range
        # Bin true values and compute error statistics
        bins = np.linspace(self.y_true.min(), self.y_true.max(), 10)
        bin_indices = np.digitize(self.y_true, bins)
        
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_means.append(np.mean(abs_errors[mask]))
                bin_stds.append(np.std(abs_errors[mask]))
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        axes[1, 2].errorbar(bin_centers, bin_means, yerr=bin_stds,
                           fmt='o-', capsize=5)
        axes[1, 2].set_xlabel('True Value Range')
        axes[1, 2].set_ylabel('Mean Absolute Error')
        axes[1, 2].set_title('Error by Value Range')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Evaluation plots saved to {save_path}")
    
    def print_report(self):
        """Print detailed evaluation report"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*70)
        print("REGRESSION EVALUATION REPORT")
        print("="*70)
        
        print("\n1. Basic Metrics:")
        print(f"   MSE:  {metrics['MSE']:.4f}")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE:  {metrics['MAE']:.4f}")
        print(f"   R²:   {metrics['R2']:.4f}")
        
        print("\n2. Error Statistics:")
        print(f"   Max Error:    {metrics['Max_Error']:.4f}")
        print(f"   Mean Residual: {metrics['Mean_Residual']:.4f}")
        print(f"   Std Residual:  {metrics['Std_Residual']:.4f}")
        print(f"   MAPE:         {metrics['MAPE']:.2f}%")
        
        print("\n3. Correlation:")
        print(f"   Pearson r:  {metrics['Pearson_r']:.4f} (p={metrics['Pearson_p']:.4e})")
        print(f"   Spearman r: {metrics['Spearman_r']:.4f} (p={metrics['Spearman_p']:.4e})")
        
        print("\n4. Data Statistics:")
        print(f"   True values:  Mean={np.mean(self.y_true):.4f}, "
              f"Std={np.std(self.y_true):.4f}")
        print(f"   Predictions:  Mean={np.mean(self.y_pred):.4f}, "
              f"Std={np.std(self.y_pred):.4f}")
        print(f"   N samples: {len(self.y_true)}")
        
        print("\n" + "="*70)

# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(results_dict, metric='RMSE'):
    """
    Compare multiple models
    
    Args:
        results_dict: Dict of {model_name: {'y_true': ..., 'y_pred': ...}}
        metric: Metric to use for comparison
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        evaluator = RegressionEvaluator(results['y_true'], results['y_pred'])
        metrics = evaluator.compute_metrics()
        
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'Pearson_r': metrics['Pearson_r']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(by=metric)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['RMSE', 'MAE', 'R2']
    for idx, metric in enumerate(metrics_to_plot):
        axes[idx].bar(df['Model'], df[metric])
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randn(n_samples) * 2 + 5
    y_pred = y_true + np.random.randn(n_samples) * 0.5
    
    # Evaluate
    evaluator = RegressionEvaluator(y_true, y_pred)
    evaluator.print_report()
    evaluator.plot_results()
    
    # Compare models
    results_dict = {
        'Model A': {'y_true': y_true, 'y_pred': y_pred},
        'Model B': {'y_true': y_true, 'y_pred': y_true + np.random.randn(n_samples) * 0.7},
        'Model C': {'y_true': y_true, 'y_pred': y_true + np.random.randn(n_samples) * 0.3},
    }
    
    compare_models(results_dict)
```