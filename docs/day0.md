# Day 0: Machine Learning and Deep Learning Fundamentals

## Welcome to the Course!

Before diving into molecular and materials applications, this foundational day ensures everyone has a solid understanding of core machine learning concepts. Whether you're refreshing your knowledge or learning these concepts for the first time, this material will prepare you for the advanced topics in Days 1-5.

## Learning Objectives
By the end of Day 0, you will:
- Understand the fundamental concepts of machine learning
- Distinguish between different types of learning problems
- Recognize and address overfitting and underfitting
- Apply proper validation and evaluation techniques
- Understand the basics of neural networks
- Know how to optimize and tune models

---

## 1. What is Machine Learning?

### 1.1 Definition
Machine learning is the science of programming computers to learn from data without being explicitly programmed. Instead of writing rules, we provide examples and let the algorithm discover patterns.

**Traditional Programming**:
```
Rules + Data → Output
```

**Machine Learning**:
```
Data + Output → Rules (Model)
```

### 1.2 Why Machine Learning for Science?

In molecular and materials science, ML helps us:
- **Predict properties** without expensive experiments or simulations
- **Discover patterns** in complex datasets
- **Generate hypotheses** for new molecules or materials
- **Accelerate discovery** by orders of magnitude
- **Navigate high-dimensional spaces** that are intractable by traditional methods

### 1.3 Types of Machine Learning

#### Supervised Learning
Learn from labeled examples (input-output pairs):
- **Regression**: Predict continuous values (e.g., binding affinity, melting point)
- **Classification**: Predict categories (e.g., toxic/non-toxic, active/inactive)

```python
# Example: Predicting molecular solubility (regression)
X = molecular_features  # Input: molecular descriptors
y = solubility_values   # Output: measured solubility

model.fit(X, y)  # Learn the relationship
prediction = model.predict(new_molecule)  # Predict for new molecule
```

#### Unsupervised Learning
Find patterns in unlabeled data:
- **Clustering**: Group similar molecules together
- **Dimensionality reduction**: Visualize high-dimensional chemical space
- **Anomaly detection**: Find unusual molecules

```python
# Example: Clustering molecules by similarity
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(molecular_fingerprints)
# Molecules in same cluster are structurally similar
```

#### Reinforcement Learning
Learn through trial and error with rewards:
- **Molecular optimization**: Generate molecules with desired properties
- **Synthesis planning**: Find optimal reaction pathways
- **Experiment design**: Choose most informative experiments

```python
# Example: RL agent learns to design molecules
for episode in range(num_episodes):
    state = initial_molecule
    while not done:
        action = agent.select_action(state)  # Modify molecule
        reward = evaluate_properties(new_molecule)
        agent.learn(state, action, reward)
        state = new_molecule
```

---

## 2. The Machine Learning Workflow

### 2.1 Problem Definition
1. **Define the goal**: What do you want to predict or discover?
2. **Choose the task type**: Regression, classification, generation?
3. **Define success metrics**: How will you measure performance?

Example: "Predict whether a molecule can cross the blood-brain barrier (binary classification) with >85% accuracy."

### 2.2 Data Collection and Preparation

#### Data Collection
- Experimental measurements
- Computational simulations (DFT, MD)
- Public databases (PubChem, ChEMBL, Materials Project)
- Literature mining

#### Data Quality Checks
```python
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('molecular_data.csv')

# Check for missing values
print(data.isnull().sum())

# Check for duplicates
print(f"Duplicates: {data.duplicated().sum()}")

# Check distributions
print(data.describe())

# Remove outliers (example: 3 sigma rule)
z_scores = np.abs((data['property'] - data['property'].mean()) / data['property'].std())
data_clean = data[z_scores < 3]
```

#### Feature Engineering
Transform raw data into meaningful features:
```python
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    features = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol)
    }
    
    return features
```

### 2.3 Train-Test Split

**Critical principle**: Never test on training data!

```python
from sklearn.model_selection import train_test_split

# Basic split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (maintains class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Important for molecules**: Use scaffold-based splits to test generalization to new chemical structures:
```python
from sklearn.model_selection import GroupShuffleSplit

# Split by molecular scaffold (core structure)
scaffolds = [get_scaffold(mol) for mol in molecules]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=scaffolds))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

---

## 3. Overfitting and Underfitting

### 3.1 The Bias-Variance Tradeoff

**Underfitting (High Bias)**:
- Model is too simple
- Poor performance on both training and test data
- Doesn't capture underlying patterns

**Overfitting (High Variance)**:
- Model is too complex
- Excellent on training data, poor on test data
- Memorizes noise instead of learning patterns

**Sweet Spot**:
- Balanced complexity
- Good performance on both training and test data
- Generalizes to new examples

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate example data
X = np.linspace(0, 10, 50)
y = 2 * X + 1 + np.random.normal(0, 2, 50)

# Underfitting: degree 1 polynomial (too simple)
underfit_model = np.poly1d(np.polyfit(X, y, 1))

# Good fit: degree 2 polynomial
good_model = np.poly1d(np.polyfit(X, y, 2))

# Overfitting: degree 15 polynomial (too complex)
overfit_model = np.poly1d(np.polyfit(X, y, 15))

# Visualize
X_plot = np.linspace(0, 10, 200)
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_plot, underfit_model(X_plot), label='Underfitting', linestyle='--')
plt.plot(X_plot, good_model(X_plot), label='Good Fit')
plt.plot(X_plot, overfit_model(X_plot), label='Overfitting', linestyle=':')
plt.legend()
plt.show()
```

### 3.2 Detecting Overfitting

**Learning Curves**: Plot training and validation performance vs. training set size

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_squared_error'
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training Error')
plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation Error')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Learning Curves')
plt.show()

# Signs of overfitting:
# - Large gap between training and validation curves
# - Training error much lower than validation error
# - Validation error increases or plateaus
```

### 3.3 Preventing Overfitting

#### 1. Get More Data
The most effective solution when possible:
```python
# Data augmentation for molecules
def augment_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    # Generate different SMILES representations
    augmented = []
    for _ in range(5):
        random_smiles = Chem.MolToSmiles(mol, doRandom=True)
        augmented.append(random_smiles)
    
    return augmented
```

#### 2. Regularization
Add penalty for model complexity:

**L1 Regularization (Lasso)**: Encourages sparsity
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)  # alpha controls regularization strength
model.fit(X_train, y_train)
```

**L2 Regularization (Ridge)**: Penalizes large weights
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

**Elastic Net**: Combines L1 and L2
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
```

#### 3. Cross-Validation
Use all data for both training and validation:
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validation R²: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### 4. Feature Selection
Remove irrelevant or redundant features:
```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top 10 features
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features.tolist()}")
```

#### 5. Early Stopping (for neural networks)
Stop training when validation error starts increasing:
```python
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10  # Stop if no improvement for 10 epochs
)
```

#### 6. Dropout (for neural networks)
Randomly deactivate neurons during training:
```python
import torch.nn as nn

class MolecularNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)  # Drop 30% of neurons
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
```

### 3.4 Addressing Underfitting

1. **Use a more complex model**: Add more features, use deeper networks
2. **Remove regularization**: Reduce alpha/lambda values
3. **Engineer better features**: Domain knowledge can help
4. **Train longer**: Increase number of epochs/iterations
5. **Check for errors**: Ensure data is preprocessed correctly

---

## 4. Cross-Validation

### 4.1 Why Cross-Validation?

- Makes efficient use of limited data
- Provides more reliable performance estimates
- Reduces sensitivity to train-test split
- Helps detect overfitting

### 4.2 K-Fold Cross-Validation

Split data into K parts, train on K-1, test on 1, repeat K times:

```python
from sklearn.model_selection import KFold, cross_validate

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Get multiple metrics
scoring = {
    'r2': 'r2',
    'rmse': 'neg_mean_squared_error',
    'mae': 'neg_mean_absolute_error'
}

results = cross_validate(
    model, X, y,
    cv=kf,
    scoring=scoring,
    return_train_score=True
)

print(f"Test R²: {results['test_r2'].mean():.3f} ± {results['test_r2'].std():.3f}")
print(f"Test RMSE: {np.sqrt(-results['test_rmse'].mean()):.3f}")
print(f"Test MAE: {-results['test_mae'].mean():.3f}")
```

### 4.3 Stratified K-Fold

Maintains class distribution in each fold (for classification):

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}:")
    print(f"  Training class distribution: {np.bincount(y[train_idx])}")
    print(f"  Validation class distribution: {np.bincount(y[val_idx])}")
```

### 4.4 Leave-One-Out Cross-Validation (LOOCV)

Each sample is used once as test set:

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
print(f"LOOCV R²: {scores.mean():.3f}")

# Note: Computationally expensive for large datasets!
# Use only when data is very limited
```

### 4.5 Time Series Cross-Validation

For temporal data (don't peek into the future!):

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}:")
    print(f"  Train: {train_idx.min()} to {train_idx.max()}")
    print(f"  Test: {test_idx.min()} to {test_idx.max()}")
```

---

## 5. Model Evaluation Metrics

### 5.1 Regression Metrics

#### Mean Absolute Error (MAE)
Average absolute difference between predictions and true values:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.3f}")

# Interpretation: Average prediction error in original units
# Lower is better
# Robust to outliers
```

#### Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")

# Interpretation: Penalizes large errors more than MAE
# RMSE in same units as target variable
# Lower is better
```

#### R² (Coefficient of Determination)
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")

# Interpretation: 
# R² = 1: Perfect predictions
# R² = 0: As good as predicting mean
# R² < 0: Worse than predicting mean
# Range: (-∞, 1]
```

#### Visualization
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], 
         [y_true.min(), y_true.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'R² = {r2:.3f}, RMSE = {rmse:.3f}')
plt.legend()
plt.axis('equal')
plt.show()
```

### 5.2 Classification Metrics

#### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Inactive', 'Active'])
disp.plot()
plt.show()

#                 Predicted
#              Negative  Positive
# Actual  Neg     TN        FP
#         Pos     FN        TP
```

#### Accuracy, Precision, Recall, F1-Score
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.3f}  - (TP + TN) / Total")
print(f"Precision: {precision:.3f} - TP / (TP + FP) - How many predicted positives are correct?")
print(f"Recall:    {recall:.3f}    - TP / (TP + FN) - How many actual positives did we find?")
print(f"F1-Score:  {f1:.3f}       - Harmonic mean of precision and recall")
```

#### ROC Curve and AUC
```python
from sklearn.metrics import roc_curve, roc_auc_score

# Get probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# AUC interpretation:
# 1.0: Perfect classifier
# 0.5: Random guessing
# < 0.5: Worse than random (flip predictions!)
```

#### Precision-Recall Curve
Better than ROC for imbalanced datasets:

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### 5.3 Choosing the Right Metric

**For Regression**:
- MAE: Easy to interpret, robust to outliers
- RMSE: Penalizes large errors
- R²: Normalized, easy to compare across datasets

**For Classification**:
- Balanced classes: Accuracy, F1-score
- Imbalanced classes: Precision, Recall, AUC, Average Precision
- Cost-sensitive: Define custom metric based on FP/FN costs

**For Molecular Applications**:
- Drug discovery: Prioritize recall (find all active compounds)
- Toxicity prediction: Prioritize precision (avoid false negatives)
- Property prediction: RMSE or MAE depending on outlier sensitivity

---

## 6. Hyperparameter Tuning

### 6.1 What are Hyperparameters?

Parameters set before training (not learned from data):
- Learning rate
- Number of layers/neurons
- Regularization strength
- Number of trees in random forest
- Kernel parameters in SVM

### 6.2 Grid Search

Try all combinations in a grid:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,  # Use all CPUs
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

### 6.3 Random Search

Sample random combinations (more efficient):

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions,
    n_iter=50,  # Number of random combinations to try
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.3f}")
```

### 6.4 Bayesian Optimization

Intelligent search using previous results:

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_spaces = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(10, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

bayes_search = BayesSearchCV(
    RandomForestRegressor(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_:.3f}")
```

---

## 7. Neural Networks Basics

### 7.1 Architecture Components

#### Neuron (Perceptron)
Basic building block:
```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

```python
import numpy as np

class Neuron:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
    
    def forward(self, x):
        return np.dot(self.weights, x) + self.bias
    
    def activate(self, z):
        # ReLU activation
        return np.maximum(0, z)
```

#### Layers
```python
import torch.nn as nn

# Feedforward network
model = nn.Sequential(
    nn.Linear(input_dim, 128),      # Input layer
    nn.ReLU(),                       # Activation
    nn.Dropout(0.2),                 # Regularization
    nn.Linear(128, 64),              # Hidden layer
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, output_dim)        # Output layer
)
```

### 7.2 Activation Functions

#### Common Activations
```python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 100)

# ReLU: max(0, x)
relu = F.relu(x)

# Sigmoid: 1 / (1 + e^-x)
sigmoid = torch.sigmoid(x)

# Tanh: (e^x - e^-x) / (e^x + e^-x)
tanh = torch.tanh(x)

# Leaky ReLU: max(0.01x, x)
leaky_relu = F.leaky_relu(x, negative_slope=0.01)

# Softmax (for multiple classes)
logits = torch.tensor([[1.0, 2.0, 3.0]])
softmax = F.softmax(logits, dim=1)
print(f"Softmax output: {softmax}")  # Sums to 1
```

**When to use**:
- **ReLU**: Default choice for hidden layers
- **Sigmoid**: Binary classification output
- **Tanh**: When you want outputs in [-1, 1]
- **Softmax**: Multi-class classification output
- **Leaky ReLU**: When dealing with dying ReLU problem

### 7.3 Loss Functions

#### Regression
```python
import torch.nn as nn

# Mean Squared Error
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)

# Mean Absolute Error
mae_loss = nn.L1Loss()
loss = mae_loss(predictions, targets)

# Huber Loss (robust to outliers)
huber_loss = nn.SmoothL1Loss()
loss = huber_loss(predictions, targets)
```

#### Classification
```python
# Binary Cross-Entropy
bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid
loss = bce_loss(logits, targets)

# Multi-class Cross-Entropy
ce_loss = nn.CrossEntropyLoss()  # Includes softmax
loss = ce_loss(logits, targets)
```

### 7.4 Optimization

#### Gradient Descent
```python
# Basic gradient descent
learning_rate = 0.01

for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X)
    loss = loss_function(predictions, y)
    
    # Backward pass
    loss.backward()  # Compute gradients
    
    # Update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    
    # Zero gradients
    model.zero_grad()
```

#### Common Optimizers
```python
import torch.optim as optim

# Stochastic Gradient Descent (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (Adaptive Moment Estimation) - most common
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_function(predictions, y)
    loss.backward()
    optimizer.step()
```

### 7.5 Batch Training

```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training with batches
for epoch in range(num_epochs):
    epoch_loss = 0
    
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = loss_function(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
```

### 7.6 Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# Step decay: reduce LR every N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Reduce on plateau: reduce when metric stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Training with scheduler
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, dataloader, optimizer)
    val_loss = validate(model, val_dataloader)
    
    # Update learning rate
    scheduler.step()  # For StepLR and CosineAnnealingLR
    # scheduler.step(val_loss)  # For ReduceLROnPlateau
    
    print(f"Epoch {epoch + 1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

---

## 8. Common Pitfalls and Best Practices

### 8.1 Data Leakage

**Problem**: Information from test set influences training

**Common mistakes**:
```python
# WRONG: Standardize before splitting
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses info from entire dataset!
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# CORRECT: Fit on training, transform both
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training
X_test_scaled = scaler.transform(X_test)  # Transform using training stats
```

```python
# WRONG: Feature selection on entire dataset
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Leakage!
X_train, X_test = train_test_split(X_selected)

# CORRECT: Feature selection in each CV fold
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('model', RandomForestRegressor())
])

scores = cross_val_score(pipeline, X, y, cv=5)
```

### 8.2 Not Using Validation Set

**Problem**: Tuning hyperparameters on test set

```python
# WRONG: Tune on test set
best_accuracy = 0
best_params = None

for params in parameter_grid:
    model.set_params(**params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)  # Leakage!
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

# CORRECT: Use separate validation set or cross-validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

best_accuracy = 0
best_params = None

for params in parameter_grid:
    model.set_params(**params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)  # Tune on validation
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

# Final evaluation on test set
model.set_params(**best_params)
model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
```

### 8.3 Ignoring Class Imbalance

**Problem**: Poor performance on minority class

**Solutions**:
```python
# 1. Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

model = RandomForestClassifier(class_weight=class_weight_dict)

# 2. Resampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Undersample majority class
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

# 3. Use appropriate metrics
# Don't use accuracy! Use precision, recall, F1, or AUC
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 8.4 Not Checking for Errors

```python
# Always validate your preprocessing
def validate_data(X, y):
    """Comprehensive data validation"""
    
    # Check for NaN
    assert not np.isnan(X).any(), "Features contain NaN values"
    assert not np.isnan(y).any(), "Targets contain NaN values"
    
    # Check for infinite values
    assert not np.isinf(X).any(), "Features contain infinite values"
    
    # Check shapes
    assert X.shape[0] == y.shape[0], "X and y have different number of samples"
    
    # Check for constant features
    constant_features = (X.std(axis=0) == 0).sum()
    if constant_features > 0:
        print(f"Warning: {constant_features} constant features detected")
    
    # Check target distribution
    print(f"Target distribution: mean={y.mean():.3f}, std={y.std():.3f}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return True

validate_data(X_train, y_train)
```

### 8.5 Forgetting to Set Random Seeds

```python
# For reproducibility, set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 9. Practical Exercise: Complete ML Pipeline

### Task
Build a complete machine learning pipeline to predict molecular solubility.

### Dataset
```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load data - adjust column name for target variable
data = pd.read_csv('esol.csv')

print(f"Dataset size: {len(data)}")
print(data.head())
print(f"Column names: {data.columns.tolist()}")
```

### Step 1: Feature Engineering
```python
def calculate_molecular_features(smiles):
    """Calculate molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol)
    }
    
    return features

# Calculate features for all molecules
features_list = []
valid_indices = []

for idx, smiles in enumerate(data['SMILES']):
    features = calculate_molecular_features(smiles)
    if features is not None:
        features_list.append(features)
        valid_indices.append(idx)

# Create feature DataFrame
X = pd.DataFrame(features_list)
# Use the correct column name for solubility
y = data.loc[valid_indices, 'measured log(solubility:mol/L)'].values

print(f"\nFeatures shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Valid molecules: {len(valid_indices)} / {len(data)}")
```

### Step 2: Data Validation and Exploration
```python
# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())

# Check distributions
print("\nFeature statistics:")
print(X.describe())

print("\nTarget statistics:")
print(f"Mean: {y.mean():.3f}")
print(f"Std: {y.std():.3f}")
print(f"Min: {y.min():.3f}")
print(f"Max: {y.max():.3f}")

# Visualize correlations
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
correlations = X.corrwith(pd.Series(y, index=X.index))
correlations.sort_values().plot(kind='barh')
plt.xlabel('Correlation with Solubility')
plt.title('Feature Correlations')
plt.tight_layout()
#plt.show()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Step 3: Train-Test Split
```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
```

### Step 4: Preprocessing
```python
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 5: Model Training with Cross-Validation
```python
# Try different models
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0)
}

# Cross-validation
print("\nCross-validation results:")
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:20s}: R² = {scores.mean():.3f} ± {scores.std():.3f}")

# Select best model
best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")
```

### Step 6: Hyperparameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV

# Tune the best model (Random Forest in this example)
if best_model_name == 'Random Forest':
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, 1.0, None]  # Changed: removed 'auto', added valid options
    }
    
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV R²: {random_search.best_score_:.3f}")
    
    best_model = random_search.best_estimator_
```

### Step 7: Final Evaluation
```python
# Train on full training set
best_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test = best_model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\nFinal Results:")
print(f"Training   - R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
print(f"Test       - R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")

# Check for overfitting
if train_r2 - test_r2 > 0.1:
    print("\nWarning: Possible overfitting detected!")
else:
    print("\nModel generalizes well!")
```

### Step 8: Visualization and Analysis
```python
# Prediction plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_pred_train, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('True Solubility')
axes[0].set_ylabel('Predicted Solubility')
axes[0].set_title(f'Training Set (R² = {train_r2:.3f})')
axes[0].axis('equal')

# Test set
axes[1].scatter(y_test, y_pred_test, alpha=0.5)
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('True Solubility')
axes[1].set_ylabel('Predicted Solubility')
axes[1].set_title(f'Test Set (R² = {test_r2:.3f})')
axes[1].axis('equal')

plt.tight_layout()
#plt.show()
plt.savefig('predicted_solubility.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(importances.head())
    
    plt.figure(figsize=(10, 6))
    importances.plot(x='feature', y='importance', kind='barh')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    #plt.show()
    plt.savefig('best_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Residual analysis
residuals = y_test - y_pred_test

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(132)
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.subplot(133)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
#plt.show()
plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Step 9: Model Persistence
```python
import joblib

# Save model and scaler
joblib.dump(best_model, 'solubility_model.pkl')
joblib.dump(scaler, 'solubility_scaler.pkl')

print("\nModel saved successfully!")

# Load and use model
loaded_model = joblib.load('solubility_model.pkl')
loaded_scaler = joblib.load('solubility_scaler.pkl')

# Make prediction for new molecule
new_smiles = "CCO"  # Ethanol
new_features = calculate_molecular_features(new_smiles)
new_X = pd.DataFrame([new_features])
new_X_scaled = loaded_scaler.transform(new_X)
prediction = loaded_model.predict(new_X_scaled)

print(f"\nPrediction for {new_smiles}: {prediction[0]:.3f}")
```

---

## 10. Key Takeaways

### Essential Concepts
1. **Always split your data** before any preprocessing
2. **Use cross-validation** for reliable performance estimates
3. **Watch for overfitting**: Monitor both training and validation performance
4. **Choose appropriate metrics** based on your problem
5. **Tune hyperparameters** systematically
6. **Validate your data**: Check for errors, outliers, and leakage
7. **Set random seeds** for reproducibility
8. **Document everything**: Parameters, preprocessing steps, results

### Machine Learning Workflow Summary
```
1. Define Problem → 2. Collect Data → 3. Explore Data → 
4. Preprocess → 5. Split Data → 6. Train Models → 
7. Cross-Validate → 8. Tune Hyperparameters → 
9. Evaluate on Test Set → 10. Deploy/Iterate
```

### Red Flags
- Training accuracy >> Test accuracy → Overfitting
- Both accuracies low → Underfitting
- Test accuracy > Training accuracy → Data leakage
- Inconsistent CV scores → Data problems or small dataset
- Perfect scores → Check for data leakage!

---

## 11. Preparation for Days 1-5

### Prerequisites Check
You should now understand:
- ✓ Supervised vs unsupervised learning
- ✓ Training, validation, and test sets
- ✓ Overfitting and underfitting
- ✓ Cross-validation
- ✓ Common metrics (MSE, MAE, R², accuracy, precision, recall, AUC)
- ✓ Hyperparameter tuning
- ✓ Basic neural network concepts

### What's Next?
- **Day 1**: Apply these concepts to molecular representations
- **Day 2**: Deep learning for molecular property prediction
- **Day 3**: Graph neural networks for molecular structures
- **Day 4**: Generative models for molecular design
- **Day 5**: Advanced applications and deployment

### Recommended Practice
Before Day 1, try:
1. Implement the solubility prediction exercise above
2. Experiment with different models and hyperparameters
3. Try other sklearn datasets (boston housing, wine quality)
4. Read sklearn documentation on your favorite algorithms

---

## 12. Additional Resources

### Books
- "Hands-On Machine Learning" - Aurélien Géron
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Deep Learning" - Goodfellow, Bengio, and Courville

### Online Courses
- Andrew Ng's Machine Learning (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS229 (Machine Learning)

### Documentation
- Scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/
- TensorFlow: https://tensorflow.org/

### Practice Platforms
- Kaggle: https://kaggle.com/
- Google Colab: https://colab.research.google.com/
- Papers with Code: https://paperswithcode.com/

---

## Homework Assignment

Complete the following before Day 1:

1. **Implement a basic ML pipeline**: Use the solubility prediction example or choose your own dataset
2. **Compare 3 models**: Try Random Forest, SVR, and a neural network
3. **Perform hyperparameter tuning**: Use Grid Search or Random Search
4. **Analyze results**: Create visualizations and interpret feature importance
5. **Document your findings**: What worked? What didn't? Why?

### Bonus Challenges
- Implement k-fold cross-validation from scratch
- Build a simple neural network using only NumPy
- Create a function to detect and handle data leakage
- Visualize the decision boundary of a classifier

---