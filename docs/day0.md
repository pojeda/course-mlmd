# Day 0: Machine Learning and Deep Learning Fundamentals

## Welcome to the Course!

Before exploring applications in molecular science and materials research, this introductory section is designed to build a strong 
foundation in the core principles of machine learning. Whether you are revisiting familiar concepts or encountering them for the 
first time, these fundamentals will prepare you for the more advanced topics covered throughout Days 1–.

## Learning Objectives

By the end of Day 0, you will be able to:

* Explain the fundamental principles of machine learning
* Differentiate among the main types of learning tasks
* Identify and mitigate overfitting and underfitting
* Apply appropriate validation and model evaluation methods
* Describe the basic structure and function of neural networks
* Understand key strategies for model optimization and hyperparameter tuning

---

## 1. What Is Machine Learning?

### 1.1 Definition

Machine learning is a branch of artificial intelligence focused on developing algorithms that learn patterns directly from data. Rather 
than explicitly programming a computer with a fixed set of rules, we provide examples and allow the model to infer the underlying relationships 
on its own.

### Traditional Programming

```text
Rules + Data → Output
```

### Machine Learning

```text
Data + Desired Output → Learned Model
```

## 1.2 Why Use Machine Learning in Science?

Machine learning has become an essential tool in molecular and materials science because it enables 
researchers to extract valuable insights from large and complex datasets. By learning directly 
from data, ML methods can complement traditional theoretical and experimental approaches in several 
important ways:

* **Predict material and molecular properties** without relying solely on computationally expensive 
simulations or laboratory experiments
* **Identify hidden relationships and trends** within complex scientific data
* **Support the design of new molecules and materials** by proposing promising candidates and guiding 
hypothesis generation
* **Speed up scientific discovery** by dramatically reducing the time required for screening and analysis
* **Handle high-dimensional problems** that are difficult or impossible to address using conventional techniques

### 1.3 Main Categories of Machine Learning

#### Supervised Learning

Supervised learning involves training a model using labeled data, where both the inputs and the 
corresponding outputs are known. The goal is to learn the relationship between them in order to make 
predictions on new, unseen data.

Common supervised learning tasks include:

* **Regression** — predicting continuous numerical quantities, such as binding energies, melting temperatures, or reaction rates
* **Classification** — assigning data to discrete categories, such as toxic vs. non-toxic compounds or active vs. inactive molecules

??? hint "Example"
    ```python
    # Example: Predicting molecular solubility using linear regression

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # ---------------------------------------------------
    # Input features (molecular descriptors)
    # Columns:
    # [molecular_weight, polar_surface_area, logP]
    # ---------------------------------------------------

    X = np.array([
        [180.1, 45.2, 1.2],
        [250.3, 60.1, 2.5],
        [320.5, 75.0, 3.8],
        [150.2, 30.5, 0.8],
        [275.4, 68.2, 2.9],
        [210.0, 50.0, 1.7]
    ])

    # Experimental solubility values
    y = np.array([12.5, 8.1, 3.2, 15.0, 6.4, 10.2])

    # ---------------------------------------------------
    # Split dataset into training and testing sets
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------
    # Create and train the model
    # ---------------------------------------------------

    model = LinearRegression()

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Evaluate the model
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error:", mse)

    # ---------------------------------------------------
    # Predict solubility for a new molecule
    # Example molecule:
    # molecular_weight = 240
    # polar_surface_area = 55
    # logP = 2.1
    # ---------------------------------------------------

    new_molecule = np.array([[240.0, 55.0, 2.1]])

    prediction = model.predict(new_molecule)

    print("Predicted solubility:", prediction[0])
    ```


#### Unsupervised Learning

Unsupervised learning focuses on analyzing data without predefined labels or target values. Instead of 
learning from known answers, the algorithm explores the data to uncover hidden structures, relationships, and patterns.

Common unsupervised learning techniques include:

* **Clustering** — organizing molecules or materials into groups based on similarities in their properties or 
structural features
* **Dimensionality reduction** — simplifying high-dimensional datasets to enable visualization and interpretation 
of complex chemical or materials spaces
* **Anomaly detection** — identifying rare, unusual, or unexpected molecules that differ significantly from the 
majority of the dataset



```python
# Example: Clustering molecules by similarity using K-Means

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Example molecular fingerprints
# Each row represents a molecule
# Each column represents a simplified molecular feature
# ---------------------------------------------------

molecular_fingerprints = np.array([
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0]
])

# ---------------------------------------------------
# Create the K-Means clustering model
# ---------------------------------------------------

kmeans = KMeans(n_clusters=4, random_state=42)

# Assign each molecule to a cluster
clusters = kmeans.fit_predict(molecular_fingerprints)

# ---------------------------------------------------
# Display clustering results
# ---------------------------------------------------

for i, cluster_id in enumerate(clusters):
    print(f"Molecule {i + 1} belongs to Cluster {cluster_id}")

# ---------------------------------------------------
# Visualize clusters using the first two features
# ---------------------------------------------------

plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    molecular_fingerprints[:, 0],
    molecular_fingerprints[:, 1],
    c=clusters,
    s=100
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Molecular Clustering with K-Means")
plt.savefig("clustering.png", dpi=300, bbox_inches="tight")
plt.show()
```

#### Reinforcement Learning

Reinforcement learning is a machine learning approach in which an agent learns to make decisions 
through interaction with an environment. By receiving rewards or penalties based on its actions, 
the model gradually discovers strategies that maximize long-term performance.

In molecular and materials science, reinforcement learning can be applied to tasks such as:

* **Molecular optimization** — designing or modifying molecules to achieve target properties such as 
improved stability, activity, or solubility
* **Synthesis planning** — identifying efficient reaction routes and optimal synthetic pathways 
for chemical compounds
* **Experimental design** — selecting the most informative experiments to accelerate discovery while 
minimizing cost and computational effort


```python
# Basic Reinforcement Learning Example for Molecular Optimization

import random

# ---------------------------------------------------
# Example molecules represented by simple properties
# Each molecule has:
# - size
# - stability
# - solubility
# ---------------------------------------------------

molecule_space = [
    {"name": "Molecule A", "size": 2, "stability": 5, "solubility": 8},
    {"name": "Molecule B", "size": 5, "stability": 7, "solubility": 4},
    {"name": "Molecule C", "size": 3, "stability": 9, "solubility": 6},
    {"name": "Molecule D", "size": 7, "stability": 4, "solubility": 3},
    {"name": "Molecule E", "size": 4, "stability": 8, "solubility": 7},
]

# ---------------------------------------------------
# Reward function
# Goal:
# Favor molecules with high stability and solubility
# ---------------------------------------------------

def evaluate_properties(molecule):

    reward = (
        molecule["stability"] +
        molecule["solubility"]
    )

    return reward

# ---------------------------------------------------
# Simple RL agent
# ---------------------------------------------------

class RandomAgent:

    def select_action(self, state):

        # Randomly choose a new molecule
        return random.choice(molecule_space)

    def learn(self, state, action, reward):

        print(
            f"Learning from transition:\n"
            f"  {state['name']} -> {action['name']}\n"
            f"  Reward = {reward}\n"
        )

# ---------------------------------------------------
# Initialize agent
# ---------------------------------------------------

agent = RandomAgent()

num_episodes = 5

# ---------------------------------------------------
# Reinforcement learning loop
# ---------------------------------------------------

for episode in range(num_episodes):

    print(f"\nEpisode {episode + 1}")

    # Start from a random molecule
    state = random.choice(molecule_space)

    done = False
    step = 0

    while not done:

        # Agent proposes a molecular modification
        new_molecule = agent.select_action(state)

        # Evaluate molecular properties
        reward = evaluate_properties(new_molecule)

        # Agent learns from the reward
        agent.learn(state, new_molecule, reward)

        # Update current state
        state = new_molecule

        step += 1

        # Stop after a few optimization steps
        if step >= 3:
            done = True
```

---

## 2. The Machine Learning Workflow

### 2.1 Defining the Machine Learning Problem

A successful machine learning project begins with a clear and well-structured problem definition. Before 
selecting algorithms or training models, it is important to establish the scientific objective and 
determine how success will be evaluated.

The typical workflow includes:

1. **Identify the objective** — determine the property, behavior, or phenomenon you want to predict, 
classify, or explore.
2. **Select the appropriate learning task** — decide whether the problem is best formulated as regression, 
classification, clustering, generative modeling, or another ML approach.
3. **Establish evaluation criteria** — define the metrics that will be used to measure model performance and 
determine whether the model meets the desired objectives.

Example:

> “Develop a binary classification model capable of predicting whether a molecule can cross the blood–brain barrier with an accuracy greater than 85%.”

---

### 2.2 Data Collection and Preparation

High-quality data is one of the most important components of any machine learning workflow. The reliability 
and performance of a model strongly depend on the quality, diversity, and consistency of the training data.

#### Data Sources

Scientific datasets can originate from several different sources, including:

* **Experimental measurements** obtained from laboratory characterization and testing
* **Computational simulations**, such as Density Functional Theory (DFT) calculations or Molecular Dynamics 
(MD) simulations
* **Public scientific databases**, including resources such as PubChem, ChEMBL, and Materials Project


#### Data Quality Checks
```python
# Fully working example: data preparation + feature engineering

import pandas as pd
import numpy as np

# ---------------------------------------------------
# 1. Create a small example dataset
# ---------------------------------------------------

data = pd.DataFrame({
    "molecule_name": [
        "Ethanol",
        "Acetic acid",
        "Benzene",
        "Acetone",
        "Phenol",
        "Ethanol",          # duplicate row
        "Invalid molecule",
        "Large outlier"
    ],
    "smiles": [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(=O)C",
        "c1ccccc1O",
        "CCO",              # duplicate row
        "not_a_smiles",     # invalid molecule
        "CCCCCCCCCCCCCCCC"
    ],
    "property": [
        -0.31,
        -0.17,
        -2.13,
        -0.24,
        -1.46,
        -0.31,              # duplicate value
        np.nan,             # missing value
        50.0                # artificial outlier
    ]
})

# Save dataset as CSV
data.to_csv("molecular_data.csv", index=False)

# ---------------------------------------------------
# 2. Load data
# ---------------------------------------------------

data = pd.read_csv("molecular_data.csv")

print("Original data:")
print(data)

# ---------------------------------------------------
# 3. Check for missing values
# ---------------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())

# ---------------------------------------------------
# 4. Check for duplicates
# ---------------------------------------------------

print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# ---------------------------------------------------
# 5. Check distributions
# ---------------------------------------------------

print("\nSummary statistics:")
print(data.describe())

# ---------------------------------------------------
# 6. Remove missing values
# ---------------------------------------------------

data = data.dropna(subset=["smiles", "property"])

# ---------------------------------------------------
# 7. Remove outliers using the 3-sigma rule
# ---------------------------------------------------

z_scores = np.abs(
    (data["property"] - data["property"].mean()) / data["property"].std()
)

data_clean = data[z_scores < 3].copy()

print("\nCleaned data:")
print(data_clean)

# ---------------------------------------------------
# 8. Feature engineering with RDKit
# ---------------------------------------------------

from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_features(smiles):
    """
    Convert a molecule represented by a SMILES string
    into numerical molecular descriptors.
    """

    mol = Chem.MolFromSmiles(smiles)

    # Handle invalid molecules
    if mol is None:
        return None

    features = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_aromatic_rings": Descriptors.NumAromaticRings(mol)
    }

    return features

# ---------------------------------------------------
# 9. Apply feature engineering to each molecule
# ---------------------------------------------------

feature_rows = []

for _, row in data_clean.iterrows():
    features = calculate_features(row["smiles"])

    if features is not None:
        features["molecule_name"] = row["molecule_name"]
        features["smiles"] = row["smiles"]
        features["property"] = row["property"]
        feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

# Reorder columns
features_df = features_df[
    [
        "molecule_name",
        "smiles",
        "molecular_weight",
        "logP",
        "num_h_donors",
        "num_h_acceptors",
        "tpsa",
        "num_rotatable_bonds",
        "num_aromatic_rings",
        "property"
    ]
]

print("\nFinal feature table:")
print(features_df)

# ---------------------------------------------------
# 10. Save final processed dataset
# ---------------------------------------------------

features_df.to_csv("molecular_features.csv", index=False)

print("\nProcessed dataset saved as molecular_features.csv")
```

**pandas** is a widely used Python library for data analysis and manipulation. It provides powerful tools for working 
with structured data such as tables and spreadsheets through objects called DataFrames. In machine learning and scientific 
computing, pandas is commonly used to load datasets, clean missing values, filter rows, compute statistics, and organize 
data before training models.

**RDKit*** is an open-source cheminformatics library designed for working with molecular and chemical data. It allows 
researchers to represent molecules computationally, calculate molecular descriptors and fingerprints, visualize chemical 
structures, and perform tasks such as similarity analysis and feature engineering for machine learning applications in 
chemistry, drug discovery, and materials science.


### 2.3 Training, Validation, and Test Sets

A fundamental principle in machine learning is that models must be evaluated on data they have never seen before.

> **Critical principle:** Never test a model using the same data used for training.

If the model is evaluated on training data, it may memorize examples instead of learning general patterns, leading to overfitting and poor performance on new data.

To avoid this problem, datasets are usually divided into three parts:

#### Training Set

The **training set** is used to teach the model and learn patterns from the data.

#### Validation Set

The **validation set** is used during development to tune hyperparameters, compare models, and monitor overfitting.

#### Test Set

The **test set** is used only for the final evaluation of the model on unseen data.

### Typical Dataset Split

| Dataset        | Typical Fraction |
| -------------- | ---------------- |
| Training Set   | 70%              |
| Validation Set | 15%              |
| Test Set       | 15%              |


### Conceptual Workflow

```text id="u7gj59"
Training Set   → Learn patterns
Validation Set → Tune the model
Test Set       → Final evaluation
```



```python
# Basic example: training, validation, and test split

import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. Create a small example dataset
# ---------------------------------------------------

# X = input features
# y = target values

X = np.array([
    [1.0, 2.0],
    [2.0, 1.5],
    [3.0, 3.5],
    [4.0, 4.5],
    [5.0, 5.5],
    [6.0, 6.5],
    [7.0, 7.5],
    [8.0, 8.5],
    [9.0, 9.5],
    [10.0, 10.5]
])

y = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

# ---------------------------------------------------
# 2. First split: training set and temporary set
# ---------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ---------------------------------------------------
# 3. Second split: validation set and test set
# ---------------------------------------------------

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

# ---------------------------------------------------
# 4. Print the results
# ---------------------------------------------------

print("Training set:")
print("X_train:")
print(X_train)
print("y_train:")
print(y_train)

print("\nValidation set:")
print("X_val:")
print(X_val)
print("y_val:")
print(y_val)

print("\nTest set:")
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)

# ---------------------------------------------------
# 5. Print the sizes
# ---------------------------------------------------

print("\nDataset sizes:")
print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))
```

### Scaffold-Based Splitting

In molecular machine learning, **scaffold-based splitting** divides datasets according to the core chemical 
structure of molecules rather than randomly splitting individual samples.

A molecular scaffold represents the main structural framework of a molecule, such as its ring systems and backbone.

This approach is important because molecules with similar scaffolds often have similar properties. With a random 
split, very similar molecules may appear in both the training and test sets, leading to overly optimistic performance.

Scaffold-based splitting provides a more realistic evaluation by ensuring that structurally related molecules 
remain in the same subset.


### Conceptual Example

```text id="9ev7qo"
Random Split:
Train → Benzene
Test  → Phenol

Very similar molecules appear in both sets
```

```text id="6w3w49"
Scaffold Split:
Train → Aromatic compounds
Test  → Different chemical scaffolds

Test molecules are structurally different
```

This strategy is widely used in molecular property prediction, drug discovery, and materials science to 
better evaluate model generalization to new chemical structures.


```python
# example: scaffold-based train/test split

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------
# 1. Example molecules (SMILES strings)
# ---------------------------------------------------

molecules = [
    "CCO",                  # Ethanol
    "CCCO",                 # Propanol
    "c1ccccc1",             # Benzene
    "c1ccccc1O",            # Phenol
    "CC(=O)O",              # Acetic acid
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CCN(CC)CC",            # Triethylamine
    "c1ccncc1"              # Pyridine
]

# Example target property
# (e.g., solubility or biological activity)

y = np.array([1.2, 1.5, 0.3, 0.4, 2.1, 0.8, 1.7, 0.5])

# ---------------------------------------------------
# 2. Generate simple numerical features
# ---------------------------------------------------

X = []

for smiles in molecules:

    mol = Chem.MolFromSmiles(smiles)

    features = [
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
        mol.GetRingInfo().NumRings()
    ]

    X.append(features)

X = np.array(X)

# ---------------------------------------------------
# 3. Define scaffold extraction function
# ---------------------------------------------------

def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    return scaffold

# ---------------------------------------------------
# 4. Compute molecular scaffolds
# ---------------------------------------------------

scaffolds = [get_scaffold(smiles) for smiles in molecules]

print("Molecular scaffolds:\n")

for mol, scaffold in zip(molecules, scaffolds):
    print(f"{mol:35s} -> {scaffold}")

# ---------------------------------------------------
# 5. Perform scaffold-based split
# ---------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=scaffolds)
)

# ---------------------------------------------------
# 6. Create training and test sets
# ---------------------------------------------------

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

train_molecules = [molecules[i] for i in train_idx]
test_molecules = [molecules[i] for i in test_idx]

# ---------------------------------------------------
# 7. Display results
# ---------------------------------------------------

print("\nTraining molecules:")

for mol in train_molecules:
    print(mol)

print("\nTest molecules:")

for mol in test_molecules:
    print(mol)

# ---------------------------------------------------
# 8. Dataset sizes
# ---------------------------------------------------

print("\nDataset sizes:")
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
```
--------------------------------------------------------
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
Build a complete machine learning pipeline to predict molecular solubility. The dataset can be downloaded from:
J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005 (https://pubs.acs.org/doi/10.1021/ci034243x)

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