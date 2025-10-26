# Day 1: Foundations of Machine Learning for Molecular Systems

## Course Overview
Welcome to the Machine Learning and Deep Learning for Biomolecular Systems and Material Science course. This intensive 5-day program will equip you with the knowledge and skills to apply cutting-edge ML techniques to molecular design, property prediction, and materials discovery.

## Learning Objectives
By the end of Day 1, you will:
- Understand the fundamental ML concepts relevant to molecular sciences
- Learn multiple molecular representation methods and their trade-offs
- Implement basic ML models for molecular property prediction
- Work with chemical databases and molecular descriptors
- Understand the unique challenges of applying ML to chemistry

---

## 1. Introduction to ML in Molecular Sciences

### 1.1 Why Machine Learning for Molecules?

The chemical space is vast—estimates suggest there are 10^60 possible drug-like molecules, far more than atoms in the universe. Traditional approaches to drug discovery and materials design involve:
- Synthesizing and testing compounds one by one (expensive, slow)
- Running quantum mechanical calculations for each molecule (computationally expensive)
- Trial-and-error experimentation (low success rate)

Machine learning has revolutionized our ability to:

**Predict Properties Without Experiments**
- Calculate solubility, toxicity, binding affinity computationally
- Screen millions of compounds in silico before synthesis
- Reduce time from years to weeks

**Discover Structure-Property Relationships**
- Identify which molecular features drive desired properties
- Understand mechanisms of action
- Transfer knowledge across molecular families

**Navigate Chemical Space Efficiently**
- Explore 10^60 possible molecules intelligently
- Focus experimental resources on most promising candidates
- Find novel scaffolds outside known chemistry

**Accelerate Discovery Pipelines**
- Traditional drug discovery: 10-15 years, $2.6B per drug
- ML-assisted discovery: Potentially 2-3x faster, significantly cheaper
- Example: Insilico Medicine designed a novel drug candidate in 46 days

### 1.2 Success Stories

**COVID-19 Drug Repurposing**
- ML models screened 6,000+ FDA-approved drugs against SARS-CoV-2
- Identified Baricitinib (arthritis drug) as potential treatment
- Approved by FDA for COVID-19 treatment in 2020

**Antibiotic Discovery**
- ML identified Halicin, a novel antibiotic
- Effective against drug-resistant bacteria
- Different from existing antibiotics (discovered through ML, not traditional chemistry)

**Materials Science**
- ML accelerated discovery of solid electrolytes for batteries
- Predicted thermal conductivity of materials 1000x faster than simulations
- Identified new photovoltaic materials

### 1.3 Key Challenges in Molecular ML

#### High Dimensionality
Molecules exist in complex, high-dimensional spaces:
- 3D coordinates for each atom
- Electronic structure information
- Conformational flexibility
- Quantum mechanical properties

**Solution**: Learn compact representations that capture essential features

#### Data Scarcity
Unlike computer vision (millions of labeled images), molecular datasets are small:
- Typical drug dataset: 1,000 - 100,000 compounds
- Experimental measurements are expensive and time-consuming
- Many properties are difficult to measure accurately

**Solution**: Transfer learning, data augmentation, semi-supervised learning

#### Physical Constraints
Models must respect fundamental laws:
- Conservation of energy
- Valence rules (atoms have specific bonding patterns)
- Symmetries (rotation, translation, permutation)
- Quantum mechanical principles

**Solution**: Physics-informed neural networks, equivariant architectures

#### Interpretability Requirements
Black-box predictions aren't enough in science:
- Need to understand WHY predictions work
- Identify key molecular features
- Generate hypotheses for experiments
- Build trust with domain experts

**Solution**: Attention mechanisms, feature importance analysis, explainable AI

#### Distribution Shift
Models trained on one chemical space may fail on another:
- Different molecular scaffolds
- Novel functional groups
- Extreme property values

**Solution**: Domain adaptation, uncertainty quantification, active learning

---

## 2. Molecular Representations

The choice of molecular representation is crucial—it determines what information is available to the model and how efficiently it can learn.

### 2.1 SMILES (Simplified Molecular Input Line Entry System)

SMILES is a text-based notation that represents molecular structure as a string.

#### Basic SMILES Syntax

**Simple Molecules**:
```
Methane:    C
Ethanol:    CCO
Benzene:    c1ccccc1  (lowercase = aromatic)
Water:      O
```

**Branches**:
```
Isobutane:     CC(C)C
               └─ branch in parentheses
```

**Double and Triple Bonds**:
```
Ethene:     C=C
Ethyne:     C#C
CO2:        O=C=O
```

**Rings**:
```
Cyclohexane:    C1CCCCC1
                └─ matching numbers close ring

Naphthalene:    c1ccc2ccccc2c1
                └─ fused rings
```

**Stereochemistry**:
```
(R)-Alanine:    N[C@@H](C)C(=O)O
                  └─ @ indicates chirality
```

#### Working with SMILES in Python

```python
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Create molecule from SMILES
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)

# Check validity
if mol is None:
    print("Invalid SMILES!")
else:
    print(f"Valid molecule with {mol.GetNumAtoms()} atoms")

# Visualize
img = Draw.MolToImage(mol, size=(300, 300))
plt.imshow(img)
plt.axis('off')
plt.title('Aspirin')
plt.show()

# Get canonical SMILES (standardized form)
canonical_smiles = Chem.MolToSmiles(mol)
print(f"Canonical SMILES: {canonical_smiles}")

# Generate randomized SMILES (useful for data augmentation)
for i in range(5):
    random_smiles = Chem.MolToSmiles(mol, doRandom=True)
    print(f"Random SMILES {i+1}: {random_smiles}")
```

#### Advantages of SMILES
- **Compact**: Short strings for complex molecules
- **Human-readable**: Chemists can interpret them
- **Widely used**: Most databases provide SMILES
- **Easy to store**: Plain text format

#### Limitations of SMILES
- **Not unique**: Same molecule can have multiple SMILES representations
  ```python
  # All represent ethanol:
  smiles_variants = ["CCO", "OCC", "C(O)C"]
  # Solution: Use canonical SMILES
  ```

- **No 3D information**: Only connectivity, not geometry
  ```python
  # Both are C3H8O but different 3D shapes:
  propanol = "CCCO"      # Linear
  isopropanol = "CC(O)C"  # Branched
  ```

- **Sequence-based**: Hard to capture graph structure directly

- **Fragile**: Single character error invalidates entire SMILES
  ```python
  valid = "CCO"
  invalid = "C CO"  # Space breaks it
  ```

### 2.2 Molecular Fingerprints

Fingerprints are fixed-length binary or count vectors that encode molecular structure.

#### Morgan Fingerprints (ECFP - Extended Connectivity Fingerprints)

Morgan fingerprints capture circular neighborhoods around each atom.

**Algorithm**:
1. Initialize each atom with a unique identifier based on properties
2. For each radius (0, 1, 2, ...), update atom identifiers based on neighbors
3. Hash identifiers to fixed-length bit vector

```python
from rdkit.Chem import AllChem
import numpy as np

# Generate Morgan fingerprint
mol = Chem.MolFromSmiles("CCO")
morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
    mol,
    radius=2,      # ECFP4 (diameter = 2*radius = 4)
    nBits=2048     # Fingerprint length
)

# Convert to numpy array
fp_array = np.zeros((2048,))
AllChem.DataStructs.ConvertToNumpyArray(morgan_fp, fp_array)
print(f"Fingerprint shape: {fp_array.shape}")
print(f"Number of set bits: {fp_array.sum()}")

# Morgan fingerprint with counts
morgan_count = AllChem.GetHashedMorganFingerprint(
    mol,
    radius=2,
    nBits=2048
)
# Stores how many times each feature appears
```

**Visualization of Circular Neighborhoods**:
```python
# Visualize which atoms contribute to which bits
from rdkit.Chem import Draw

info = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)

# Show atom environments for specific bits
for bit_id in list(info.keys())[:5]:  # First 5 bits
    atom_ids = [atom_id for atom_id, radius in info[bit_id]]
    img = Draw.DrawMorganBit(mol, bit_id, info)
    # This shows which atoms/bonds contribute to this bit
```

**Parameters**:
- **Radius**: Larger radius captures more context
  - Radius 1 (ECFP2): Immediate neighbors
  - Radius 2 (ECFP4): Common choice, balances local and broader context
  - Radius 3 (ECFP6): Larger substructures

- **nBits**: Fingerprint length
  - 1024: Fast, but more collisions
  - 2048: Common default
  - 4096: More unique features, slower

#### MACCS Keys

166 predefined structural keys based on common molecular features.

```python
from rdkit.Chem import MACCSkeys

maccs = MACCSkeys.GenMACCSKeys(mol)
print(f"MACCS keys length: {len(maccs)}")

# Each bit represents specific structural feature:
# Bit 1: Contains isotope
# Bit 44: C-O bond
# Bit 79: Aromatic ring
# etc.

# Convert to numpy
maccs_array = np.array(list(maccs.ToBitString()), dtype=int)
```

**Advantages**:
- Interpretable: Each bit has defined meaning
- Compact: Only 166 bits
- Good for similarity searching

**Limitations**:
- Fixed features: Can't capture novel patterns
- Less flexible than Morgan fingerprints

#### RDKit Fingerprints

Topological fingerprints based on molecular paths.

```python
from rdkit.Chem import RDKFingerprint

rdkit_fp = RDKFingerprint(mol, fpSize=2048, maxPath=7)
# maxPath: maximum path length to consider
```

#### Atom Pair and Topological Torsion Fingerprints

Encode distances between atom pairs or torsion angles.

```python
from rdkit.Chem import rdMolDescriptors

# Atom pairs
atom_pairs = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)

# Topological torsions (4 consecutive atoms)
torsions = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
```

#### Comparing Molecules with Fingerprints

```python
from rdkit import DataStructs

mol1 = Chem.MolFromSmiles("CCO")
mol2 = Chem.MolFromSmiles("CCCO")
mol3 = Chem.MolFromSmiles("c1ccccc1")

fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
fp3 = AllChem.GetMorganFingerprintAsBitVect(mol3, 2, 2048)

# Tanimoto similarity (Jaccard index for binary vectors)
sim_12 = DataStructs.TanimotoSimilarity(fp1, fp2)
sim_13 = DataStructs.TanimotoSimilarity(fp1, fp3)

print(f"Similarity(ethanol, propanol): {sim_12:.3f}")  # High (similar structures)
print(f"Similarity(ethanol, benzene): {sim_13:.3f}")   # Low (different structures)

# Other similarity metrics
dice = DataStructs.DiceSimilarity(fp1, fp2)
cosine = DataStructs.CosineSimilarity(fp1, fp2)
```

### 2.3 Molecular Descriptors

Numerical features that capture molecular properties.

#### Types of Descriptors

**1. Physical Descriptors**
```python
from rdkit.Chem import Descriptors, Crippen

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin

# Molecular weight
mw = Descriptors.MolWt(mol)
print(f"Molecular Weight: {mw:.2f} g/mol")

# Lipophilicity (logP - octanol/water partition coefficient)
logp = Crippen.MolLogP(mol)
print(f"LogP: {logp:.2f}")
# LogP > 5: Too lipophilic (Lipinski's Rule of Five)

# Polar Surface Area
tpsa = Descriptors.TPSA(mol)
print(f"TPSA: {tpsa:.2f} Ų")
# TPSA < 140: Likely to cross blood-brain barrier

# Molar Refractivity
mr = Crippen.MolMR(mol)
print(f"Molar Refractivity: {mr:.2f}")
```

**2. Structural Descriptors**
```python
# Hydrogen bond donors and acceptors
h_donors = Descriptors.NumHDonors(mol)
h_acceptors = Descriptors.NumHAcceptors(mol)
print(f"H-Bond Donors: {h_donors}")
print(f"H-Bond Acceptors: {h_acceptors}")

# Rotatable bonds (flexibility)
rot_bonds = Descriptors.NumRotatableBonds(mol)
print(f"Rotatable Bonds: {rot_bonds}")

# Ring information
num_rings = Descriptors.RingCount(mol)
aromatic_rings = Descriptors.NumAromaticRings(mol)
print(f"Total Rings: {num_rings}, Aromatic: {aromatic_rings}")

# Fraction of sp3 carbons (saturation)
frac_sp3 = Descriptors.FractionCsp3(mol)
print(f"Fraction Csp3: {frac_sp3:.2f}")
```

**3. Topological Descriptors**
```python
from rdkit.Chem import GraphDescriptors

# Balaban J index (molecular branching)
balaban = GraphDescriptors.BalabanJ(mol)

# Bertz complexity index
bertz = GraphDescriptors.BertzCT(mol)

# Chi indices (connectivity)
chi0 = GraphDescriptors.Chi0(mol)
chi1 = GraphDescriptors.Chi1(mol)
```

**4. 3D Descriptors**
```python
from rdkit.Chem import AllChem, Descriptors3D

# Generate 3D coordinates
mol_3d = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol_3d, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol_3d)

# 3D descriptors
asphericity = Descriptors3D.Asphericity(mol_3d)
eccentricity = Descriptors3D.Eccentricity(mol_3d)
inertial_shape = Descriptors3D.InertialShapeFactor(mol_3d)
radius_of_gyration = Descriptors3D.RadiusOfGyration(mol_3d)

print(f"Radius of Gyration: {radius_of_gyration:.2f} Ų")
```

#### Drug-Likeness Metrics

**Lipinski's Rule of Five**
```python
def lipinski_rule_of_five(mol):
    """
    Predicts if molecule is drug-like
    Rules:
    - MW <= 500
    - LogP <= 5
    - H-bond donors <= 5
    - H-bond acceptors <= 10
    """
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return violations <= 1  # Allow 1 violation

is_druglike = lipinski_rule_of_five(mol)
print(f"Passes Lipinski's Rule: {is_druglike}")
```

**QED (Quantitative Estimate of Drug-likeness)**
```python
from rdkit.Chem import QED

qed_score = QED.qed(mol)
print(f"QED Score: {qed_score:.3f}")
# Range: [0, 1], higher is more drug-like
# Based on 8 molecular properties
```

**Synthetic Accessibility Score**
```python
from rdkit.Chem import RDConfig
import sys
sys.path.append(f'{RDConfig.RDContribDir}/SA_Score')
import sascorer

sa_score = sascorer.calculateScore(mol)
print(f"SA Score: {sa_score:.2f}")
# Range: [1, 10]
# 1: Easy to synthesize
# 10: Difficult to synthesize
```

#### Creating Feature Vectors

```python
def calculate_molecular_descriptors(smiles):
    """
    Comprehensive descriptor calculation
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens for accurate calculations
    mol = Chem.AddHs(mol)
    
    descriptors = {
        # Physical
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MolMR': Crippen.MolMR(mol),
        
        # Structural
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'RingCount': Descriptors.RingCount(mol),
        
        # Complexity
        'BertzCT': GraphDescriptors.BertzCT(mol),
        'NumBridgeheadAtoms': Descriptors.NumBridgeheadAtoms(mol),
        'NumSpiroAtoms': Descriptors.NumSpiroAtoms(mol),
        
        # Electronic
        'LabuteASA': Descriptors.LabuteASA(mol),
        'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
        
        # Counts
        'NumCarbon': len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]),
        'NumNitrogen': len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]),
        'NumOxygen': len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]),
        'NumHalogens': len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]]),
        
        # Saturation
        'FractionCsp3': Descriptors.FractionCsp3(mol),
        
        # Drug-likeness
        'QED': QED.qed(mol),
    }
    
    return descriptors

# Example usage
smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
import pandas as pd

descriptor_list = [calculate_molecular_descriptors(s) for s in smiles_list]
df_descriptors = pd.DataFrame(descriptor_list)
df_descriptors['SMILES'] = smiles_list

print(df_descriptors)
```

### 2.4 Graph Representations

Molecules as graphs where atoms are nodes and bonds are edges.

#### Graph Structure

```python
import networkx as nx

def mol_to_graph(smiles):
    """Convert molecule to NetworkX graph"""
    mol = Chem.MolFromSmiles(smiles)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (atoms)
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            symbol=atom.GetSymbol(),
            degree=atom.GetDegree(),
            formal_charge=atom.GetFormalCharge(),
            num_h=atom.GetTotalNumHs(),
            hybridization=str(atom.GetHybridization()),
            is_aromatic=atom.GetIsAromatic()
        )
    
    # Add edges (bonds)
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            is_conjugated=bond.GetIsConjugated(),
            is_aromatic=bond.GetIsAromatic()
        )
    
    return G

# Example
G = mol_to_graph("CCO")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Node features: {G.nodes[0]}")
```

#### Adjacency Matrix Representation

```python
def get_adjacency_matrix(smiles, max_atoms=50):
    """Get adjacency matrix with padding"""
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    
    # Initialize matrix
    adj_matrix = np.zeros((max_atoms, max_atoms))
    
    # Fill adjacency matrix
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Symmetric
    
    return adj_matrix, num_atoms

adj, n_atoms = get_adjacency_matrix("CCO")
print(f"Adjacency matrix shape: {adj.shape}")
print(f"Actual atoms: {n_atoms}")
```

#### Node and Edge Features

```python
def get_node_features(atom):
    """Extract features for a single atom"""
    return np.array([
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),  # Number of bonds
        atom.GetFormalCharge(),  # Charge
        atom.GetNumRadicalElectrons(),  # Radicals
        atom.GetHybridization().real,  # sp, sp2, sp3
        atom.GetIsAromatic(),  # Aromaticity
        atom.GetTotalNumHs(),  # Hydrogens
    ])

def get_edge_features(bond):
    """Extract features for a single bond"""
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    
    return np.array([
        bond_type_map.get(bond.GetBondType(), 0),
        bond.GetIsConjugated(),
        bond.GetIsAromatic(),
    ])
```

**Advantages of Graph Representations**:
- Natural for molecules (atoms connected by bonds)
- Permutation invariant (atom order doesn't matter)
- Captures topology and local structure
- Enables Graph Neural Networks (Day 3)

**Limitations**:
- More complex to implement
- Computationally expensive for large molecules
- Requires specialized neural network architectures

---

## 3. Traditional Machine Learning Methods

Before deep learning, these methods were (and still are) workhorses of molecular ML.

### 3.1 Feature Engineering Principles

**Domain Knowledge is Key**:
- Choose descriptors relevant to property being predicted
- For solubility: polarity, surface area, H-bond capacity
- For toxicity: reactive functional groups, lipophilicity

**Feature Scaling**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Good for: Most ML algorithms, assumes normal distribution

# Min-Max scaling (range [0, 1])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Good for: Neural networks, when you need bounded values

# Robust scaling (uses median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Good for: Data with outliers
```

**Feature Selection**:
```python
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor

# Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_high_var = selector.fit_transform(X)

# Univariate feature selection
selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
model = RandomForestRegressor()
rfe = RFE(model, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Feature importance from model
model = RandomForestRegressor()
model.fit(X, y)
selector = SelectFromModel(model, prefit=True, threshold='median')
X_important = selector.transform(X)
```

### 3.2 Random Forests

Ensemble of decision trees, each trained on random subset of data and features.

#### How it Works

1. **Bootstrap Sampling**: Create N random subsets of training data (with replacement)
2. **Random Feature Selection**: At each split, consider random subset of features
3. **Tree Building**: Build deep trees without pruning
4. **Prediction**: Average predictions from all trees (regression) or vote (classification)

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Regression example
rf_reg = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    max_depth=None,  # Grow trees fully
    min_samples_split=2,  # Minimum samples to split node
    min_samples_leaf=1,  # Minimum samples in leaf
    max_features='sqrt',  # Features to consider at each split
    bootstrap=True,  # Use bootstrap sampling
    random_state=42,
    n_jobs=-1  # Use all CPUs
)

# Train
rf_reg.fit(X_train, y_train)

# Predict
y_pred = rf_reg.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(rf_reg, X, y, cv=5, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance
importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i, idx in enumerate(indices[:10]):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Visualize feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

**Advantages**:
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Little hyperparameter tuning needed
- Resistant to overfitting (with enough trees)

**Limitations**:
- Can be slow for very large datasets
- Not great for extrapolation
- Black-box model (hard to interpret individual predictions)

**Hyperparameter Tuning**:
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

### 3.3 Model Evaluation Metrics

#### Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.3f}")

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse:.3f}")

# R² Score
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.3f}")
```

---

## 4. Working with Chemical Databases

### 4.1 Public Databases

#### PubChem

```python
import pubchempy as pcp

# Search by name
results = pcp.get_compounds('aspirin', 'name')
compound = results[0]

print(f"IUPAC Name: {compound.iupac_name}")
print(f"SMILES: {compound.isomeric_smiles}")
print(f"Molecular Formula: {compound.molecular_formula}")
print(f"Molecular Weight: {compound.molecular_weight}")

# Get properties
properties = pcp.get_properties(
    ['MolecularWeight', 'XLogP', 'TPSA', 'Complexity'],
    'aspirin',
    'name'
)
print(properties)
```

#### ChEMBL

```python
from chembl_webresource_client.new_client import new_client

# Target search
target = new_client.target
targets = target.filter(target_synonym__icontains='EGFR')

for t in targets[:5]:
    print(f"{t['pref_name']}: {t['target_chembl_id']}")

# Activity search
activity = new_client.activity
activities = activity.filter(
    target_chembl_id='CHEMBL203',
    standard_type='IC50',
    standard_relation='=',
    pchembl_value__isnull=False
)

# Convert to DataFrame
import pandas as pd

data = []
for act in activities[:1000]:
    data.append({
        'molecule_chembl_id': act['molecule_chembl_id'],
        'smiles': act['canonical_smiles'],
        'ic50': act['standard_value'],
        'pchembl_value': act['pchembl_value']
    })

df = pd.DataFrame(data)
print(df.head())
```

### 4.2 Data Preprocessing

#### Molecular Standardization

```python
from rdkit.Chem import SaltRemover

def standardize_molecule(smiles):
    """Standardize molecular representation"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Remove salts
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)
    
    # Get canonical SMILES
    canonical_smiles = Chem.MolToSmiles(mol)
    
    return canonical_smiles

# Process dataset
smiles_list = ["CC(=O)O.Na", "CCO", "[NH3+]CC[O-]"]
standardized = [standardize_molecule(s) for s in smiles_list]
print(standardized)
```

#### Train/Test Splitting

```python
from sklearn.model_selection import train_test_split

# Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaffold split for molecules
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def scaffold_split(smiles_list, test_size=0.2):
    """Split by molecular scaffolds"""
    scaffolds = defaultdict(list)
    
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(idx)
    
    # Distribute scaffolds
    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)
    
    n_total = len(smiles_list)
    n_test = int(n_total * test_size)
    
    train_idx, test_idx = [], []
    train_count = 0
    
    for scaffold_set in scaffold_sets:
        if train_count + len(scaffold_set) <= n_total - n_test:
            train_idx.extend(scaffold_set)
            train_count += len(scaffold_set)
        else:
            test_idx.extend(scaffold_set)
    
    return train_idx, test_idx
```

---

## 5. Practical Exercise: Solubility Prediction

### Complete Workflow

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Load Data
url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
df = pd.read_csv(url)
print(f"Dataset size: {len(df)}")

# Step 2: Calculate Descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumRings': Descriptors.RingCount(mol),
        'MolMR': Descriptors.MolMR(mol),
        'FractionCsp3': Descriptors.FractionCsp3(mol)
    }

desc_list = [calculate_descriptors(s) for s in df['smiles']]
X_desc = pd.DataFrame([d for d in desc_list if d is not None])
y = df['measured log solubility in mols per litre'].values[:len(X_desc)]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_desc, y, test_size=0.2, random_state=42
)

# Step 4: Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")

# Step 7: Visualize
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Solubility (log M)')
plt.ylabel('Predicted Solubility (log M)')
plt.title(f'Predictions (R² = {r2:.3f})')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Step 8: Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_desc.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance_df.head())
```

**Expected Results**:
- Test RMSE: ~0.7-0.9 log units
- R²: ~0.75-0.85
- Key features: LogP, molecular weight, polar surface area

---

## 6. Key Takeaways

### Molecular Representations
- **SMILES**: Compact text representation, requires careful handling
- **Fingerprints**: Fixed-length vectors, good for similarity and ML
- **Descriptors**: Interpretable features, require domain knowledge
- **Graphs**: Natural representation, enables GNNs (Day 3)

### Traditional ML Methods
- **Random Forests**: Robust baseline, handles non-linearity, provides feature importance
- **SVMs**: Effective in high dimensions, requires scaling
- **Gaussian Processes**: Provides uncertainty, excellent for active learning
- **Gradient Boosting**: Often best performance, requires careful tuning

### Best Practices
1. Always validate molecular structures
2. Use scaffold-based splits for realistic evaluation
3. Scale features appropriately for each algorithm
4. Compare multiple representations (descriptors vs fingerprints)
5. Report multiple metrics (RMSE, MAE, R²)
6. Analyze feature importance for insights
7. Check for data leakage in preprocessing

### Common Pitfalls
- Using random splits instead of scaffold splits
- Forgetting to scale features for SVMs
- Not handling invalid SMILES
- Overfitting due to small datasets
- Ignoring uncertainty in predictions

---

## 7. Resources and Further Reading

### Software Libraries
- **RDKit**: Cheminformatics toolkit - https://www.rdkit.org/
- **scikit-learn**: Machine learning library - https://scikit-learn.org/
- **Pandas**: Data manipulation - https://pandas.pydata.org/
- **Matplotlib**: Visualization

### Databases
- **PubChem**: https://pubchem.ncbi.nlm.nih.gov/
- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **ZINC**: https://zinc.docking.org/
- **Materials Project**: https://materialsproject.org/
- **QM9**: http://quantum-machine.org/datasets/

### Papers
- "Molecular descriptors for chemoinformatics" - Todeschini & Consonni
- "Machine Learning in Materials Informatics" - Butler et al., 2018
- "Guidelines for ML predictive models in biomedical research" - Luo et al., 2016
- "Deep Learning for Molecular Design" - Elton et al., 2019

### Tutorials
- RDKit Cookbook - https://www.rdkit.org/docs/Cookbook.html
- Scikit-learn User Guide - https://scikit-learn.org/stable/user_guide.html
- DeepChem Tutorials - https://deepchem.io/

---

## Homework Assignment

1. **Data Exploration**
   - Download QM9 dataset
   - Calculate descriptors for 1000 random molecules
   - Visualize descriptor distributions and correlations

2. **Classification Model**
   - Build a model to predict if a molecule has dipole moment > 2 Debye
   - Use both descriptors and fingerprints
   - Report precision, recall, and AUC

3. **Model Comparison**
   - Compare Random Forest, SVM, and Gradient Boosting
   - Use 5-fold cross-validation
   - Create visualizations comparing performance

4. **Feature Analysis**
   - Identify the top 5 most important features
   - Explain why these features are relevant
   - Visualize feature importance

5. **Advanced Challenge**
   - Implement scaffold-based splitting
   - Compare performance with random split
   - Discuss implications for model generalization

6. **Prepare Questions**
   - Review neural networks basics (Day 0)
   - Think about limitations of traditional ML for molecules
   - Prepare questions for Day 2 on deep learning

---

## Appendix: Quick Reference Tables

### Molecular Descriptor Ranges

| Descriptor | Range | Interpretation | Drug-like Range |
|------------|-------|----------------|-----------------|
| **Molecular Weight** | 0-∞ | Size | 150-500 g/mol |
| **LogP** | -∞ to +∞ | Lipophilicity | 0-5 |
| **TPSA** | 0-∞ Ų | Polar surface | 20-140 Ų |
| **H-Bond Donors** | 0-∞ | H-bond donors | 0-5 |
| **H-Bond Acceptors** | 0-∞ | H-bond acceptors | 0-10 |
| **Rotatable Bonds** | 0-∞ | Flexibility | 0-10 |
| **QED** | 0-1 | Drug-likeness | > 0.5 |

### Model Selection Guide

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Random Forest** | General purpose | Easy, robust, feature importance | Slower prediction |
| **SVM** | Small datasets | Good generalization | Slow training, needs scaling |
| **Gaussian Process** | Active learning | Uncertainty estimates | Very slow for large data |
| **Gradient Boosting** | Best performance | Highest accuracy | Prone to overfitting |

### Common RDKit Functions

```python
# Molecule creation
mol = Chem.MolFromSmiles("CCO")

# Validation
is_valid = mol is not None

# Canonical SMILES
canonical = Chem.MolToSmiles(mol)

# Add/remove hydrogens
mol_h = Chem.AddHs(mol)
mol_no_h = Chem.RemoveHs(mol)

# 3D coordinates
AllChem.EmbedMolecule(mol)

# Descriptors
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)

# Fingerprints
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

# Similarity
similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

# Substructure search
pattern = Chem.MolFromSmarts("[OH]")
has_match = mol.HasSubstructMatch(pattern)

# Visualization
img = Draw.MolToImage(mol, size=(300, 300))
```
