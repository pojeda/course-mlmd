# Day 3: Graph Neural Networks and Geometric Deep Learning

## Overview

Day 3 focuses on Graph Neural Networks (GNNs) and their applications in molecular and materials science. We'll explore how graphs can represent molecular structures and how specialized neural network architectures can learn from these representations while respecting physical symmetries and constraints.

## Topics Covered

### 1. Message Passing Neural Networks

Message Passing Neural Networks (MPNNs) form the foundation of modern graph neural networks for molecular property prediction. They provide a unified framework for understanding how information flows through graph-structured data, making them particularly well-suited for molecular modeling where atoms and bonds naturally form graph structures.

#### Core Concepts

**Graph Representation:**

In molecular graphs, the structure naturally maps to graph representations:
- **Nodes** represent atoms with rich feature vectors including:
  - Atomic number (element identity)
  - Formal charge and partial charge
  - Hybridization state (sp, sp2, sp3)
  - Number of hydrogen atoms
  - Aromaticity
  - Chirality
  - Atomic mass
  - Degree (number of connections)

- **Edges** represent bonds with attributes such as:
  - Bond type (single, double, triple, aromatic)
  - Bond length/distance
  - Stereochemistry (cis/trans, E/Z)
  - Conjugation
  - Ring membership

- The **graph structure** encodes molecular connectivity, preserving the topological relationships that determine chemical properties

**Why Message Passing?**

The key insight behind MPNNs is that the properties of an atom in a molecule depend not just on the atom itself, but on its chemical environment. Message passing mimics how chemical effects propagate through molecular structures:
- Inductive effects travel through sigma bonds
- Resonance effects propagate through conjugated systems
- Steric effects depend on spatial arrangements of neighboring groups

**Message Passing Framework:**

The MPNN framework consists of three main phases that iterate to build increasingly sophisticated representations:

1. **Message Phase** (T iterations):
   ```
   m_v^(t+1) = Σ_{u∈N(v)} M_t(h_v^t, h_u^t, e_{uv})
   ```
   
   In this phase:
   - Each node v receives messages from its neighbors N(v)
   - The message function M_t is a learnable neural network that combines:
     - h_v^t: the current node's hidden state
     - h_u^t: each neighbor's hidden state
     - e_{uv}: edge features connecting the nodes
   - Messages are computed for all edges simultaneously
   - The superscript t indicates the iteration number
   
   **Intuition**: Think of this as each atom "listening" to what its bonded neighbors are telling it about their local chemical environment.

2. **Update Phase**:
   ```
   h_v^(t+1) = U_t(h_v^t, m_v^(t+1))
   ```
   
   In this phase:
   - Each node updates its representation using the aggregated messages
   - U_t is typically a GRU, LSTM, or feedforward network
   - The update combines the node's previous state with new information
   - This preserves information from earlier iterations while incorporating new context
   
   **Intuition**: After listening to neighbors, each atom updates its own representation to reflect what it learned about its environment.

3. **Aggregation Within Messages**:
   
   The summation in the message phase is just one choice. Other aggregation functions include:
   - **Sum**: `Σ_{u∈N(v)} M_t(...)`  - sensitive to neighborhood size
   - **Mean**: `(1/|N(v)|) Σ_{u∈N(v)} M_t(...)` - normalizes by degree
   - **Max**: `max_{u∈N(v)} M_t(...)` - captures strongest signal
   - **Attention**: `Σ_{u∈N(v)} α_{vu} M_t(...)` - learnable weights (GAT)

**Readout Phase:**

After T message passing steps, we have node-level representations h_v^T for each atom. To predict molecular properties, we need a graph-level representation:

```
y = R({h_v^T | v ∈ G})
```

Common readout functions include:

- **Sum pooling**: `R = Σ_v h_v^T`
  - Captures total contributions from all atoms
  - Sensitive to molecule size
  - Good for extensive properties (like mass, number of electrons)

- **Mean pooling**: `R = (1/|V|) Σ_v h_v^T`
  - Normalizes by number of atoms
  - Better for intensive properties (like density, stability per atom)
  - Size-invariant representation

- **Max pooling**: `R = max_v h_v^T` (element-wise)
  - Captures most significant features
  - Can miss important distributed information

- **Set2Set**: A learnable attention-based aggregation
  - Uses LSTM to iteratively attend to nodes
  - More expressive but computationally expensive
  - Can capture complex relationships between atoms

**Depth and Receptive Fields:**

The number of message passing iterations T determines the receptive field:
- T=1: Each node sees only immediate neighbors (1-hop)
- T=2: Each node sees neighbors and neighbors-of-neighbors (2-hop)
- T=k: Each node sees all nodes within k bonds

For molecular graphs:
- Small molecules (QM9): T=3-5 is usually sufficient
- Proteins: T=5-10 may be needed for long-range interactions
- Trade-off: More iterations = larger receptive field but risk of over-smoothing

#### Key Variants

**Graph Convolutional Networks (GCN):**

GCNs simplify message passing using a spectral approach:

```
H^(t+1) = σ(D^(-1/2) Ã D^(-1/2) H^(t) W^(t))
```

Where:
- Ã = A + I (adjacency matrix with self-loops)
- D is the degree matrix
- This is equivalent to message passing with normalized averaging
- Very efficient for semi-supervised learning on large graphs
- Less flexible for edge features than general MPNNs

**Benefits for chemistry:**
- Fast computation on molecular graphs
- Symmetric normalization prevents exploding/vanishing gradients
- Can be stacked deeply with residual connections

**Limitations:**
- Doesn't naturally incorporate edge features
- Fixed aggregation (normalized sum)
- May struggle with distinguishing certain graph structures (limited expressivity)

**GraphSAGE (Sample and Aggregate):**

GraphSAGE introduces sampling for scalability:

```
h_v^(t+1) = σ(W · CONCAT(h_v^t, AGG({h_u^t | u ∈ Sample(N(v))})))
```

Key innovations:
- **Sampling**: Instead of aggregating from all neighbors, sample a fixed number
- **Multiple aggregators**:
  - Mean: `AGG = (1/|S|) Σ_{u∈S} h_u`
  - LSTM: Process neighbors sequentially (requires ordering)
  - Pooling: `AGG = max(σ(W_pool h_u + b))`
- **Concatenation**: Explicitly preserves self-information

**Benefits for chemistry:**
- Handles variable-sized neighborhoods efficiently
- Inductive learning: can generalize to new molecules not seen during training
- Scalable to very large molecular databases

**Neural Message Passing for Quantum Chemistry (MPNN):**

The original MPNN paper specialized the framework for molecules:

**Architecture:**
```
m_v^(t+1) = Σ_{u∈N(v)} M_t(h_v^t, h_u^t, e_{uv})
           = Σ_{u∈N(v)} A_t(e_{uv}) · h_u^t

h_v^(t+1) = U_t(h_v^t, m_v^(t+1))
           = GRU(h_v^t, m_v^(t+1))
```

Key design choices:
- **Edge networks**: A_t(e_{uv}) is a neural network that produces edge-specific matrices
  - Allows different bond types to transform neighbor information differently
  - Captures the idea that single/double/triple bonds transmit information differently

- **GRU updates**: Using Gated Recurrent Units for the update function
  - Helps with gradient flow through multiple message passing steps
  - Gates control what information to keep vs. update
  - More stable than simple MLPs for deep message passing

- **Virtual edges**: Can add edges between non-bonded atoms within a distance cutoff
  - Captures through-space interactions
  - Important for conformational effects and weak interactions

**Master equations:**
```
# Initialize
h_v^0 = embedding(x_v)  # x_v are input features

# Message passing
for t in range(T):
    for each edge (v,u):
        m_vu = EdgeNetwork(e_vu) @ h_u^t
    m_v = Σ_u m_vu
    h_v^(t+1) = GRU(h_v^t, m_v)

# Readout
h_G = Set2Set({h_v^T | v ∈ G})
y = MLP(h_G)
```

**Training considerations:**
- Typically T=3-6 message passing steps
- Hidden dimensions: 64-256 depending on task complexity
- Batch normalization or layer normalization helps training stability
- Dropout between layers prevents overfitting

#### Applications in Chemistry

**Molecular Property Prediction:**

MPNNs excel at predicting quantum mechanical and physical properties:

1. **Quantum properties** (QM9 dataset):
   - HOMO/LUMO energies (frontier orbitals)
   - Internal energy and enthalpy
   - Free energy and heat capacity
   - Electronic spatial extent
   - Zero-point vibrational energy
   - Atomization energy
   
   **Why MPNNs work**: These properties depend on electronic structure, which is determined by how atoms and bonds are connected. Message passing naturally captures these structural effects.

2. **Physical properties**:
   - Solubility (important for drug absorption)
   - Melting/boiling points
   - Density and refractive index
   - Viscosity
   
   **Challenge**: These properties can depend on 3D conformations, so incorporating geometry helps.

3. **Biological activity**:
   - Toxicity predictions (hERG, Ames, hepatotoxicity)
   - Binding affinity to target proteins
   - ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity)
   - Blood-brain barrier penetration
   
   **Application**: Early-stage drug filtering, reducing costly experimental screening.

**Reaction Outcome Prediction:**

Given reactants and conditions, predict the major products:
- Graph of reactants → MPNN → Product distribution
- Can incorporate reaction conditions (temperature, solvent, catalysts) as global features
- Attention mechanisms can identify reactive sites

**Retrosynthesis Planning:**

Predict how to synthesize a target molecule:
- Target molecule → MPNN → Likely precursors
- Can be formulated as a translation problem (product graph → reactant graphs)
- Helps chemists find synthetic routes for complex molecules

**Drug Discovery and Virtual Screening:**

Screen millions of compounds against target proteins:
- Fast prediction once model is trained (~1000 molecules/second)
- Can be combined with active learning to guide experimental efforts
- Multi-task learning: predict multiple properties simultaneously
- Transfer learning: pre-train on large databases, fine-tune on specific targets

**De Novo Molecular Design:**

- Use MPNNs as discriminators or reward functions in generative models
- Guide molecular generation toward desired properties
- Combine with optimization algorithms (genetic algorithms, reinforcement learning)

#### Limitations and Challenges

**Over-smoothing**: With many message passing steps, node representations become too similar
- **Solution**: Residual connections, jumping knowledge networks

**Limited expressivity**: Some graphs are indistinguishable by message passing
- **Solution**: Add higher-order structural features, use more sophisticated aggregation

**Scalability**: Large molecules or protein graphs can be computationally expensive
- **Solution**: Sampling (GraphSAGE), hierarchical approaches, graph coarsening

**3D structure**: Basic MPNNs ignore 3D geometry
- **Solution**: Add distance as edge features, use equivariant networks (next sections)

---

### 2. Graph Attention Networks

Graph Attention Networks (GATs) introduce attention mechanisms to graph learning, allowing the model to learn which neighbors are most important for each node. This is a significant advancement over basic message passing, where all neighbors contribute equally to a node's update.

#### Motivation and Intuition

**Why Attention for Graphs?**

In molecular contexts, not all bonds are equally important for determining a property:
- In a large molecule, a specific functional group might dominate reactivity
- Some atoms are in the "core" structure while others are in peripheral substituents
- Certain bonds participate in conjugation or resonance, making them more significant
- In protein-ligand binding, only residues near the binding site matter

**Analogy to NLP**: Just as in language, where "bank" means different things in "river bank" vs "savings bank" depending on context, an atom's role depends on which neighbors are most relevant.

GATs allow the network to automatically learn these context-dependent importance weights.

#### Attention Mechanism

**Core Idea:**

Unlike MPNNs that use fixed aggregation (sum, mean) or hand-crafted edge weights, GATs learn attention coefficients α_{ij} that adaptively weigh the importance of each neighbor j for node i.

**Mathematical Framework:**

The attention mechanism consists of three steps:

**Step 1: Compute Attention Logits**
```
e_{ij} = LeakyReLU(a^T [W h_i || W h_j])
```

Breaking this down:
- **W h_i** and **W h_j**: First, transform node features through a shared weight matrix W
  - This projects features into a common space where comparisons are meaningful
  - Dimension: [d_in] → [d_out]
  
- **[W h_i || W h_j]**: Concatenate transformed features of node i and neighbor j
  - Creates a pairwise feature vector
  - Dimension: [2 × d_out]
  
- **a^T [...]**: Apply learned attention vector a
  - Reduces to scalar attention logit e_{ij}
  - The attention vector a learns what feature combinations indicate importance
  - Dimension: [2 × d_out] → [1]
  
- **LeakyReLU**: Non-linearity with slope α for negative values (typically α=0.2)
  - Allows negative attention scores (before softmax)
  - Prevents dead neurons from ReLU saturation

**Intuition**: The attention logit e_{ij} measures how relevant neighbor j is to node i, based on their feature compatibility.

**Step 2: Normalize to Attention Coefficients**
```
α_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / Σ_{k∈N(i)} exp(e_{ik})
```

- **Softmax normalization**: Ensures attention weights sum to 1 over all neighbors
- **Comparison**: Only neighbors compete for attention (not all nodes in graph)
- **Interpretation**: α_{ij} ∈ (0, 1) represents the probability-like importance of neighbor j

**Why softmax?**
- Preserves differentiability (can backpropagate)
- Creates sharp distinctions (high e_{ij} → high α_{ij})
- Normalized weights prevent exploding values in aggregation

**Step 3: Aggregate with Attention Weights**
```
h_i' = σ(Σ_{j∈N(i)} α_{ij} W h_j)
```

- **Weighted sum**: Each neighbor contributes proportionally to its attention weight
- **W h_j**: Uses the same transformation from step 1 (parameter sharing)
- **σ**: Activation function (typically ELU or ReLU)

**Complete Forward Pass Example:**

For an atom with 3 neighbors:
1. Compute logits: e_{i1} = 0.8, e_{i2} = 0.3, e_{i3} = -0.2
2. Apply softmax: α_{i1} = 0.52, α_{i2} = 0.31, α_{i3} = 0.17
3. Aggregate: h_i' = σ(0.52 × W h_1 + 0.31 × W h_2 + 0.17 × W h_3)

**Key Properties:**

- **Self-attention**: Can include self-loops (i,i) so nodes attend to themselves
- **Asymmetric**: α_{ij} ≠ α_{ji} (attention from i→j differs from j→i)
- **Local**: Only attends to direct neighbors (preserves graph structure)
- **Permutation invariant**: Order of neighbors doesn't matter

#### Multi-Head Attention

To stabilize learning and capture different types of relationships simultaneously, GATs employ multiple independent attention mechanisms (heads).

**Multi-Head Aggregation (Hidden Layers):**
```
h_i' = ||_{k=1}^K σ(Σ_{j∈N(i)} α_{ij}^k W^k h_j)
```

Where:
- K = number of attention heads
- Each head k has its own parameters: W^k and a^k
- || denotes concatenation of head outputs
- Output dimension: K × d_out

**Why Multiple Heads?**

Different heads can learn complementary attention patterns:
- **Head 1**: Might focus on electronegative atoms (for polarity)
- **Head 2**: Might focus on aromatic neighbors (for conjugation)
- **Head 3**: Might focus on steric bulk (for size effects)
- **Head 4**: Might attend to formal charges

**Intuition from Chemistry**: Just as a chemist considers multiple factors simultaneously (electronics, sterics, orbital interactions), multiple heads capture different aspects of molecular structure.

**Multi-Head Averaging (Output Layer):**
```
h_i' = σ((1/K) Σ_{k=1}^K Σ_{j∈N(i)} α_{ij}^k W^k h_j)
```

For the final layer, averaging instead of concatenation:
- Keeps output dimension consistent with target
- Ensembles the predictions from different heads
- More stable for final predictions

**Implementation Considerations:**

```python
# Typical hyperparameters
num_heads = 4-8  # More heads = more capacity but more parameters
hidden_dim = 64-256  # Per-head dimension
dropout = 0.1-0.3  # On attention coefficients (attention dropout)
```

**Computational Complexity:**
- Attention computation: O(|E| × d_out) - linear in edges
- Memory for attention: O(|E| × K) - stores attention per head
- Highly parallelizable (all attention coefficients computed independently)

#### Advantages of GATs

**1. Adaptive Neighborhoods**

Unlike fixed aggregation:
```
# Fixed (GCN-style)
h_i' = Σ_{j∈N(i)} (1/√(d_i × d_j)) W h_j  # predetermined weights

# Adaptive (GAT)
h_i' = Σ_{j∈N(i)} α_{ij} W h_j  # learned weights
```

**Benefits:**
- Automatically focuses on relevant neighbors
- Can ignore uninformative connections
- Adapts to different chemical contexts

**Example**: In a drug molecule, GAT can learn to focus on:
- Pharmacophore groups (active parts)
- Hydrogen bond donors/acceptors
- Hydrophobic regions
While downweighting inert carbon chains.

**2. Interpretability**

Attention weights α_{ij} can be visualized and interpreted:

```python
# Extract attention weights
attention_weights = model.get_attention()  # shape: [num_edges, num_heads]

# Visualize for a molecule
mol_graph = molecule.to_graph()
highlight_edges_by_weight(mol_graph, attention_weights, threshold=0.3)
```

**What we can learn:**
- Which atoms influence predictions most
- How information flows through the molecule
- Whether the model learned chemically meaningful patterns
- Debugging: Are attention patterns reasonable?

**Example insights:**
- High attention on C=O bonds for carbonyl chemistry
- Focus on aromatic rings for π-stacking predictions
- Attention following conjugation pathways

**3. Parallelizability**

All attention coefficients for all edges can be computed in parallel:
```python
# Pseudo-code
edge_features = concat(h[edges[:,0]], h[edges[:,1]])  # All edges at once
attention_logits = attention_network(edge_features)  # Parallel
attention_weights = softmax_per_node(attention_logits)  # Parallel within neighbors
```

**Speed advantages:**
- GPU-friendly (matrix operations)
- Scales well to large molecules
- No sequential dependencies (unlike RNNs)

**4. Inductive Learning**

GATs can generalize to completely new graph structures:
- Train on small molecules, test on large ones
- Learn patterns that transfer across different molecular classes
- No fixed graph structure required during training

This is critical for:
- Generalizing to novel drug candidates
- Transfer learning across datasets
- Handling molecules with varying sizes

#### GAT Variants and Extensions

**GATv2: More Expressive Attention**

Original GAT limitation: Attention is somewhat static
```
# GAT: Transform then attend
e_{ij} = a^T [W h_i || W h_j]  # a can only linearly combine features
```

GATv2 improvement: Dynamic attention
```
# GATv2: Attend then transform
e_{ij} = a^T LeakyReLU(W [h_i || h_j])  # Non-linearity before attention
```

**Why it's better:**
- The non-linearity is applied AFTER concatenation
- Allows more complex attention functions
- Empirically: 10-30% better performance on many benchmarks
- Fixes theoretical expressivity limitations of original GAT

**When to use GATv2:**
- Complex molecules with subtle structural differences
- When original GAT plateaus in performance
- Tasks requiring fine-grained attention distinctions

**Molecular GAT: Edge Features**

Challenge: Chemical bonds have important attributes (single/double/triple, stereochemistry) that basic GAT ignores.

**Solution**: Incorporate edge features into attention

```
e_{ij} = a^T [W h_i || W h_j || E e_{ij}]
```

Where:
- e_{ij} = edge feature vector (bond type, distance, ring membership)
- E = edge embedding matrix
- Now attention depends on both node features AND edge features

**Applications:**
- Distinguishing single vs double bonds
- Incorporating 3D distances
- Using bond order information
- Representing stereochemistry

**Example**: In conjugated systems, π-bonds should have higher attention than σ-bonds:
```
# Single bond: e_{ij} = [1,0,0] → lower attention
# Double bond: e_{ij} = [0,1,0] → higher attention (for delocalization)
# Triple bond: e_{ij} = [0,0,1] → highest attention
```

**Graph Transformer Networks:**

Extension to full graph attention (not just neighbors):
```
α_{ij} for all pairs (i,j) in graph  # Not just edges
```

**Trade-offs:**
- **Pro**: Can capture long-range interactions
- **Pro**: More flexible attention patterns
- **Con**: O(|V|²) complexity (vs O(|E|) for GAT)
- **Con**: May lose graph structural bias

**Use when:** Long-range interactions matter (large molecules, proteins)

#### Practical Tips for Using GATs

**Hyperparameter Selection:**

```python
# Good starting points
num_layers = 3-5  # Deeper than this risks over-smoothing
num_heads = 4-8  # More heads for complex tasks
hidden_dim = 64-128 per head
dropout = 0.1-0.3  # Higher dropout for small datasets
learning_rate = 0.001  # Adam optimizer

# For QM9 dataset
config = {
    'num_layers': 4,
    'num_heads': 4,
    'hidden_dim': 128,
    'dropout': 0.2,
    'attention_dropout': 0.1  # Separate dropout on attention weights
}
```

**Common Pitfalls:**

1. **Forgetting self-loops**: Add explicit (i,i) edges or include h_i in aggregation
2. **Over-smoothing**: Too many layers → all nodes become similar
3. **Attention collapse**: All attention goes to one neighbor (use attention dropout)
4. **Memory issues**: K heads × L layers can use lots of GPU memory

**Best Practices:**

```python
# 1. Add residual connections
h_i' = h_i + GAT(h_i)  # Prevents over-smoothing

# 2. Use layer normalization
h_i' = LayerNorm(h_i + GAT(h_i))  # Stabilizes training

# 3. Attention dropout
α_{ij} = Dropout(softmax(e_{ij}))  # Regularizes attention

# 4. Edge features when available
e_{ij} = [bond_type, distance, ring_membership]  # Richer edges
```

**Visualization Strategies:**

```python
# 1. Attention heatmaps
plot_attention_matrix(attention_weights, molecule)

# 2. Highlight important edges
highlight_edges_above_threshold(molecule, attention_weights > 0.3)

# 3. Track attention across layers
for layer in model.layers:
    visualize_attention_distribution(layer.attention)

# 4. Compare heads
for head in range(num_heads):
    visualize_attention_head(head, attention_weights)
```

---

### 3. Equivariant Networks for 3D Structures

Geometric deep learning architectures respect the symmetries and geometric properties of 3D molecular structures, making them particularly powerful for computational chemistry and materials science. These networks go beyond simple graph connectivity to leverage the full 3D geometry of molecules.

#### Motivation: Why Geometry Matters

**The 3D Problem:**

Traditional GNNs treat molecular graphs as topological structures, ignoring spatial arrangements:
- Two molecules with the same connectivity but different 3D shapes (stereoisomers) would be treated identically
- Bond angles and dihedral angles contain crucial information
- 3D distance-based interactions (van der Waals, electrostatics) are not captured
- Forces and other vector properties require 3D information

**Example**: Consider two stereoisomers:
```
cis-2-butene:  H₃C-CH=CH-CH₃ (groups on same side)
trans-2-butene: H₃C-CH=CH-CH₃ (groups on opposite sides)
```
These have:
- Identical graph connectivity
- Different 3D structures
- Different physical properties (melting point, boiling point, reactivity)

A topology-only GNN cannot distinguish them, but a geometry-aware network can.

#### Understanding Symmetries

**Physical Symmetries in Molecular Systems:**

Molecular properties must respect fundamental physical symmetries:

1. **Translation Invariance**: 
   ```
   Property(molecule) = Property(molecule + translation vector)
   ```
   - Moving the entire molecule in space doesn't change its energy or properties
   - Only relative positions matter, not absolute coordinates
   - Mathematically: E(R + t) = E(R) for any translation t

2. **Rotation Invariance**:
   ```
   Property(molecule) = Property(rotate(molecule, θ))
   ```
   - Rotating the molecule doesn't change scalar properties (energy, dipole magnitude)
   - Physical measurements don't depend on orientation in space
   - Mathematically: E(QR) = E(R) for any rotation matrix Q

3. **Permutation Invariance**:
   ```
   Property(atoms[1,2,3,...]) = Property(atoms[permutation])
   ```
   - Labeling atoms 1,2,3 vs 3,1,2 shouldn't change properties
   - Physical reality has no preferred ordering
   - Node ordering is an artifact of representation

4. **Reflection (for some properties)**:
   - Chiral molecules break reflection symmetry
   - Achiral molecules maintain E(R) = E(mirror(R))

**Invariance vs Equivariance:**

- **Invariant**: Output doesn't change under transformation
  - Example: Energy is rotation-invariant scalar
  - E(rotate(molecule)) = E(molecule)

- **Equivariant**: Output transforms consistently with input
  - Example: Forces are rotation-equivariant vectors
  - F(rotate(molecule)) = rotate(F(molecule))
  - If you rotate the molecule, forces rotate the same way

**Why This Matters:**

Networks that violate these symmetries will:
- Learn to recognize rotated versions as different molecules (inefficient)
- Require much more training data to learn all orientations
- Fail to generalize to new orientations
- Predict unphysical results (energy changing with rotation)

Networks that respect symmetries:
- Are more data-efficient (one orientation teaches all)
- Generalize better to new configurations
- Satisfy physical laws by construction
- Often achieve better accuracy with fewer parameters

#### SchNet (Continuous-filter Convolutional Neural Network)

SchNet pioneered the use of continuous convolutions for molecular modeling, treating molecules as continuous 3D objects rather than discrete graphs.

**Core Philosophy:**

Instead of learning fixed filters for discrete bond types, SchNet learns continuous functions that depend smoothly on interatomic distances. This mirrors physics: interactions depend on distance in a continuous way (not discrete jumps).

**Architecture Overview:**

```
Input: 
  - Atomic numbers Z = [Z₁, Z₂, ..., Z_N]
  - 3D coordinates R = [r₁, r₂, ..., r_N]

Output:
  - Molecular property (energy, HOMO, etc.)
```

**Step 1: Atomic Embeddings**

Each atom type is embedded into a feature space:
```
x_i^(0) = Embedding(Z_i)  # Lookup table: atomic number → d-dimensional vector
```

For example:
- Carbon (Z=6) → [0.12, -0.34, 0.56, ...]
- Nitrogen (Z=7) → [0.08, -0.29, 0.61, ...]
- Oxygen (Z=8) → [0.15, -0.41, 0.48, ...]

**Step 2: Continuous Filter Generation**

The innovation of SchNet: filters are functions of distance, not learned for discrete bins.

```
# Compute all pairwise distances
d_ij = ||r_i - r_j||  # Euclidean distance

# Expand distances using radial basis functions (RBFs)
e_ij = [exp(-(d_ij - μ_k)² / σ²) for k in range(K)]

# Generate filter weights from expanded distances
W_ij = MLP(e_ij)  # Neural network: ℝ^K → ℝ^(d×d)
```

**Radial Basis Functions (RBFs):**

RBFs create a smooth representation of distances:
```
# Example: Gaussian RBFs centered at different distances
μ = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, ...]  # Centers in Ångströms
σ = 0.1  # Width

For distance d_ij = 1.8 Å:
  RBF₁(1.8) = exp(-(1.8-0.5)²/0.01) ≈ 0.00  # Far from center
  RBF₂(1.8) = exp(-(1.8-1.0)²/0.01) ≈ 0.00
  RBF₃(1.8) = exp(-(1.8-1.5)²/0.01) ≈ 0.74  # Close to center
  RBF₄(1.8) = exp(-(1.8-2.0)²/0.01) ≈ 0.82  # Closest center
  ...
```

This creates a smooth, continuous representation that:
- Captures nuances between bond lengths
- Allows interpolation to unseen distances
- Provides smooth gradients for optimization

**Step 3: Interaction Blocks (Message Passing)**

```
# For each interaction layer t:
x_i^(t+1) = x_i^(t) + Σ_{j≠i} x_j^(t) ⊙ W_ij^(t)

Where:
  ⊙ is element-wise multiplication (Hadamard product)
  W_ij^(t) = FilterNetwork_t(d_ij) is the distance-dependent filter
```

**Detailed Breakdown:**

1. **Cutoff Function**: Only interact with nearby atoms
   ```
   if d_ij > r_cutoff (typically 5-10 Å):
       W_ij = 0  # No interaction
   else:
       W_ij = FilterNetwork(d_ij) × smooth_cutoff(d_ij)
   ```

   Smooth cutoff prevents discontinuities:
   ```
   f_cutoff(d) = 0.5 × [cos(π × d / r_cutoff) + 1]  if d < r_cutoff
                 0                                     otherwise
   ```

2. **Filter Network Architecture**:
   ```
   e = RBF_expansion(d_ij)      # [K] vector
   h = Dense(e)                  # [K] → [hidden_dim]
   h = ShiftedSoftplus(h)        # Smooth non-linearity
   W = Dense(h)                  # [hidden_dim] → [d×d]
   W = W × f_cutoff(d_ij)       # Apply cutoff
   ```

3. **Atom-wise Update**:
   ```
   # Aggregate filtered neighbor features
   m_i = Σ_{j∈neighbors(i)} x_j ⊙ W_ij
   
   # Update with residual connection
   x_i = x_i + m_i
   ```

**Step 4: Output Modules**

After T interaction blocks:
```
# Atom-wise energy contributions
E_i = MLP_atom(x_i^(T))  # Scalar per atom

# Sum for total energy (size-extensive property)
E_total = Σ_i E_i

# Or average for intensive properties
Property = (1/N) Σ_i MLP(x_i^(T))
```

**Key Design Choices:**

1. **Only distances, not coordinates**:
   - Distances are rotation and translation invariant
   - ||r_i - r_j|| is the same regardless of overall position/orientation
   - This guarantees the network is invariant

2. **Continuous filters**:
   - Smoothly adapts to any distance
   - Better than discrete binning (1.0-1.5 Å, 1.5-2.0 Å, ...)
   - Allows accurate interpolation

3. **Shifted Softplus activation**:
   ```
   ShiftedSoftplus(x) = log(0.5 × exp(x) + 0.5)
   ```
   - Smooth everywhere (unlike ReLU)
   - Important for force prediction: forces = -∇E
   - Provides smooth gradients

**Strengths:**
- **Efficient**: Linear in number of atoms (with cutoff)
- **Scalable**: Handles molecules from 10 to 1000+ atoms
- **Accurate**: State-of-the-art on QM9 and other benchmarks
- **End-to-end differentiable**: Can predict forces via backpropagation
- **Physically motivated**: Design mirrors physics of interactions

**Limitations:**
- **Distance-only**: Doesn't explicitly capture angles
- **Isotropic**: Treats all directions equally (no directionality)
- **Cannot predict vector properties directly**: Forces require gradients

**Implementation Notes:**

```python
# Typical hyperparameters
num_interactions = 6        # Depth of network
num_gaussians = 50          # Number of RBF centers
cutoff = 5.0                # Interaction cutoff (Å)
hidden_dim = 128            # Feature dimension
num_filters = 128           # Filter network dimension
```

#### DimeNet (Directional Message Passing Neural Network)

DimeNet extends SchNet by incorporating angular information, making it significantly more expressive for molecular modeling.

**Key Innovation: Beyond Distances**

While distances capture bond lengths, many properties depend on:
- **Bond angles**: H-O-H angle in water determines properties
- **Dihedral angles**: Rotations around single bonds affect conformation
- **Triplet interactions**: Three-body terms in force fields

**Physics Analogy**: 
- SchNet ≈ Lennard-Jones potential (pairwise, distance-only)
- DimeNet ≈ Force fields with angle terms (CHARMM, AMBER)

**Architecture: Directional Messages**

DimeNet introduces messages that depend on triplets of atoms (i,j,k):

```
                k
               /
              /θ_{ijk}
             /
            j -------- i
          d_jk      d_ij
```

**Step 1: Distance and Angle Embeddings**

```
# Distance embedding (like SchNet)
e_dist(d_ij) = RBF_expansion(d_ij)

# NEW: Angle embedding
θ_{ijk} = angle between vectors (r_j - r_i) and (r_k - r_j)
e_angle(θ_{ijk}) = SphericalBasisFunctions(θ_{ijk})
```

**Spherical Basis Functions:**

Instead of Gaussians (for distances), use spherical harmonics for angles:
```
# Angles are periodic: 0° = 360°
# Use basis that respects this periodicity

SBF_k(θ) = Σ_n c_{nk} × exp(-(n - n_0)²/σ²) × sin(nθ)

# These form a complete basis for representing angular functions
```

**Step 2: Directional Message Passing**

The crucial innovation - messages depend on geometric triplets:

```
# For each atom i:
m_ij = Σ_{k∈N(j), k≠i} MessageBlock(d_ij, d_jk, θ_{ijk}) ⊙ x_k
```

Let's break down `MessageBlock(d_ij, d_jk, θ_{ijk})`:

```python
def MessageBlock(d_ij, d_jk, theta_ijk):
    # Embed distances
    rbf_ij = RBF(d_ij)        # [n_rbf]
    rbf_jk = RBF(d_jk)        # [n_rbf]
    
    # Embed angle
    sbf = SphericalBasis(theta_ijk)  # [n_sbf]
    
    # Combine using bilinear layer
    # This learns correlations between distances and angles
    W = BilinearLayer(rbf_ij, rbf_jk, sbf)  # [d × d] matrix
    
    return W
```

**Bilinear Layer Explained:**

The bilinear layer is crucial for combining geometric features:

```
W = Σ_m Σ_n Σ_l  U_{mnl} × rbf_ij[m] × rbf_jk[n] × sbf[l]

Where U is a learned tensor of parameters
```

This allows the network to learn patterns like:
- "When d_ij ≈ 1.5 Å AND d_jk ≈ 1.2 Å AND θ ≈ 120°" → strong interaction
- Captures correlations between geometric features
- More expressive than treating features independently

**Step 3: Update with Directional Information**

```
# Aggregate directional messages
m_i = Σ_{j∈N(i)} m_ij

# Update node features
x_i = Update(x_i, m_i)  # Typically a residual network
```

**Why This Works Better:**

1. **Angular Information**: Distinguishes linear vs bent vs tetrahedral
   - Example: CO₂ (linear, 180°) vs H₂O (bent, 104.5°)
   - Same atom types, different angles → different properties

2. **Dihedral Sensitivity**: Captures rotational barriers
   - Ethane: staggered vs eclipsed conformations
   - Different angles → different energies

3. **Three-Body Interactions**: More realistic physics
   - Many quantum effects involve three atoms
   - Necessary for accurate force fields

**Maintaining Rotational Invariance:**

Despite using angles, DimeNet remains rotationally invariant because:
- Angles are invariant: θ_{ijk} doesn't change under rotation
- Only distances and angles used (not absolute coordinates)
- No preferred orientation in space

**DimeNet++ Improvements:**

The original DimeNet was slow. DimeNet++ optimized:

```
# DimeNet: T-shaped message passing
Message: k → j → i  (sequential)

# DimeNet++: Optimized message aggregation
Message: All k → all j → all i  (more parallel)
```

**Optimization strategies:**
1. **Shared bilinear layers**: Reduce parameters
2. **Efficient triplet enumeration**: Better data structures
3. **Grouped convolutions**: Reduce computational cost
4. **Memory-efficient attention**: Lower memory footprint

**Performance:**
- DimeNet++: 2-5× faster than DimeNet
- Same or better accuracy
- Can handle larger molecules (100+ atoms)

**Advantages:**
- **Higher accuracy**: 20-40% better MAE on QM9 vs SchNet
- **Captures geometry**: Angles and dihedrals encoded
- **Physics-aware**: Mirrors force field design
- **Still rotationally invariant**: Maintains symmetries

**Disadvantages:**
- **Computational cost**: O(N × k²) where k is coordination number
- **Memory**: Stores triplets, not just pairs
- **Complexity**: More hyperparameters to tune

**Use Cases:**
- High-accuracy property prediction
- Conformational energy differences
- Transition state geometries
- Systems where angles matter (chelates, rings, etc.)

#### PaiNN (Polarizable Atom Interaction Neural Network)

PaiNN represents a paradigm shift: instead of just invariant features, it maintains both scalar (invariant) and vector (equivariant) representations.

**Motivation: Vector Properties**

Many important properties are vectors (have direction):
- **Forces**: F = -∇E (gradient of energy)
- **Dipole moments**: μ = Σ_i q_i r_i
- **Magnetic moments**: Direction matters
- **Polarizability**: Tensorial response to fields

Previous models (SchNet, DimeNet):
- Can only predict scalar outputs directly
- Forces require numerical differentiation: F = -dE/dR
- Inefficient and sometimes inaccurate

PaiNN: 
- Predicts vectors directly
- Learns force fields end-to-end
- Truly equivariant architecture

**Equivariance Explained:**

For a rotation matrix Q:
```
Invariant (scalar): f(QR) = f(R)
    Example: ||v|| = ||Qv||  (length unchanged)

Equivariant (vector): f(QR) = Q f(R)
    Example: Qv rotates the same way as input

Equivariant (rank-2 tensor): f(QR) = Q f(R) Q^T
    Example: Stress tensor transforms
```

**Feature Representation:**

Each atom i has TWO types of features:

1. **Scalar features** s_i ∈ ℝ^d:
   - Rotation invariant
   - Examples: atomic charge, energy contribution, electron density
   - Transforms: s_i → s_i (unchanged under rotation)

2. **Vector features** v_i ∈ ℝ^(d×3):
   - Rotation equivariant  
   - Examples: dipole moment, force vector, polarization
   - Transforms: v_i → Q v_i (rotates with molecule)

**Architecture Overview:**

```
Input: (s_i^(0), v_i^(0)) for each atom
       s_i^(0) = embedding(Z_i)  # Initial: just element type
       v_i^(0) = 0                # Initial: no directional info

Message Passing Layers:
    (s_i, v_i) → MessagePass → (s_i', v_i')
    
Output: 
    Scalar: energy = Σ_i MLP(s_i^(T))
    Vector: forces = Σ_i v_i^(T)  # Or per-atom force contribution
```

**Message Passing in PaiNN:**

Each layer consists of three parts:

**Part 1: Scalar Message Passing**

```python
# Compute scalar messages from neighbors
for j in neighbors(i):
    d_ij = ||r_j - r_i||  # Distance
    dir_ij = (r_j - r_i) / d_ij  # Unit direction vector
    
    # Filter based on distance
    φ_ij = FilterNetwork(d_ij)  # Like SchNet
    
    # Scalar message (rotation invariant)
    m_s_ij = φ_ij ⊙ s_j
    
    # Also incorporate magnitude of vector features
    m_s_ij += φ_ij ⊙ ||v_j||²  # Invariant: length squared

# Aggregate
m_s_i = Σ_j m_s_ij

# Update scalars
s_i = s_i + MLP(m_s_i)
```

**Part 2: Vector Message Passing**

This is where equivariance happens:

```python
# Compute vector messages
for j in neighbors(i):
    d_ij = ||r_j - r_i||
    dir_ij = (r_j - r_i) / d_ij  # ← KEY: Direction vector
    
    φ_ij = FilterNetwork(d_ij)
    
    # Vector message (rotation equivariant!)
    # Multiply by direction to make equivariant
    m_v_ij = φ_ij ⊙ v_j  # Element-wise filter
    m_v_ij += (φ_ij ⊙ s_j) × dir_ij  # Scalar-to-vector term
    
# Aggregate vectors
m_v_i = Σ_j m_v_ij

# Update vectors
v_i = v_i + m_v_i
```

**Why this is equivariant:**

```
Under rotation Q:
    dir_ij → Q dir_ij  (direction rotates)
    v_j → Q v_j        (vector features rotate)
    
    m_v_ij = φ_ij ⊙ v_j + (φ_ij ⊙ s_j) × dir_ij
         → φ_ij ⊙ (Q v_j) + (φ_ij ⊙ s_j) × (Q dir_ij)
         = Q [φ_ij ⊙ v_j + (φ_ij ⊙ s_j) × dir_ij]
         = Q m_v_ij  ✓
```

**Part 3: Mixing Scalars and Vectors**

Cross-interactions between scalar and vector features:

```python
# Vector → Scalar: Extract invariant info from vectors
s_i = s_i + MLP(||v_i||)  # Length is invariant
s_i = s_i + MLP(v_i · v_i)  # Dot product is invariant

# Scalar → Vector: Modulate vectors by scalars
v_i = v_i ⊙ σ(U s_i)  # Element-wise gating
```

**Complete Update Equations:**

```
# Scalar update
Δs_i = Σ_j [W_s(d_ij) ⊙ s_j + W_vs(d_ij) ⊙ ||v_j||²]
s_i = s_i + MLP(Δs_i + ||v_i||²)

# Vector update  
Δv_i = Σ_j [W_v(d_ij) ⊙ v_j + W_sv(d_ij) ⊙ s_j ⊙ (r_j - r_i) / d_ij]
v_i = v_i + Δv_i

# Mix scalar and vector
s_i = s_i + U_vs ||v_i||
v_i = (W_vv v_i) ⊙ σ(U_sv s_i)
```

**Output Predictions:**

```python
# Energy (scalar invariant)
E_i = MLP_scalar(s_i)
E_total = Σ_i E_i

# Forces (vector equivariant)
F_i = v_i  # Already in correct format!
# Or: F_i = Linear(v_i) for learned scaling

# Other vector properties
dipole = Σ_i q_i × v_i  # If q_i are charges
```

**Advantages of PaiNN:**

1. **Direct vector prediction**:
   - Forces without numerical differentiation
   - More accurate and efficient
   - Can predict multiple vector properties

2. **True equivariance**:
   - Guarantees physical consistency
   - Rotated inputs → correctly rotated outputs
   - No violation of physics

3. **Expressive representations**:
   - Vectors encode directional information
   - Richer than scalar-only features
   - Better for anisotropic systems

4. **Force field learning**:
   - Can be trained on forces directly
   - Learns better potential energy surfaces
   - Useful for molecular dynamics

**Training Considerations:**

```python
# Loss function combining energy and forces
loss = w_E ||E_pred - E_true||² + w_F ||F_pred - F_true||²

# Typical weights
w_E = 1.0   # Energy in eV or kcal/mol
w_F = 100.0  # Forces in eV/Å (forces are smaller, need higher weight)
```

**Applications:**

1. **Molecular Dynamics**:
   - Learn accurate force fields from DFT
   - 1000× faster than ab initio MD
   - Maintains accuracy of quantum calculations

2. **Transition State Search**:
   - Accurate forces guide optimization
   - Find saddle points efficiently
   - Predict reaction barriers

3. **Dipole Moment Prediction**:
   - Important for spectroscopy
   - Drug-like properties
   - Solvent effects

4. **Polarizability**:
   - Response to external fields
   - Optical properties
   - Intermolecular interactions

**Implementation Notes:**

```python
# Hyperparameters
num_layers = 5
hidden_dim_scalar = 128
hidden_dim_vector = 64  # Vectors have 3× more parameters (x,y,z)
num_rbf = 20
cutoff = 5.0

# Vector features typically smaller dimension to save memory
# v_i ∈ ℝ^(d×3) uses 3× memory of s_i ∈ ℝ^d
```

#### Comparison of Approaches

| Model | Distance | Angles | Equivariance | Outputs | Complexity | Use Case |
|-------|----------|--------|--------------|---------|------------|----------|
| **SchNet** | ✓ | ✗ | Invariant | Scalars | O(N×k) | Fast, general purpose, good baseline |
| **DimeNet** | ✓ | ✓ | Invariant | Scalars | O(N×k²) | High accuracy, angle-dependent properties |
| **DimeNet++** | ✓ | ✓ | Invariant | Scalars | O(N×k²)* | DimeNet with 3-5× speedup |
| **PaiNN** | ✓ | Implicit | Equivariant | Scalars + Vectors | O(N×k) | Forces, vector properties, MD |

*DimeNet++ optimized but same computational complexity

**When to Choose Each:**

**SchNet:**
-  Need fast inference
-  Large molecules (>100 atoms)
-  Don't need highest accuracy
-  Conformational search (many evaluations)
- X Highly angle-dependent properties
- X Need force predictions

**DimeNet/DimeNet++:**
-  Need highest accuracy
-  Angle and dihedral effects important
-  Small to medium molecules (<50 atoms)
-  Property prediction only
- X Computational budget limited
- X Need force predictions
- X Very large molecules

**PaiNN:**
-  Need force predictions
-  Molecular dynamics simulations
-  Vector property prediction
-  Want equivariant representations
-  Good accuracy/speed trade-off
- X Only need scalar properties
- X Maximum simplicity desired

**Practical Decision Tree:**

```
Do you need vector outputs (forces, dipoles)?
├─ Yes → Use PaiNN
└─ No
    │
    └─ Is accuracy critical and dataset small?
        ├─ Yes → Use DimeNet++
        └─ No → Use SchNet (fastest)
```

**Benchmarks (QM9 dataset, HOMO energy):**

```
Method          MAE (meV)   Time/molecule   Parameters
SchNet          41          0.5 ms          600K
DimeNet         33          3.0 ms          2M  
DimeNet++       29          1.2 ms          2M
PaiNN           35          0.8 ms          800K

Note: Exact numbers vary by implementation and hardware
```

---

### 4. Protein-Ligand Interaction Modeling

Understanding how small molecules (ligands) bind to proteins is crucial for drug discovery. GNNs provide powerful tools for modeling these complex interactions, potentially accelerating the drug development pipeline from years to months.

#### The Drug Discovery Challenge

**Traditional Drug Discovery:**

1. **Target Identification**: Identify disease-related protein (months-years)
2. **Hit Discovery**: Screen 10⁵-10⁶ compounds experimentally (months, $$$)
3. **Lead Optimization**: Iteratively improve binding (years, $$$$)
4. **Clinical Trials**: Test in humans (years, $$$$$)

**Total**: 10-15 years, $1-2 billion per drug

**AI-Accelerated Discovery:**

GNNs can predict binding without experiments:
- Virtual screening: 10⁶ compounds in hours
- Structure-based optimization
- Reduced experimental testing
- Faster iteration cycles

**Impact**: Several AI-discovered drugs now in clinical trials

#### Problem Formulation

**Key Tasks:**

1. **Binding Affinity Prediction**: 
   - Predict the strength of protein-ligand binding
   - Metrics: IC₅₀, Ki, Kd, ΔG_bind
   - Range: nM (strong) to mM (weak)
   - Applications: Virtual screening, lead optimization

2. **Binding Pose Prediction**: 
   - Determine the 3D orientation of ligand in binding pocket
   - Must satisfy spatial constraints
   - Account for protein flexibility
   - Applications: Structure-based drug design

3. **Virtual Screening**: 
   - Rank large libraries of compounds
   - Prioritize for experimental testing
   - Requires fast inference (<1s per compound)
   - Applications: Hit discovery, library filtering

4. **Selectivity Prediction**:
   - Binding to target vs off-targets
   - Crucial for drug safety
   - Multi-protein modeling
   - Applications: Toxicity prediction, side effect profiling

**Input Data:**

```
Protein:
  - Sequence: MKTAYIAKQRQ... (amino acid sequence)
  - Structure: 3D coordinates of atoms or residues
  - Features: Secondary structure, surface accessibility, physicochemical properties

Ligand:
  - SMILES: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O (structure string)
  - 3D Conformation: Atomic coordinates
  - Features: Atom types, charges, pharmacophore points

Complex:
  - Binding pose (if known from X-ray crystallography)
  - Interaction types (H-bonds, hydrophobic, π-stacking)
```

#### Representation Strategies

**Challenge**: How to represent a protein-ligand complex as a graph?

**Strategy 1: Separate Protein and Ligand Graphs**

```
   [Protein Graph]    [Ligand Graph]
   Nodes: Residues    Nodes: Atoms
   Edges: Contacts    Edges: Bonds
          ↓                  ↓
      [GNN_prot]        [GNN_lig]
          ↓                  ↓
      h_protein         h_ligand
          └─────→ [Concatenate] ←─────┘
                       ↓
                   [MLP head]
                       ↓
                   Affinity
```

**Protein Graph Construction:**

```python
# Option A: Residue-level (coarse-grained)
for residue in protein.residues:
    node_features = [
        residue.amino_acid_type,     # One-hot: 20 amino acids
        residue.secondary_structure,  # Helix, sheet, coil
        residue.surface_accessibility,
        residue.charge,
        residue.hydrophobicity
    ]
    
# Edges: spatial proximity
for i, j in combinations(residues, 2):
    if distance(i.CA, j.CA) < 10.0:  # C-alpha distance
        add_edge(i, j, distance=distance(i.CA, j.CA))

# Option B: Atom-level (fine-grained)
for atom in protein.atoms:
    node_features = [
        atom.element,
        atom.charge,
        atom.in_backbone,
        atom.in_sidechain
    ]
```

**Ligand Graph Construction:**

```python
# Atoms as nodes
for atom in ligand.atoms:
    node_features = [
        atom.atomic_number,
        atom.formal_charge,
        atom.hybridization,    # sp, sp2, sp3
        atom.is_aromatic,
        atom.num_hydrogens,
        atom.degree,
        atom.chirality
    ]

# Bonds as edges
for bond in ligand.bonds:
    edge_features = [
        bond.bond_type,    # Single, double, triple, aromatic
        bond.is_conjugated,
        bond.is_in_ring,
        bond.stereo        # E/Z, cis/trans
    ]
```

**Pros:**
- Simple architecture
- Can pre-train on protein/ligand datasets separately
- Modular: easy to swap GNN architectures

**Cons:**
- No explicit inter-molecular interactions
- Late fusion may miss important binding details
- Less interpretable (black box combination)

**Strategy 2: Joint Protein-Ligand Interaction Graph**

```
    Protein nodes + Ligand nodes
            ↓
    Intra-molecular edges (protein bonds, ligand bonds)
            +
    Inter-molecular edges (binding interactions)
            ↓
        [Joint GNN]
            ↓
      [Global pooling]
            ↓
         Affinity
```

**Interaction Graph Construction:**

```python
# Combine protein and ligand into one graph
G = Graph()

# Add protein nodes
G.add_nodes(protein_atoms, type='protein')

# Add ligand nodes  
G.add_nodes(ligand_atoms, type='ligand')

# Add intra-molecular edges
G.add_edges(protein_bonds)
G.add_edges(ligand_bonds)

# Add inter-molecular edges (KEY!)
for p_atom in protein_atoms:
    for l_atom in ligand_atoms:
        dist = distance(p_atom, l_atom)
        if dist < 5.0:  # Interaction cutoff
            interaction_type = classify_interaction(p_atom, l_atom, dist)
            G.add_edge(p_atom, l_atom, 
                      distance=dist,
                      interaction=interaction_type)
```

**Interaction Types:**

Classify inter-molecular edges by interaction:

```python
def classify_interaction(p_atom, l_atom, distance):
    interactions = []
    
    # Hydrogen bond
    if is_h_bond_donor(p_atom) and is_h_bond_acceptor(l_atom):
        if distance < 3.5 and angle_ok:
            interactions.append('H-bond')
    
    # Hydrophobic
    if is_hydrophobic(p_atom) and is_hydrophobic(l_atom):
        if distance < 4.5:
            interactions.append('hydrophobic')
    
    # Pi-stacking
    if is_aromatic(p_atom) and is_aromatic(l_atom):
        if 3.5 < distance < 4.5:
            interactions.append('pi-stacking')
    
    # Electrostatic
    if charge(p_atom) * charge(l_atom) < 0:  # Opposite charges
        interactions.append('electrostatic')
    
    # Salt bridge
    if is_charged_residue(p_atom) and is_charged_group(l_atom):
        if distance < 4.0:
            interactions.append('salt-bridge')
    
    return interactions
```

**Pros:**
- Explicit interaction modeling
- Message passing directly between protein and ligand
- More interpretable (can visualize key interactions)
- Better captures binding geometry

**Cons:**
- Larger graphs (more nodes and edges)
- Requires 3D structure (binding pose)
- More complex to implement

**Strategy 3: Attention-Based Cross-Attention**

```
    [Protein GNN] → h_protein_nodes
    [Ligand GNN]  → h_ligand_nodes
          ↓              ↓
    [Cross-Attention Layer]
          ↓
    Attended features
          ↓
    [Prediction head]
```

**Cross-Attention Mechanism:**

```python
# Protein attends to ligand
for p_node in protein_nodes:
    # Compute attention to all ligand nodes
    attention = []
    for l_node in ligand_nodes:
        # Attention score based on features and geometry
        score = attention_function(
            h_protein[p_node], 
            h_ligand[l_node],
            distance(p_node, l_node)
        )
        attention.append(score)
    
    # Softmax normalize
    attention = softmax(attention)
    
    # Attended ligand features
    h_protein[p_node] += weighted_sum(attention, h_ligand)

# Ligand attends to protein (symmetric)
for l_node in ligand_nodes:
    # Similar process in reverse
    ...
```

**Attention Function:**

```python
def attention_function(h_p, h_l, distance):
    # Feature similarity
    feat_sim = dot_product(W_p @ h_p, W_l @ h_l)
    
    # Distance penalty (closer = more attention)
    dist_weight = exp(-distance / sigma)
    
    # Combined score
    score = feat_sim * dist_weight
    return score
```

**Pros:**
- Learns which protein-ligand pairs interact
- Flexible: works without predefined interaction edges
- Can handle multiple binding modes
- Interpretable attention weights

**Cons:**
- O(N_protein × N_ligand) complexity
- May overfit on small datasets
- Requires careful regularization

#### Architecture Patterns

**Pattern 1: Separate Encoding with Late Fusion**

```python
class SeparateFusionModel(nn.Module):
    def __init__(self):
        self.protein_gnn = GNN(protein_features, hidden_dim)
        self.ligand_gnn = GNN(ligand_features, hidden_dim)
        self.fusion_mlp = MLP(2 * hidden_dim, output_dim)
    
    def forward(self, protein_graph, ligand_graph):
        # Encode separately
        h_prot = self.protein_gnn(protein_graph)
        h_lig = self.ligand_gnn(ligand_graph)
        
        # Global pooling
        z_prot = global_mean_pool(h_prot, protein_graph.batch)
        z_lig = global_mean_pool(h_lig, ligand_graph.batch)
        
        # Concatenate and predict
        z = torch.cat([z_prot, z_lig], dim=-1)
        affinity = self.fusion_mlp(z)
        return affinity
```

**Pattern 2: Joint Encoding**

```python
class JointGraphModel(nn.Module):
    def __init__(self):
        self.gnn = GNN(node_features, hidden_dim, num_layers=5)
        self.interaction_embedding = nn.Embedding(num_interactions, edge_dim)
        self.readout = Set2Set(hidden_dim)
        self.predictor = MLP(hidden_dim, 1)
    
    def forward(self, complex_graph):
        # Embed interactions
        edge_attr = self.interaction_embedding(complex_graph.edge_type)
        
        # Joint message passing
        h = self.gnn(complex_graph.x, 
                     complex_graph.edge_index,
                     edge_attr)
        
        # Global pooling
        z = self.readout(h, complex_graph.batch)
        
        # Predict affinity
        affinity = self.predictor(z)
        return affinity
```

**Pattern 3: Cross-Attention**

```python
class CrossAttentionModel(nn.Module):
    def __init__(self):
        self.protein_encoder = GNN(protein_features, hidden_dim)
        self.ligand_encoder = GNN(ligand_features, hidden_dim)
        self.cross_attention = CrossAttentionLayer(hidden_dim)
        self.predictor = MLP(hidden_dim, 1)
    
    def forward(self, protein_graph, ligand_graph, distances):
        # Encode separately
        h_prot = self.protein_encoder(protein_graph)
        h_lig = self.ligand_encoder(ligand_graph)
        
        # Cross-attention
        h_prot_updated, h_lig_updated = self.cross_attention(
            h_prot, h_lig, distances
        )
        
        # Pool both
        z_prot = global_attention_pool(h_prot_updated)
        z_lig = global_attention_pool(h_lig_updated)
        
        # Predict from combined representation
        affinity = self.predictor(z_prot + z_lig)
        return affinity
```

#### Key Considerations

**1. Geometric Information**

3D coordinates are essential for accurate binding prediction:

```python
# Distance features (crucial!)
edge_features = []

for edge in edges:
    i, j = edge
    d = distance(coords[i], coords[j])
    
    # Distance encoding
    rbf = gaussian_rbf(d, centers=[1.0, 2.0, 3.0, 4.0, 5.0])
    edge_features.append(rbf)
    
    # Direction (for equivariant models)
    direction = (coords[j] - coords[i]) / d
    
# Use in GNN
h = GNN(nodes, edges, edge_features)
```

**Why geometry matters:**
- Binding pocket shape determines complementarity
- Hydrogen bonds have geometric constraints (distance + angle)
- Hydrophobic interactions are distance-dependent
- Steric clashes prevent binding

**2. Data Challenges**

**Limited Experimental Data:**

```
Available binding data:
- PDBbind: ~20,000 protein-ligand complexes
- ChEMBL: ~2M bioactivity measurements (but many without structures)
- BindingDB: ~2M Ki/Kd values

Compare to:
- ImageNet: 14M images
- GPT training: trillions of tokens

Challenge: Deep learning typically needs more data
```

**Solutions:**

a) **Data Augmentation:**
```python
# Rotation augmentation
for angle in [0, 90, 180, 270]:
    rotated_complex = rotate(complex, angle)
    train_on(rotated_complex)

# Conformational sampling
for conf in generate_conformations(ligand, n=10):
    augmented_complex = (protein, conf)
    train_on(augmented_complex)

# Noise injection
for noise_level in [0.1, 0.2]:
    noisy_coords = coords + noise_level * random_normal()
    train_on(noisy_coords)
```

b) **Transfer Learning:**
```python
# Pre-train on easier tasks
model.pretrain(
    task='molecular_property_prediction',
    dataset='QM9',  # Millions of molecules
    epochs=100
)

# Fine-tune on binding
model.finetune(
    task='binding_affinity',
    dataset='PDBbind',  # Thousands of complexes
    epochs=50,
    learning_rate=1e-4  # Lower learning rate
)
```

c) **Multi-Task Learning:**
```python
# Learn related tasks simultaneously
loss = (w1 * loss_binding_affinity +
        w2 * loss_binding_pose +
        w3 * loss_protein_function +
        w4 * loss_ligand_properties)

# Shared representations benefit all tasks
# More efficient use of limited data
```

**Missing Structures:**

Many bioactivity measurements lack 3D structures:

```python
# Sequence-only prediction (when no structure)
if protein_structure is None:
    # Use sequence-based protein representation
    h_prot = ProteinLanguageModel(protein_sequence)
else:
    # Use structure-based GNN
    h_prot = ProteinGNN(protein_structure)
```

**3. Interpretability**

Understanding WHY a compound binds is as important as predicting IF it binds:

**Attention Visualization:**

```python
# Extract attention weights
attentions = model.get_attention_weights()

# Identify key protein residues
important_residues = []
for residue, attention in zip(residues, attentions):
    if attention > threshold:
        important_residues.append(residue)

# Visualize in 3D
visualize_protein_ligand(
    protein, 
    ligand,
    highlight_residues=important_residues,
    highlight_atoms=high_attention_ligand_atoms
)
```

**Interaction Decomposition:**

```python
# Analyze contribution of each interaction type
for interaction_type in ['H-bond', 'hydrophobic', 'pi-stack']:
    # Ablation: remove this interaction type
    affinity_without = model.predict(
        complex, 
        exclude_interaction=interaction_type
    )
    
    contribution = affinity_full - affinity_without
    print(f"{interaction_type}: {contribution:.2f} kcal/mol")
```

**Gradient-Based Explanations:**

```python
# Which atoms matter most?
ligand.requires_grad = True
affinity = model(protein, ligand)
affinity.backward()

# Atoms with large gradients are important
importance = ligand.grad.norm(dim=-1)
visualize_atom_importance(ligand, importance)
```

**Applications:**

1. **SAR (Structure-Activity Relationship)**:
   - "This hydrogen bond donor is crucial"
   - "Hydrophobic tail can be modified"
   - Guides medicinal chemistry

2. **Failure Analysis**:
   - "Model focuses on wrong pocket"
   - "Missed key water-mediated H-bond"
   - Improves model training

3. **Knowledge Discovery**:
   - Identify novel binding motifs
   - Understand selectivity patterns
   - Generate hypotheses for experiments

#### State-of-the-Art Models

**EquiBind (2021)**

```
Key Innovation: SE(3)-equivariant blind docking

Architecture:
- Separate protein and ligand encoders
- Equivariant graph neural network (EGNN)
- Predicts keypoint matches
- Direct coordinate prediction (no search)

Performance:
- 38% success rate (<2Å RMSD) on PDBbind
- 1000× faster than traditional docking
- Fully differentiable
```

**GraphDTA / DeepDTA (2018)**

```
Key Innovation: End-to-end learning from sequences

Architecture:
- CNN for protein sequences
- GNN for ligand graphs
- Concatenation + MLP
- Trained on drug-target affinity

Performance:
- Competitive on Davis and KIBA datasets
- Works without 3D structures
- Fast inference for virtual screening
```

**ATOM3D (2021)**

```
Key Innovation: 3D structure benchmarks

Datasets:
- Protein-ligand binding (PDB)
- Protein structure prediction
- RNA structure
- Molecular dynamics

Models:
- SchNet, DimeNet applied to biomolecules
- Transformers with 3D positional encoding
- Graph transformers
```

**TANKBind (2023)**

```
Key Innovation: Equivariant blind docking with diffusion

Architecture:
- Diffusion model for pose generation
- Equivariant score matching
- Iterative refinement
- Confidence estimation

Performance:
- State-of-the-art on blind docking
- Handles protein flexibility
- Generalizes to unseen proteins
```

#### Practical Workflow

```python
# 1. Data preparation
from biopandas.pdb import PandasPdb

# Load protein
protein = PandasPdb().fetch_pdb('1a2b')
protein_graph = protein_to_graph(protein)

# Load ligand
ligand = Chem.MolFromMol2File('ligand.mol2')
ligand_graph = mol_to_graph(ligand)

# 2. Model training
model = ProteinLigandGNN(
    protein_features=37,
    ligand_features=11,
    hidden_dim=128,
    num_layers=4
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    for protein_graph, ligand_graph, affinity in dataloader:
        pred_affinity = model(protein_graph, ligand_graph)
        loss = F.mse_loss(pred_affinity, affinity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 3. Virtual screening
candidates = load_library('drugbank.sdf')  # 10,000 compounds

predictions = []
for ligand in tqdm(candidates):
    ligand_graph = mol_to_graph(ligand)
    affinity = model(protein_graph, ligand_graph)
    predictions.append((ligand, affinity))

# Sort by predicted affinity
predictions.sort(key=lambda x: x[1], reverse=True)

# Test top 100 experimentally
top_hits = predictions[:100]
```

#### Future Directions

1. **Physics-Informed Neural Networks**:
   - Incorporate electrostatics, solvation
   - Constrain predictions with physical laws
   - Hybrid quantum/classical approaches

2. **Generative Models**:
   - Generate ligands for target protein
   - Optimize for multiple objectives
   - De novo drug design

3. **Allostery and Dynamics**:
   - Model protein conformational changes
   - Long-range allosteric effects
   - Molecular dynamics integration

4. **Multi-Target Modeling**:
   - Selectivity across protein families
   - Polypharmacology
   - Side effect prediction

---

### 5. Crystal Structures for Materials Science

GNNs are revolutionizing materials science by predicting properties of crystalline materials from their atomic structures. This enables high-throughput screening of millions of hypothetical materials, dramatically accelerating materials discovery.

#### Understanding Crystalline Materials

**What Makes Crystals Different?**

Unlike molecules, crystals are:
- **Infinite**: Periodic repetition in 3D space
- **Ordered**: Atoms arranged in regular lattices
- **Defined by symmetry**: Space groups and point groups
- **Bulk properties**: Properties of the infinite system, not just a cluster

**Real-World Impact:**

Materials with specific properties enable technology:
- **Batteries**: Li-ion conductors (electric vehicles)
- **Solar cells**: High-efficiency photovoltaics
- **Catalysts**: Green chemical production
- **Semiconductors**: Computing and electronics
- **Superconductors**: Lossless power transmission

**The Discovery Challenge:**

```
Possible stable materials: ~10^50 (combinatorial explosion!)
Known materials: ~200,000 (Materials Project, ICSD)
Fully characterized: ~50,000
Currently used: ~10,000

DFT calculation: 1-1000 CPU-hours per material
ML prediction: 0.001 seconds per material

Speed-up: 10^6× faster!
```

#### Crystal Representation

**Unit Cell Description:**

A crystal is fully specified by:

1. **Lattice Vectors**: Define the unit cell
   ```
   a⃗ = [a_x, a_y, a_z]  # First lattice vector
   b⃗ = [b_x, b_y, b_z]  # Second lattice vector  
   c⃗ = [c_x, c_y, c_z]  # Third lattice vector
   
   Lattice matrix: L = [a⃗ | b⃗ | c⃗]
   ```

2. **Lattice Parameters**:
   ```
   a, b, c = lengths of vectors (Ångströms)
   α, β, γ = angles between vectors (degrees)
   ```

3. **Basis**: Atomic positions within the unit cell
   ```
   Fractional coordinates: (u, v, w) where 0 ≤ u,v,w < 1
   Cartesian coordinates: r⃗ = u·a⃗ + v·b⃗ + w·c⃗
   ```

4. **Space Group**: Symmetry operations
   ```
   230 possible space groups in 3D
   Examples: P21/c (monoclinic), Fm3̄m (cubic), P63/mmc (hexagonal)
   ```

**Example: Diamond (Carbon)**

```
Lattice: Face-centered cubic (FCC)
  a = b = c = 3.567 Å
  α = β = γ = 90°

Basis: Two carbon atoms at
  C₁: (0, 0, 0)
  C₂: (0.25, 0.25, 0.25)

Space group: Fd3̄m (227)

Infinite crystal:
  All positions (n₁, n₂, n₃) + basis
  where n₁, n₂, n₃ ∈ ℤ
```

#### Periodic Boundary Conditions

**The Periodicity Challenge:**

For graph construction, we need neighbors, but:
- Atoms near cell boundaries have neighbors in adjacent cells
- Must account for periodic images
- Same atom appears infinitely many times

**Graph Construction Algorithm:**

```python
def construct_crystal_graph(atoms, lattice, cutoff_radius):
    """
    Build graph respecting periodic boundaries
    """
    graph = Graph()
    
    # Add nodes for atoms in unit cell
    for atom in atoms:
        graph.add_node(
            element=atom.element,
            position=atom.frac_coords,  # Fractional coordinates
            features=get_atom_features(atom)
        )
    
    # Find neighbors using minimum image convention
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            # Check all periodic images of atom_j
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    for n3 in [-1, 0, 1]:
                        # Skip self-loops (unless different cells)
                        if i == j and (n1, n2, n3) == (0, 0, 0):
                            continue
                        
                        # Compute distance through periodic boundaries
                        image_position = atom_j.frac_coords + [n1, n2, n3]
                        cart_position = lattice.to_cartesian(image_position)
                        distance = np.linalg.norm(
                            cart_position - lattice.to_cartesian(atom_i.frac_coords)
                        )
                        
                        if distance < cutoff_radius:
                            graph.add_edge(
                                i, j,
                                distance=distance,
                                cell_offset=(n1, n2, n3),  # Which periodic image
                                direction=cart_position - atom_i.cart_coords
                            )
    
    return graph
```

**Minimum Image Convention:**

```
For each pair of atoms, consider all periodic images
Choose the closest one (minimum distance)

Example: 1D crystal with cell size 10 Å
  Atom A at x=1
  Atom B at x=9
  
  Direct distance: |9-1| = 8 Å
  Through boundary: |9-1-10| = 2 Å  ← MINIMUM (use this!)
  
This handles wrapped distances correctly
```

**Challenges:**

1. **Variable coordination**: Atoms may have different numbers of neighbors
2. **Long-range order**: Some properties depend on distant atoms
3. **Supercell construction**: May need to replicate unit cell for larger cutoffs

#### Property Prediction Tasks

**Electronic Properties**

These determine conducting and optical behavior:

**1. Band Gap (E_g)**
```
Definition: Energy difference between valence and conduction bands
Units: eV
Range: 0 (metal) to >10 eV (insulator)

Applications:
- E_g ≈ 1.3 eV: Solar cells (optimal for sunlight)
- E_g > 5 eV: Transparent insulators (windows)
- E_g = 0: Metals (wires, electrodes)

Prediction challenge: 
- DFT often underestimates (by 30-50%)
- GNNs can learn correction factors
```

**2. Formation Energy (E_f)**
```
Definition: Energy to form compound from elements
Formula: E_f = E_compound - Σ_i n_i × E_element(i)
Units: eV/atom

Applications:
- E_f < 0: Thermodynamically stable
- E_f > 0.1 eV/atom: Likely unstable
- Guides synthesis feasibility

Prediction: Critical for materials discovery
```

**3. Energy Above Hull (E_hull)**
```
Definition: Energy above stable composition convex hull
Measures: Thermodynamic stability relative to competing phases

E_hull = 0: On convex hull (thermodynamically stable)
E_hull < 0.025 eV/atom: Potentially synthesizable
E_hull > 0.1 eV/atom: Very unlikely to be stable

Applications:
- Virtual materials screening
- Stability prediction before synthesis
```

**Mechanical Properties**

Determine material strength and elasticity:

**1. Bulk Modulus (B)**
```
Definition: Resistance to uniform compression
Formula: B = -V (∂P/∂V)_T
Units: GPa
Range: 1 GPa (soft) to 400 GPa (diamond)

Applications:
- High B: Armor, cutting tools
- Low B: Flexible substrates, cushioning
```

**2. Shear Modulus (G)**
```
Definition: Resistance to shear deformation
Units: GPa

Applications:
- G/B ratio indicates ductility
- High G: Stiff materials
- Critical for structural applications
```

**3. Elastic Constants (C_ij)**
```
Tensor: 6×6 matrix (21 independent components for general case)
Symmetry: Reduces components based on crystal system
  - Cubic: 3 independent constants
  - Hexagonal: 5 constants
  - Triclinic: 21 constants

Applications:
- Full mechanical characterization
- Anisotropic behavior prediction
```

**Thermodynamic Properties**

**1. Phonon Properties**
```
Frequency spectrum: Vibrational modes
Heat capacity: C_v(T) from phonons
Thermal expansion: α(T)
Thermal conductivity: κ

Challenge: Requires dynamical matrix (expensive!)
GNN potential: Fast phonon calculations
```

**2. Free Energy**
```
Temperature-dependent stability
Phase transitions
Chemical potential

Applications:
- Phase diagram prediction
- High-temperature materials
```

#### Specialized Architectures

**CGCNN (Crystal Graph Convolutional Neural Networks)**

One of the first GNN architectures specifically designed for crystals.

**Architecture:**

```python
class CGCNNConv(nn.Module):
    """CGCNN convolution layer"""
    def __init__(self, node_features, edge_features, hidden):
        self.node_fc = nn.Linear(2*node_features + edge_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)
    
    def forward(self, x, edge_index, edge_attr):
        # For each edge (i → j)
        i, j = edge_index
        
        # Concatenate: [h_i || h_j || e_ij]
        messages = torch.cat([x[i], x[j], edge_attr], dim=-1)
        
        # Transform and normalize
        messages = self.bn(self.node_fc(messages))
        messages = F.softplus(messages)
        
        # Aggregate to each node
        # Sum over all incoming edges
        x_new = scatter_add(messages, i, dim=0)
        
        return x + x_new  # Residual connection
```

**Key Features:**

1. **Distance-based edge weights**:
   ```python
   def edge_features(distance, cutoff=8.0):
       # Gaussian distance encoding
       centers = torch.linspace(0, cutoff, 100)
       gamma = (centers[1] - centers[0])
       return torch.exp(-gamma * (distance - centers)**2)
   ```

2. **Pooling for crystal-level properties**:
   ```python
   # Average over all atoms
   h_crystal = global_mean_pool(h_atoms, batch)
   
   # Or weighted by composition
   h_crystal = Σ_i (n_i / N_total) × h_i
   ```

3. **Handling variable composition**:
   ```python
   # Works for any stoichiometry
   # Na₂Cl₂: 2 Na nodes, 2 Cl nodes
   # Na₁₀Cl₁₀: 10 Na nodes, 10 Cl nodes
   # Same model, different graph sizes
   ```

**Training:**

```python
# Dataset: Materials Project
dataset = CrystalDataset('materials_project')

model = CGCNN(
    atom_features=92,  # One-hot: 92 elements
    edge_features=100,  # Gaussian RBFs
    hidden_dim=128,
    num_layers=4
)

# Predict formation energy
for crystal_graph in dataloader:
    pred_energy = model(crystal_graph)
    loss = F.mse_loss(pred_energy, crystal_graph.y)
```

**Performance:**
- MAE ≈ 0.04 eV/atom for formation energy
- Handles diverse chemistries (metals, semiconductors, insulators)
- Fast: 1000× faster than DFT

**MEGNet (MatErials Graph Network)**

Multi-level graph network with atom, bond, and global state.

**Three-Level Architecture:**

```
        ┌─────────────┐
        │ Global State│  (lattice, volume, composition)
        │   g ∈ ℝᵈ    │
        └──────┬──────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐    ┌──────▼──────┐
│Atom States│    │ Bond States │
│  v ∈ ℝᵈ   │◄──►│   e ∈ ℝᵈ    │
└───────────┘    └─────────────┘
```

**Update Rules:**

```python
# Bond update: use atoms + global
e_ij' = φ_e([e_ij || v_i || v_j || g])

# Atom update: aggregate bonds + global  
v_i' = φ_v([v_i || Σ_j e_ij' || g])

# Global update: aggregate atoms + bonds
g' = φ_g([g || Σ_i v_i' || Σ_ij e_ij'])
```

**Advantages:**

1. **Multi-scale information**:
   - Local: Atom and bond features
   - Global: Crystal-level properties (volume, space group)
   
2. **Richer representations**:
   - Bonds have their own learned features
   - Global state captures extensive properties

3. **Better for complex properties**:
   - Properties that depend on global structure
   - Example: Thermal conductivity (collective behavior)

**Applications:**

```python
# Multi-task learning
model = MEGNet(tasks=[
    'formation_energy',
    'band_gap',
    'bulk_modulus',
    'shear_modulus'
])

# Shared encoder, separate heads
h_shared = model.encode(crystal)
E_f = model.head_energy(h_shared)
E_g = model.head_gap(h_shared)
B = model.head_bulk(h_shared)
G = model.head_shear(h_shared)
```

**SchNet for Crystals**

Adapting continuous filters for periodic systems.

**Modifications for Periodicity:**

```python
class SchNetCrystal(nn.Module):
    def __init__(self):
        self.rbf_expansion = GaussianRBF(cutoff=5.0)
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock() for _ in range(6)
        ])
    
    def forward(self, crystal):
        # Handle periodic images
        distances, cell_offsets = get_neighbor_distances(
            crystal.frac_coords,
            crystal.lattice,
            cutoff=5.0
        )
        
        # Distance features (same as molecular SchNet)
        edge_features = self.rbf_expansion(distances)
        
        # Message passing with periodic edges
        h = self.embed(crystal.atomic_numbers)
        for block in self.interaction_blocks:
            h = block(h, crystal.edge_index, edge_features)
        
        # Crystal-level pooling
        return global_mean_pool(h, crystal.batch)
```

**Handling Variable Unit Cell Size:**

```python
# Challenge: Different crystals have different numbers of atoms
# Solution: Normalization

# Per-atom properties → extensive
energy_per_atom = total_energy / num_atoms

# Or use composition-weighted features
composition = get_composition(crystal)  # {'Na': 2, 'Cl': 2}
weights = [composition[atom] / sum(composition.values()) 
           for atom in atoms]
h_crystal = Σ_i weights[i] × h_i
```

**Allegro / NequIP**

State-of-the-art equivariant models for materials.

**E(3)-Equivariant Architecture:**

```python
class E3NN_Crystal(nn.Module):
    """Based on e3nn library"""
    def __init__(self):
        # Irreducible representations (irreps)
        # l=0: scalars (rotation invariant)
        # l=1: vectors (rotation equivariant)
        # l=2: rank-2 tensors
        
        self.irreps_in = "92x0e"  # 92 elements, scalar features
        self.irreps_hidden = "128x0e + 64x1o + 32x2e"  # Mixed irreps
        self.irreps_out = "1x0e"  # Energy (scalar)
        
        self.convolution = E3Convolution(
            self.irreps_in, 
            self.irreps_hidden
        )
    
    def forward(self, crystal):
        # Features transform correctly under rotations
        # Scalars: unchanged
        # Vectors: rotate with crystal
        # Tensors: rotate as rank-2 tensors
        ...
```

**Applications:**

1. **Force Field Learning**:
   ```python
   # Train on energy and forces
   energy_pred = model(crystal)
   forces_pred = -autograd.grad(energy_pred, crystal.coords)
   
   loss = (w_E * (energy_pred - energy_true)**2 +
           w_F * (forces_pred - forces_true)**2)
   ```

2. **Molecular Dynamics**:
   ```
   Learned potential: 1000-10000× faster than DFT
   Accuracy: Near DFT quality
   Application: Simulate nanoseconds to microseconds
   ```

3. **Stress Tensor Prediction**:
   ```python
   # Stress tensor: rank-2 equivariant quantity
   stress = model.predict_stress(crystal)
   # Transforms as: σ → Q σ Q^T under rotation Q
   ```

#### Handling Periodicity: Technical Details

**Minimum Image Convention in Practice:**

```python
def minimum_image_distance(frac_coords_i, frac_coords_j, lattice):
    """
    Compute minimum distance through periodic boundaries
    """
    # Difference in fractional coordinates
    d_frac = frac_coords_j - frac_coords_i
    
    # Wrap to [-0.5, 0.5] (minimum image)
    d_frac = d_frac - np.round(d_frac)
    
    # Convert to Cartesian
    d_cart = lattice @ d_frac
    
    # Distance
    return np.linalg.norm(d_cart)
```

**Edge Attributes for Periodic Systems:**

```python
edge_attr = {
    'distance': d_ij,
    'direction': (r_j - r_i) / d_ij,  # Unit vector
    'cell_offset': (n1, n2, n3),      # Which periodic image
    'is_boundary': (n1 != 0 or n2 != 0 or n3 != 0)
}
```

**Lattice as Global Feature:**

```python
# Include lattice parameters as global features
lattice_features = torch.tensor([
    a, b, c,           # Lengths
    alpha, beta, gamma, # Angles
    volume,            # Cell volume
    np.log(volume),    # Log volume (more uniform distribution)
])

# Broadcast to all atoms
for atom in crystal:
    atom.features = torch.cat([atom.features, lattice_features])
```

#### Applications

**Materials Discovery Workflow:**

```python
# 1. Generate candidate structures
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure

candidates = []
for composition in element_combinations:
    for prototype in common_structures:
        structure = substitute(prototype, composition)
        if is_reasonable(structure):  # Basic filters
            candidates.append(structure)

print(f"Generated {len(candidates)} candidates")  # ~10⁶

# 2. Screen with GNN
model = load_model('megnet_bandgap.pt')

promising = []
for structure in tqdm(candidates):
    graph = structure_to_graph(structure)
    E_g = model.predict(graph)
    
    if 1.0 < E_g < 1.5:  # Target range for solar cells
        promising.append((structure, E_g))

print(f"Found {len(promising)} promising candidates")  # ~10³

# 3. Refine with DFT
for structure, E_g_predicted in promising[:100]:  # Top 100
    E_g_dft = run_dft(structure)  # Expensive but accurate
    
    if abs(E_g_dft - E_g_predicted) < 0.2:
        # GNN was accurate!
        save_for_synthesis(structure)

# 4. Experimental synthesis (top 10)
```

**Process Optimization:**

```python
# Predict stability under different conditions
def predict_stability_map(composition):
    structures = generate_polymorphs(composition)
    
    temperatures = np.linspace(300, 2000, 100)  # K
    pressures = np.linspace(1, 100, 50)  # GPa
    
    phase_diagram = np.zeros((len(temperatures), len(pressures)))
    
    for i, T in enumerate(temperatures):
        for j, P in enumerate(pressures):
            energies = [
                model.predict_free_energy(s, T, P) 
                for s in structures
            ]
            phase_diagram[i,j] = np.argmin(energies)
    
    return phase_diagram
```

**Doping and Substitution:**

```python
# Explore chemical substitutions
base_structure = Structure.from_file('LiFePO4.cif')

for dopant in ['Mn', 'Co', 'Ni']:
    for site in iron_sites:
        doped = base_structure.copy()
        doped.replace(site, dopant)
        
        # Predict properties of doped material
        E_g = model.predict(doped, property='band_gap')
        conductivity = model.predict(doped, property='ionic_conductivity')
        
        print(f"Li{dopant}PO4: E_g={E_g:.2f} eV, σ={conductivity:.2e} S/cm")
```

**Catalysis Applications:**

```python
# Surface models for catalysis
def model_surface(bulk_structure, miller_indices=(1,1,1)):
    # Create surface slab
    slab = bulk_structure.get_surface(miller_indices, thickness=4)
    
    # Add adsorbate
    adsorbate = Molecule(['H', 'H'], [[0,0,0], [0,0,0.74]])
    slab_with_ads = add_adsorbate(slab, adsorbate, site='ontop')
    
    # Predict adsorption energy
    E_slab = model.predict_energy(slab)
    E_slab_ads = model.predict_energy(slab_with_ads)
    E_H2 = model.predict_energy(adsorbate)
    
    E_ads = E_slab_ads - E_slab - 0.5 * E_H2
    return E_ads
```

#### Challenges and Future Directions

**Current Limitations:**

1. **Size limitations**: Most models use small unit cells
   - Typical: 10-50 atoms
   - Real systems: Can have 100-1000 atoms (supercells, defects)

2. **Accuracy vs speed trade-off**:
   - GNNs: Fast but ~10% error
   - DFT: Slow but ~1% error
   - Need: Fast AND accurate

3. **Out-of-distribution generalization**:
   - Models trained on known materials
   - May fail on truly novel chemistries
   - Active learning can help

**Emerging Directions:**

1. **Foundation Models**:
   ```
   Pre-train on ALL known structures (~200K)
   Transfer to specific tasks
   Few-shot learning for new properties
   ```

2. **Inverse Design**:
   ```
   Input: Desired properties (E_g=1.3 eV, stable, earth-abundant)
   Output: Candidate structures
   Challenge: Generative models for crystals
   ```

3. **Multi-fidelity Learning**:
   ```
   Low-fidelity: GNN predictions (fast, many)
   High-fidelity: DFT calculations (slow, few)
   Combine: Bayesian optimization, active learning
   ```

4. **Uncertainty Quantification**:
   ```python
   prediction, uncertainty = model.predict_with_uncertainty(structure)
   
   if uncertainty < threshold:
       # High confidence, trust prediction
       use_prediction()
   else:
       # Low confidence, run DFT
       run_dft_calculation()
   ```

---

### 6. Practical: Building a GAT for QM9 Dataset

In this hands-on session, we'll build a Graph Attention Network to predict molecular properties on the QM9 dataset. This practical will cover the complete workflow from data loading to model evaluation and interpretation.

#### Dataset Overview

**QM9 Dataset:**

The QM9 dataset is a quantum chemistry benchmark containing:
- **134,000 small organic molecules** (subset of GDB-17 database)
- **Up to 9 heavy atoms** (C, O, N, F) - excludes hydrogen in counting
- **19 regression targets** computed with Density Functional Theory (DFT)
- **3D molecular geometries** with optimized coordinates

**Target Properties:**

```
Index  Property                          Units        Typical Range
0      Dipole moment                     D            0-5
1      Isotropic polarizability          a₀³          10-100  
2      HOMO energy                       eV           -12 to -5
3      LUMO energy                       eV           -5 to 5
4      HOMO-LUMO gap                     eV           2-15
5      Electronic spatial extent         a₀²          200-1500
6      Zero-point vibrational energy     eV           0.5-3.0
7      Internal energy (0K)              eV           -100 to 0
8      Internal energy (298K)            eV           -100 to 0
9      Enthalpy (298K)                   eV           -100 to 0
10     Free energy (298K)                eV           -100 to 0
11     Heat capacity (298K)              cal/mol/K    10-40
12     Atomization energy (0K)           eV           -100 to 0
13     Atomization energy (298K)         eV           -100 to 0
14-18  Rotational constants A,B,C        GHz          Various
```

**Why QM9 is Important:**

1. **Benchmark standard**: Most papers report QM9 results
2. **Diverse chemistry**: Various functional groups and structures
3. **Ground truth quality**: High-level DFT calculations (B3LYP/6-31G(2df,p))
4. **Moderate size**: Large enough to train deep models, small enough to iterate quickly
5. **Multi-task learning**: 19 targets allow testing generalization

**Data Format:**

```python
# Each molecule has:
molecule = {
    'num_atoms': 18,
    'atomic_numbers': [6, 6, 6, 1, 1, ...],  # Element types
    'positions': [[x1,y1,z1], [x2,y2,z2], ...],  # 3D coordinates (Å)
    'edge_index': [[0,1], [1,2], ...],  # Bond connectivity
    'edge_attr': [[1], [2], ...],  # Bond types (1=single, 2=double, 3=triple)
    'y': [2.7, 48.2, -7.8, ...],  # 19 target properties
}
```

#### Implementation Steps

**Step 1: Data Loading and Preprocessing**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load QM9 dataset
# This will download ~200MB on first run
print("Loading QM9 dataset...")
dataset = QM9(root='./data/QM9')
print(f"Total molecules: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of targets: {dataset.num_tasks}")

# Select target property to predict
# Let's predict HOMO energy (index 2)
target_idx = 2
target_name = 'HOMO_energy'

print(f"\nTarget: {target_name} (index {target_idx})")
print(f"Mean: {dataset.data.y[:, target_idx].mean():.4f}")
print(f"Std: {dataset.data.y[:, target_idx].std():.4f}")

# Examine a sample molecule
sample = dataset[0]
print(f"\nSample molecule:")
print(f"  Number of atoms: {sample.num_nodes}")
print(f"  Number of bonds: {sample.num_edges}")
print(f"  Node features shape: {sample.x.shape}")
print(f"  Edge indices shape: {sample.edge_index.shape}")
print(f"  Target values: {sample.y.shape}")

# Split dataset (80% train, 10% validation, 10% test)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Use specific indices to ensure consistent splits
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

print(f"\nDataset splits:")
print(f"  Train: {len(train_dataset)} molecules")
print(f"  Val: {len(val_dataset)} molecules")
print(f"  Test: {len(test_dataset)} molecules")

# Compute normalization statistics from training set only
# This prevents data leakage
y_train = torch.stack([data.y[target_idx] for data in train_dataset])
target_mean = y_train.mean()
target_std = y_train.std()

print(f"\nNormalization (from train set):")
print(f"  Mean: {target_mean:.4f}")
print(f"  Std: {target_std:.4f}")

# Create data loaders
batch_size = 32

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,  # Shuffle training data
    num_workers=4  # Parallel data loading
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

print(f"\nBatch information:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")
```

**Understanding the Data:**

```python
# Visualize feature distributions
def analyze_dataset(dataset, name='Dataset'):
    """Analyze and visualize dataset statistics"""
    
    # Collect statistics
    num_atoms = [data.num_nodes for data in dataset]
    num_bonds = [data.num_edges // 2 for data in dataset]  # Divide by 2 (undirected)
    
    print(f"\n{name} Statistics:")
    print(f"  Molecules with 5 atoms: {sum(n == 5 for n in num_atoms)}")
    print(f"  Molecules with 9 atoms: {sum(n == 9 for n in num_atoms)}")
    print(f"  Average atoms: {np.mean(num_atoms):.2f}")
    print(f"  Average bonds: {np.mean(num_bonds):.2f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(num_atoms, bins=range(5, 11), edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Atoms')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Molecule Size Distribution')
    axes[0].grid(alpha=0.3)
    
    # Get target values
    targets = [data.y[target_idx].item() for data in dataset]
    axes[1].hist(targets, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel(f'{target_name} (eV)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Target Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{name.lower()}_analysis.png', dpi=150)
    print(f"  Saved: {name.lower()}_analysis.png")

analyze_dataset(train_dataset, 'Training Set')
```

**Step 2: Define GAT Model**

```python
class GATForQM9(nn.Module):
    """
    Graph Attention Network for molecular property prediction
    
    Architecture:
    - Initial embedding layer
    - Multiple GAT layers with multi-head attention
    - Batch normalization and residual connections
    - Global pooling
    - MLP prediction head
    """
    
    def __init__(self, 
                 num_node_features,
                 hidden_dim=64,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.2,
                 target_mean=0.0,
                 target_std=1.0):
        super(GATForQM9, self).__init__()
        
        # Store normalization parameters
        self.register_buffer('target_mean', torch.tensor(target_mean))
        self.register_buffer('target_std', torch.tensor(target_std))
        
        # Initial node embedding
        # Maps input features to hidden dimension
        self.embedding = nn.Linear(num_node_features, hidden_dim)
        self.embedding_bn = nn.BatchNorm1d(hidden_dim)
        
        # GAT layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Input dimension changes after concatenation
            if i == 0:
                in_channels = hidden_dim
            else:
                in_channels = hidden_dim * num_heads
            
            # Last layer averages heads instead of concatenation
            if i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        heads=num_heads,
                        concat=False,  # Average instead of concatenate
                        dropout=dropout,
                        add_self_loops=True,  # Include self-attention
                        bias=True
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.gat_layers.append(
                    GATConv(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        heads=num_heads,
                        concat=True,  # Concatenate attention heads
                        dropout=dropout,
                        add_self_loops=True,
                        bias=True
                    )
                )
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Prediction head
        # Two-layer MLP with ReLU activation
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Batch object with x, edge_index, batch
        
        Returns:
            predictions: Tensor of shape [batch_size]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.embedding(x)
        x = self.embedding_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # GAT layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            # Store input for residual connection
            x_residual = x
            
            # GAT convolution
            x = gat(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            # Residual connection (with projection if dimensions don't match)
            if x_residual.size(1) == x.size(1):
                x = x + x_residual
            elif i > 0:  # Skip for first layer
                # Project residual to match dimensions
                x_residual_proj = self._project_residual(x_residual, x.size(1))
                x = x + x_residual_proj
        
        # Global pooling: aggregate node features to graph-level
        # Mean pooling: intensive property (per-atom average)
        x = global_mean_pool(x, batch)
        
        # Prediction head
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Denormalize predictions
        x = x * self.target_std + self.target_mean
        
        return x.squeeze(-1)  # Remove last dimension [batch, 1] -> [batch]
    
    def _project_residual(self, x_residual, target_dim):
        """Helper to project residual connection"""
        if not hasattr(self, 'residual_projections'):
            self.residual_projections = {}
        
        key = (x_residual.size(1), target_dim)
        if key not in self.residual_projections:
            self.residual_projections[key] = nn.Linear(
                x_residual.size(1), 
                target_dim,
                bias=False
            ).to(x_residual.device)
        
        return self.residual_projections[key](x_residual)

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = GATForQM9(
    num_node_features=dataset.num_features,
    hidden_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.2,
    target_mean=target_mean.item(),
    target_std=target_std.item()
).to(device)

# Count parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params:,}")

# Print model architecture
print("\nModel Architecture:")
print(model)
```

**Step 3: Training Loop with Best Practices**

```python
def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for data in loader:
        data = data.to(device)
        
        # Forward pass
        pred = model(data)
        target = data.y[:, target_idx]
        
        # Compute loss
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
    
    return total_loss / num_samples

def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            target = data.y[:, target_idx]
            
            loss = criterion(pred, target)
            total_loss += loss.item() * data.num_graphs
            
            predictions.append(pred.cpu())
            targets.append(target.cpu())
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    # Compute metrics
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.mean((predictions - targets)**2)).item()
    r2 = r2_score(targets.numpy(), predictions.numpy())
    
    return {
        'loss': total_loss / len(loader.dataset),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,  # Reduce LR by half
    patience=10,  # Wait 10 epochs before reducing
    verbose=True,
    min_lr=1e-6  # Don't go below this
)
criterion = nn.L1Loss()  # MAE loss

# Training loop
num_epochs = 200
best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 30

# Track history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_mae': [],
    'val_r2': [],
    'learning_rate': []
}

print(f"\nStarting training for {num_epochs} epochs...")
print("=" * 70)

for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, device)
    val_loss = val_metrics['loss']
    val_mae = val_metrics['mae']
    val_r2 = val_metrics['r2']
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)
    history['val_r2'].append(val_r2)
    history['learning_rate'].append(current_lr)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae,
        }, 'best_gat_qm9.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch+1:3d}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss:   {val_loss:.4f}  |  MAE: {val_mae:.4f}  |  R²: {val_r2:.4f}')
        print(f'  LR: {current_lr:.2e}  |  Best Val: {best_val_loss:.4f}  |  Patience: {patience_counter}/{early_stop_patience}')
        print("-" * 70)
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        break

print("=" * 70)
print("Training completed!")
```

**Step 4: Model Evaluation and Analysis**

```python
# Load best model
print("\nLoading best model...")
checkpoint = torch.load('best_gat_qm9.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model from epoch {checkpoint['epoch']+1}")
print(f"Best validation MAE: {checkpoint['val_mae']:.4f} eV")

# Evaluate on test set
print("\nEvaluating on test set...")
test_metrics = evaluate(model, test_loader, criterion, device)

print("\nTest Set Results:")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"  MAE:  {test_metrics['mae']:.4f} eV")
print(f"  RMSE: {test_metrics['rmse']:.4f} eV")
print(f"  R²:   {test_metrics['r2']:.4f}")

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MAE)')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# MAE over epochs
axes[0, 1].plot(history['val_mae'], color='green', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE (eV)')
axes[0, 1].set_title('Validation MAE')
axes[0, 1].grid(alpha=0.3)

# R² over epochs
axes[1, 0].plot(history['val_r2'], color='purple', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Validation R²')
axes[1, 0].grid(alpha=0.3)

# Learning rate schedule
axes[1, 1].plot(history['learning_rate'], color='red', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\nSaved: training_curves.png")
```

**Step 5: Visualization and Interpretation**

```python
def plot_predictions(predictions, targets, dataset_name='Test'):
    """Create comprehensive prediction plots"""
    
    predictions = predictions.numpy()
    targets = targets.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Scatter plot: Predicted vs Actual
    axes[0, 0].scatter(targets, predictions, alpha=0.3, s=20)
    
    # Perfect prediction line
    min_val, max_val = min(targets.min(), predictions.min()), max(targets.max(), predictions.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add statistics text
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    axes[0, 0].text(
        0.05, 0.95,
        f'MAE = {mae:.4f} eV\nR² = {r2:.4f}',
        transform=axes[0, 0].transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    axes[0, 0].set_xlabel(f'Actual {target_name} (eV)', fontsize=12)
    axes[0, 0].set_ylabel(f'Predicted {target_name} (eV)', fontsize=12)
    axes[0, 0].set_title(f'{dataset_name} Set: Predictions vs Actual', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Error distribution
    errors = predictions - targets
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0, 1].set_xlabel('Prediction Error (eV)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Error Distribution', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Error vs actual value
    axes[1, 0].scatter(targets, np.abs(errors), alpha=0.3, s=20, color='green')
    axes[1, 0].axhline(mae, color='r', linestyle='--', linewidth=2, label=f'Mean MAE = {mae:.4f}')
    axes[1, 0].set_xlabel(f'Actual {target_name} (eV)', fontsize=12)
    axes[1, 0].set_ylabel('Absolute Error (eV)', fontsize=12)
    axes[1, 0].set_title('Absolute Error vs Actual Value', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Q-Q plot (Quantile-Quantile)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Error Normality)', fontsize=14)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_predictions.png', dpi=150)
    print(f"Saved: {dataset_name.lower()}_predictions.png")

# Plot test set predictions
plot_predictions(
    test_metrics['predictions'],
    test_metrics['targets'],
    dataset_name='Test'
)

# Analyze errors by molecule size
def analyze_errors_by_size(model, loader, device):
    """Analyze how prediction error varies with molecule size"""
    
    model.eval()
    errors_by_size = {i: [] for i in range(5, 10)}
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            target = data.y[:, target_idx]
            errors = torch.abs(pred - target).cpu()
            
            # Group by number of atoms
            for i, error in enumerate(errors):
                num_atoms = (data.batch == i).sum().item()
                if num_atoms in errors_by_size:
                    errors_by_size[num_atoms].append(error.item())
    
    # Plot
    plt.figure(figsize=(10, 6))
    sizes = sorted(errors_by_size.keys())
    mean_errors = [np.mean(errors_by_size[size]) if errors_by_size[size] else 0 for size in sizes]
    std_errors = [np.std(errors_by_size[size]) if errors_by_size[size] else 0 for size in sizes]
    
    plt.errorbar(sizes, mean_errors, yerr=std_errors, marker='o', linewidth=2, markersize=8, capsize=5)
    plt.xlabel('Number of Heavy Atoms', fontsize=12)
    plt.ylabel('Mean Absolute Error (eV)', fontsize=12)
    plt.title('Prediction Error vs Molecule Size', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_by_size.png', dpi=150)
    print("Saved: error_by_size.png")
    
    return errors_by_size

errors_by_size = analyze_errors_by_size(model, test_loader, device)
```

#### Expected Results

**Performance Benchmarks:**

```
For HOMO energy prediction (in meV):

Method          MAE     RMSE    R²      Parameters
Baseline        300     420     0.00    -
Linear          180     250     0.45    10K
MLP             120     170     0.72    50K
GCN             45      65      0.94    500K
GAT (ours)      35-40   50-55   0.96    800K
DimeNet++       25-30   40-45   0.98    2M

Note: Results vary by hyperparameters and random seed
```

**Training Time:**

```
Hardware: Single GPU (V100)
Batch size: 32
Epochs: ~100-150 to convergence

Time per epoch: ~30 seconds
Total training: ~60-90 minutes
Inference: ~1000 molecules/second
```

#### Extension Ideas

**1. Multi-Task Learning**

```python
class MultiTaskGAT(nn.Module):
    """Predict multiple properties simultaneously"""
    def __init__(self, num_tasks=19):
        super().__init__()
        # Shared encoder
        self.encoder = GATEncoder(...)
        
        # Separate heads for each task
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])
    
    def forward(self, data):
        h = self.encoder(data)
        return torch.cat([head(h) for head in self.task_heads], dim=-1)

# Train with multi-task loss
loss = sum(F.mse_loss(pred[:, i], target[:, i]) for i in range(num_tasks))
```

**2. Add 3D Distance Information**

```python
# Add edge features based on 3D distances
edge_attr = []
for edge in data.edge_index.t():
    i, j = edge
    dist = torch.norm(data.pos[i] - data.pos[j])
    
    # Gaussian RBF encoding
    rbf = torch.exp(-(dist - centers)**2 / gamma)
    edge_attr.append(rbf)

# Use in GAT
model = GATConv(edge_dim=num_rbf)  # Enable edge features
```

**3. Attention Visualization**

```python
def visualize_attention(model, molecule_idx):
    """Visualize attention weights for a molecule"""
    
    data = dataset[molecule_idx].to(device)
    model.eval()
    
    # Hook to capture attention
    attention_weights = []
    def hook_fn(module, input, output):
        # GATConv returns (output, attention_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attention_weights.append(output[1].detach().cpu())
    
    # Register hooks on GAT layers
    hooks = []
    for layer in model.gat_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize (requires RDKit or py3Dmol)
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # Highlight high-attention bonds
    mol = Chem.MolFromSmiles(data.smiles)  # If SMILES available
    highlight_bonds = []
    for i, (src, dst) in enumerate(data.edge_index.t()):
        if attention_weights[0][i] > threshold:
            highlight_bonds.append(mol.GetBondBetweenAtoms(int(src), int(dst)).GetIdx())
    
    img = Draw.MolToImage(mol, highlightBonds=highlight_bonds)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'attention_mol_{molecule_idx}.png')
```

**4. Hyperparameter Tuning**

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    
    # Train model
    model = GATForQM9(hidden_dim=hidden_dim, num_heads=num_heads, ...)
    # ... training loop ...
    
    return val_mae

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(f"Best hyperparameters: {study.best_params}")
```

**5. Ensemble Methods**

```python
# Train multiple models with different seeds
models = []
for seed in range(5):
    torch.manual_seed(seed)
    model = GATForQM9(...)
    # ... train model ...
    models.append(model)

# Ensemble prediction
def ensemble_predict(data):
    predictions = [model(data) for model in models]
    return torch.stack(predictions).mean(dim=0)

# Usually 2-5% better than single model
```

This practical provides a complete, production-ready implementation of GAT for molecular property prediction with extensive analysis and interpretation tools.

---

## Summary

Today we covered the fundamentals of graph neural networks and their applications in computational chemistry and materials science:

1. **Message Passing Neural Networks** provide a flexible framework for learning on graphs
2. **Graph Attention Networks** learn adaptive edge weights through attention mechanisms
3. **Equivariant Networks** (SchNet, DimeNet, PaiNN) respect 3D geometric symmetries
4. **Protein-Ligand Modeling** enables AI-driven drug discovery
5. **Crystal Structure Prediction** accelerates materials discovery
6. **Practical Implementation** demonstrated how to build and train GATs on real molecular data

## Additional Resources

**Papers:**
- "Neural Message Passing for Quantum Chemistry" (Gilmer et al., 2017)
- "Graph Attention Networks" (Veličković et al., 2018)
- "SchNet: A continuous-filter convolutional neural network" (Schütt et al., 2017)
- "Directional Message Passing for Molecular Graphs" (Klicpera et al., 2020)
- "Equivariant message passing for the prediction of tensorial properties" (Schütt et al., 2021)

**Tutorials:**
- PyTorch Geometric documentation and tutorials
- Open Catalyst Project tutorials
- Deep Graph Library (DGL) examples

**Datasets:**
- QM9: Small organic molecules
- PCQM4M: Large-scale quantum chemistry dataset
- Materials Project: Inorganic crystals
- PDBbind: Protein-ligand binding affinities
- Open Catalyst: Catalytic materials
