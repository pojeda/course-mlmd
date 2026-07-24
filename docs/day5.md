# Graph Neural Networks and Geometric Deep Learning

This section introduces **Graph Neural Networks (GNNs)**, one of the core techniques in **geometric
deep learning**, and their applications in molecular and materials science. Molecules can naturally
be represented as graphs, where atoms correspond to nodes and chemical bonds correspond to edges.
Unlike traditional neural networks, GNNs learn directly from these graph structures by exchanging
information between neighboring atoms, while preserving the connectivity and relational information
that determine molecular properties.

## 1. Message Passing Neural Networks

Message Passing Neural Networks (MPNNs) provide the foundation for many modern graph neural network
architectures used for molecular property prediction. They offer a unified framework for learning
from graph-structured data, making them particularly well suited for molecular systems, where atoms
and chemical bonds naturally form graphs.

### 1.1 Graph Representation

In molecular graphs, chemical structures are represented as graphs composed of nodes and edges.

#### Nodes

Each node represents an atom and is associated with a feature vector that may include:

* Atomic number
* Formal charge or partial charge
* Hybridization state ($sp$, $sp^2$, $sp^3$)
* Number of bonded hydrogen atoms
* Aromaticity
* Chirality
* Atomic mass
* Degree (number of bonded neighbors)

#### Edges

Each edge represents a chemical bond and may include features such as:

* Bond type (single, double, triple, aromatic)
* Bond length or interatomic distance
* Stereochemistry (cis/trans or E/Z)
* Conjugation
* Ring membership

The graph topology preserves molecular connectivity and encodes the structural relationships that
determine many chemical and physical properties.

### 1.2 Why message passing?

The fundamental idea behind MPNNs is that the properties of an atom depend not only on the atom
itself but also on its local chemical environment.

Message passing is inspired by this principle. During each iteration, neighboring atoms exchange
information, allowing the network to progressively build richer representations of the molecular
structure. As information propagates through the graph, the model learns local bonding patterns,
conjugation effects, and larger structural motifs that influence molecular behavior.

### 1.3 The message passing framework

An MPNN consists of three stages that are repeated over several message-passing iterations.

#### Message phase

At iteration $t$, each node receives messages from its neighboring nodes:

$$
m_v^{(t+1)}
=
\sum_{u\in\mathcal N(v)}
M_t
\left(
h_v^{(t)},
h_u^{(t)},
e_{uv}
\right)
$$

where:

* $h_v^{(t)}$ is the hidden representation of node $v$,
* $h_u^{(t)}$ is the hidden representation of neighboring node $u$,
* $e_{uv}$ represents the edge features,
* $\mathcal N(v)$ denotes the neighbors of node $v$, and
* $M_t$ is a learnable message function.

Intuitively, each node aggregates information from its neighboring nodes and combines it into a
message that summarizes its local chemical environment.

#### Update phase

The node representation is updated using the aggregated message:

$$
h_v^{(t+1)}
=
U_t
\left(
h_v^{(t)},
m_v^{(t+1)}
\right)
$$

where $U_t$ is typically implemented using

* a feedforward neural network,
* a GRU (gated recurrent unit), or
* an LSTM (long short-term memory network).

Each node updates its hidden representation by combining its previous embedding with the aggregated
information received from its neighbors.

#### Aggregation functions

The summation operator in the message phase can be replaced by different aggregation strategies. All
of these aggregate over a node's **neighbors**.

**Sum aggregation**

$$
\sum_{u\in\mathcal N(v)} M_t(\cdot)
$$

* Preserves information about neighborhood size
* Suitable for extensive molecular properties

**Mean aggregation**

$$
\frac{1}{|\mathcal N(v)|} \sum_{u\in\mathcal N(v)} M_t(\cdot)
$$

* Normalizes for variations in node degree
* More stable across graphs of different sizes

**Max aggregation**

$$
\max_{u\in\mathcal N(v)} M_t(\cdot)
$$

* Captures the strongest local features
* May ignore information contributed by less dominant neighbors

**Attention-based aggregation**

$$
\sum_{u\in\mathcal N(v)} \alpha_{vu}\, M_t(\cdot)
$$

where $\alpha_{vu}$ are learnable attention coefficients. This strategy is employed by **Graph
Attention Networks (GATs)** and enables the network to assign different importance to neighboring
nodes.

#### Readout phase

After $T$ message-passing iterations, each node has a learned embedding $h_v^{(T)}$. To predict
graph-level properties, the node embeddings are combined into a single graph representation

$$
h_G = R\left(\{ h_v^{(T)} \}_{v\in V}\right),
$$

where $R$ denotes the readout function — which, unlike the neighbor aggregation above, pools over
**all nodes in the graph**. A prediction is then obtained using $\hat y = f(h_G)$.

**Common readout functions:**

*Sum pooling*

$$
h_G = \sum_{v\in V} h_v^{(T)}
$$

Appropriate for extensive molecular properties; sensitive to molecular size.

*Mean pooling*

$$
h_G = \frac{1}{|V|} \sum_{v\in V} h_v^{(T)}
$$

Produces size-invariant representations; suitable for intensive properties.

*Max pooling*

$$
h_G = \max_{v\in V} h_v^{(T)}
$$

Captures the most prominent node features.

*Set2Set pooling*

A learnable attention-based pooling mechanism that iteratively attends to node embeddings using a
recurrent network, producing expressive graph-level representations.

### 1.4 Receptive fields and network depth

The number of message-passing iterations determines the **receptive field** of each node.

* $T = 1$: information from immediate neighbors
* $T = 2$: information from neighbors-of-neighbors
* $T = k$: information propagates across paths containing up to $k$ edges

Most molecular GNNs use between three and six message-passing layers, providing sufficient receptive
fields for many small and medium-sized molecules.

**Trade-off.** Increasing the number of message-passing layers expands the receptive field, but
excessively deep GNNs may suffer from two well-known problems:

* **Over-smoothing**, where node embeddings become nearly indistinguishable.
* **Over-squashing**, where information from distant nodes is compressed into fixed-size vectors,
  limiting the amount of information that can propagate through the graph.

### 1.5 Key variants of message passing networks

#### Graph Convolutional Networks (GCNs)

Graph Convolutional Networks simplify the message-passing framework by using a normalized adjacency
matrix to aggregate information from neighboring nodes:

$$
H^{(t+1)}
=
\sigma
\left(
\tilde{D}^{-1/2}
\tilde A
\tilde{D}^{-1/2}
H^{(t)}
W^{(t)}
\right),
$$

where:

* $\tilde A = A + I$ is the adjacency matrix with self-loops,
* $\tilde{D}$ is the degree matrix **of $\tilde A$**, that is $\tilde{D}_{ii} = \sum_j \tilde A_{ij}$,
* $W^{(t)}$ is a learnable weight matrix, and
* $\sigma$ is a nonlinear activation function.

It is important that the degree matrix is computed from $\tilde A$ (with self-loops), not from $A$;
this is the Kipf & Welling renormalization.

*Advantages:* computationally efficient, stable normalization, easy to train.

*Limitations:* limited support for edge features, a fixed aggregation strategy, and an inability to
distinguish certain graph structures that are indistinguishable under the Weisfeiler-Lehman graph
isomorphism test.

#### GraphSAGE

GraphSAGE introduces neighborhood sampling, making GNNs scalable to very large graphs. Its update
rule can be written as

$$
h_v^{(t+1)}
=
\sigma
\left(
W
\left[
h_v^{(t)}
\,\Vert\,
\mathrm{AGG}
\left(
\{ h_u^{(t)} : u\in\mathcal N(v) \}
\right)
\right]
\right),
$$

where $\Vert$ denotes vector concatenation.

*Key ideas:* it randomly samples neighboring nodes, supports multiple aggregation strategies, and
enables inductive learning on previously unseen graphs.

*Benefits:* scales efficiently to very large datasets, is suitable for large molecular databases,
and handles graphs of varying sizes without requiring the entire graph to reside in memory.

### 1.6 Neural message passing for quantum chemistry

The original MPNN framework for quantum chemistry (Gilmer et al., 2017) introduced edge-conditioned
message functions:

$$
m_v^{(t+1)}
=
\sum_{u \in \mathcal{N}(v)}
A_t(e_{uv})\, h_u^{(t)}
$$

where $A_t(e_{uv})$ is a neural network that generates edge-specific transformation matrices. Node
updates are commonly performed using GRUs:

$$
h_v^{(t+1)}
=
\text{GRU}
\left(
h_v^{(t)},
m_v^{(t+1)}
\right)
$$

*Advantages:* different bond types can transmit information differently, GRUs improve gradient flow,
and the scheme supports deeper message passing.

**Virtual edges.** Additional edges may connect non-bonded atoms within a spatial cutoff distance to
capture long-range interactions, conformational effects, and weak intermolecular interactions.

**Typical MPNN workflow:**

```python
# Initialize node embeddings
h_v = embedding(x_v)

# Message passing
for t in range(T):
    for edge (v, u):
        m_vu = EdgeNetwork(e_vu) @ h_u
    m_v = sum(m_vu)
    h_v = GRU(h_v, m_v)

# Graph readout
h_G = Readout({h_v})

# Final prediction
y = MLP(h_G)
```

#### Applications in chemistry and materials science

**Quantum mechanical properties.** MPNNs are effective on QM9-style targets such as HOMO/LUMO
energies, internal energy, enthalpy, free energy, heat capacity, and atomization energy. These
properties depend strongly on molecular connectivity and local electronic environments.

**Physical properties.** Solubility, density, melting and boiling points, refractive index, and
viscosity. Incorporating 3D geometry often improves performance.

**Biological activity prediction.** Toxicity, binding affinity, ADMET properties, and blood-brain
barrier permeability — widely used in early-stage drug discovery.

**Reaction outcome prediction.** MPNNs can predict reaction products from reactants and conditions:

$$
\text{Reactant graphs} \rightarrow \text{MPNN} \rightarrow \text{Product distribution}
$$

Reaction conditions such as temperature, solvent, and catalysts can be incorporated as global
features.

**Retrosynthesis planning.**

$$
\text{Target molecule} \rightarrow \text{MPNN} \rightarrow \text{Candidate precursors}
$$

These systems assist chemists in designing synthetic routes.

**Drug discovery and virtual screening.** High-throughput virtual screening, multi-task property
prediction, active learning, and transfer learning. Once trained, these models can evaluate
thousands of molecules per second.

**De novo molecular design.** MPNNs are frequently combined with generative approaches —
reinforcement learning, genetic algorithms, and property-guided generation — to design molecules
with target properties.

#### Limitations and challenges

* **Over-smoothing.** With many message-passing layers, node embeddings can become nearly identical.
  Possible solutions: residual connections, jumping-knowledge networks, and normalization.
* **Limited expressivity.** Some graph structures cannot be distinguished by standard message-passing
  schemes. Possible solutions: higher-order graph representations, more expressive aggregation, and
  subgraph-based methods.
* **Scalability.** Large molecular systems and protein graphs can become expensive. Possible
  solutions: neighbor sampling, hierarchical pooling, and graph coarsening.
* **Lack of 3D geometric information.** Basic MPNNs operate only on graph topology and often ignore
  molecular geometry. Possible solutions: include distances as edge features, or use geometric and
  equivariant architectures such as SchNet, DimeNet, EGNN, and SE(3)-Transformers.

## 2. Graph Attention Networks

Graph Attention Networks (GATs) introduce attention mechanisms to graph learning, allowing the model
to learn which neighbors are most important for each node. This is a significant advance over basic
message passing, where all neighbors contribute equally to a node's update.

### 2.1 Motivation and intuition

In molecular contexts, not all bonds are equally important for determining a property:

* In a large molecule, a specific functional group might dominate reactivity.
* Some atoms are in the "core" structure while others are in peripheral substituents.
* Certain bonds participate in conjugation or resonance, making them more significant.
* In protein-ligand binding, only residues near the binding site matter.

Just as in language, where "bank" means different things in "river bank" versus "savings bank"
depending on context, an atom's role depends on which neighbors are most relevant. GATs allow the
network to automatically learn these context-dependent importance weights.

### 2.2 Attention mechanism

Unlike MPNNs that use fixed aggregation (sum, mean) or hand-crafted edge weights, GATs learn
attention coefficients $\alpha_{ij}$ that adaptively weigh the importance of each neighbor $j$ for
node $i$. The mechanism has three steps.

**Step 1: Compute attention logits**

$$
e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [W h_i \,\|\, W h_j]\right)
$$

Breaking this down:

* $W h_i$ and $W h_j$: transform node features through a shared weight matrix $W$, projecting them
  into a common space where comparisons are meaningful ($\mathbb{R}^{d_{\text{in}}} \to
  \mathbb{R}^{d_{\text{out}}}$).
* $[W h_i \,\|\, W h_j]$: concatenate the transformed features into a pairwise vector of dimension
  $2 d_{\text{out}}$.
* $\mathbf{a}^T[\cdots]$: apply a learned attention vector $\mathbf{a}$ to reduce this to a scalar
  logit $e_{ij}$.
* LeakyReLU: a nonlinearity with a small negative slope (typically $0.2$). The negative slope keeps
  gradients flowing for negative inputs, avoiding the dying-ReLU problem.

The attention logit $e_{ij}$ measures how relevant neighbor $j$ is to node $i$, based on their
feature compatibility.

**Step 2: Normalize to attention coefficients**

$$
\alpha_{ij} = \mathrm{softmax}_j(e_{ij})
= \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

Softmax normalization ensures the weights sum to 1 over all neighbors; only neighbors compete for
attention. Each $\alpha_{ij} \in (0, 1)$ represents the probability-like importance of neighbor $j$.
Softmax is differentiable, creates sharp distinctions, and keeps the aggregated values bounded.

**Step 3: Aggregate with attention weights**

$$
h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j\right)
$$

Each neighbor contributes proportionally to its attention weight, using the same transformation $W$
from Step 1. Here $\sigma$ is typically ELU or ReLU.

**Worked example.** For an atom with three neighbors and logits $e_{i1} = 0.8,\ e_{i2} = 0.3,\
e_{i3} = -0.2$:

1. Compute logits: $e_{i1} = 0.8,\ e_{i2} = 0.3,\ e_{i3} = -0.2$
2. Apply softmax: $\alpha_{i1} = 0.51,\ \alpha_{i2} = 0.31,\ \alpha_{i3} = 0.19$
3. Aggregate: $h_i' = \sigma\left(0.51 \cdot W h_1 + 0.31 \cdot W h_2 + 0.19 \cdot W h_3\right)$

**Key properties:**

* **Self-attention:** self-loops $(i,i)$ can be included so nodes attend to themselves.
* **Asymmetric:** $\alpha_{ij} \ne \alpha_{ji}$ in general.
* **Local:** only direct neighbors are attended to, preserving graph structure.
* **Permutation invariant:** the order of neighbors does not matter.

### 2.3 Multi-head attention

To stabilize learning and capture different types of relationships simultaneously, GATs employ
multiple independent attention mechanisms (heads).

**Multi-head aggregation (hidden layers):**

$$
h_i' = \Big\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j\right)
$$

where $K$ is the number of heads, each head $k$ has its own parameters $W^k$ and $\mathbf{a}^k$,
$\|$ denotes concatenation, and the output dimension is $K \times d_{\text{out}}$.

Different heads can learn complementary attention patterns — for example, one head might focus on
electronegative atoms (polarity), another on aromatic neighbors (conjugation), another on steric
bulk, and another on formal charges. This mirrors how a chemist considers electronics, sterics, and
orbital interactions simultaneously.

**Multi-head averaging (output layer):**

$$
h_i' = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j\right)
$$

Averaging (rather than concatenation) is used only at the **final** layer, so that the output
dimension matches the target and the heads are effectively ensembled. Hidden layers concatenate.

**Implementation considerations:**

```python
# Typical hyperparameters
num_heads = 4-8       # More heads = more capacity but more parameters
hidden_dim = 64-256   # Per-head dimension
dropout = 0.1-0.3     # On attention coefficients (attention dropout)
```

**Computational complexity:** attention computation is $O(|E| \times d_{\text{out}})$ — linear in
the number of edges — and highly parallelizable, since all coefficients are computed independently.

### 2.4 Advantages of GATs

**Adaptive neighborhoods.** Unlike the fixed, degree-based weights of a GCN,

$$
\text{GCN:}\quad h_i' = \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i\, d_j}}\, W h_j,
\qquad
\text{GAT:}\quad h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j,
$$

GAT learns which neighbors matter. In a drug molecule, it can focus on pharmacophore groups,
hydrogen-bond donors/acceptors, and hydrophobic regions while downweighting inert carbon chains.

**Interpretability.** Attention weights $\alpha_{ij}$ can be visualized and interpreted:

```python
attention_weights = model.get_attention()  # shape: [num_edges, num_heads]
mol_graph = molecule.to_graph()
highlight_edges_by_weight(mol_graph, attention_weights, threshold=0.3)
```

This can reveal which atoms influence predictions most, and whether the model has learned chemically
meaningful patterns. As with any attention mechanism, though, these weights are suggestive rather
than definitive explanations, and should be corroborated with other evidence.

**Parallelizability.** All attention coefficients for all edges can be computed in parallel, which is
GPU-friendly and has no sequential dependencies (unlike RNNs).

**Inductive learning.** GATs can generalize to entirely new graph structures — train on small
molecules and test on larger ones — which is critical for novel drug candidates and transfer
learning across datasets.

### 2.5 GAT variants and extensions

**GATv2: more expressive attention.** The original GAT applies the learned vector after a fixed
linear transform, which makes the attention *static*: the ranking of neighbors does not depend on the
query node. GATv2 (Brody, Alon, and Yahav, 2022) moves the nonlinearity so that the attention becomes
*dynamic*:

$$
\text{GAT:}\quad e_{ij} = \mathbf{a}^T [W h_i \,\|\, W h_j],
\qquad
\text{GATv2:}\quad e_{ij} = \mathbf{a}^T\, \text{LeakyReLU}\left(W [h_i \,\|\, h_j]\right).
$$

Applying the nonlinearity after the concatenation yields a strictly more expressive attention
function and fixes a known theoretical limitation of the original GAT. It often improves performance,
though the size of the gain is task-dependent. GATv2 is a good choice for complex molecules with
subtle structural differences, or when the original GAT plateaus.

**Molecular GAT: edge features.** Basic GAT ignores bond attributes (single/double/triple,
stereochemistry). These can be incorporated into the attention logit:

$$
e_{ij} = \mathbf{a}^T [W h_i \,\|\, W h_j \,\|\, E e_{ij}]
$$

where $e_{ij}$ is the edge feature vector (bond type, distance, ring membership) and $E$ is an edge
embedding matrix. This lets attention depend on both node and edge features — distinguishing single
from double bonds, incorporating 3D distances, and representing stereochemistry.

**Graph Transformer Networks.** These extend attention to *all* pairs of nodes, not just neighbors,
computing $\alpha_{ij}$ for every pair $(i,j)$. This captures long-range interactions and allows more
flexible attention patterns, at the cost of $O(|V|^2)$ complexity (versus $O(|E|)$ for GAT) and a
weaker structural inductive bias. Use them when long-range interactions matter (large molecules,
proteins).

### 2.6 Practical tips for using GATs

**Hyperparameter selection:**

```python
# Good starting points
num_layers = 3-5    # Deeper than this risks over-smoothing
num_heads = 4-8     # More heads for complex tasks
hidden_dim = 64-128 # per head
dropout = 0.1-0.3   # Higher dropout for small datasets
learning_rate = 0.001  # Adam optimizer

# For QM9
config = {
    'num_layers': 4,
    'num_heads': 4,
    'hidden_dim': 128,
    'dropout': 0.2,
    'attention_dropout': 0.1  # Separate dropout on attention weights
}
```

**Common pitfalls:**

1. **Forgetting self-loops**: add explicit $(i,i)$ edges or include $h_i$ in the aggregation.
2. **Over-smoothing**: too many layers make all nodes similar.
3. **Attention collapse**: all attention concentrates on one neighbor (use attention dropout).
4. **Memory**: $K$ heads $\times$ $L$ layers can use a lot of GPU memory.

**Best practices:**

```python
h_i' = h_i + GAT(h_i)              # 1. Residual connections mitigate over-smoothing
h_i' = LayerNorm(h_i + GAT(h_i))   # 2. Layer normalization stabilizes training
alpha_ij = Dropout(softmax(e_ij))  # 3. Attention dropout regularizes attention
e_ij = [bond_type, distance, ring_membership]  # 4. Edge features when available
```

## 3. Equivariant Networks for 3D Structures

Geometric deep learning architectures respect the symmetries and geometric properties of 3D molecular
structures, making them particularly powerful for computational chemistry and materials science.
These networks go beyond graph connectivity to use the full 3D geometry of molecules.

### 3.1 Motivation: why geometry matters

Traditional GNNs treat molecular graphs as topological structures, ignoring spatial arrangement:

* Two molecules with the same connectivity but different 3D shapes (stereoisomers) are treated
  identically.
* Bond and dihedral angles carry crucial information.
* Distance-based interactions (van der Waals, electrostatics) are not captured.
* Forces and other vector properties require 3D information.

Consider cis- and trans-2-butene. Both are written as `H3C-CH=CH-CH3` and have identical graph
connectivity, but they differ in 3D structure and in physical properties (melting point, boiling
point, reactivity). A topology-only GNN cannot distinguish them; a plain 2D graph without explicit
E/Z bond annotations cannot either, so the distinguishing information is specifically the 3D geometry.
A geometry-aware network can.

### 3.2 Understanding symmetries

Molecular properties must respect fundamental physical symmetries.

1. **Translation invariance.** Moving the whole molecule does not change its energy or properties;
   only relative positions matter: $E(R + t) = E(R)$ for any translation $t$.
2. **Rotation invariance.** Rotating the molecule does not change scalar properties (energy, dipole
   magnitude): $E(QR) = E(R)$ for any rotation matrix $Q$.
3. **Permutation invariance.** Relabeling atoms does not change properties; node ordering is an
   artifact of the representation.
4. **Reflection.** Chiral molecules break reflection symmetry; achiral molecules satisfy
   $E(R) = E(\text{mirror}(R))$.

**Invariance versus equivariance.** An *invariant* output does not change under a transformation:
energy is a rotation-invariant scalar, $E(QR) = E(R)$. An *equivariant* output transforms
consistently with the input: forces are rotation-equivariant vectors, $F(QR) = Q\, F(R)$ — rotate the
molecule and the forces rotate the same way.

Networks that violate these symmetries treat rotated copies as different molecules, need far more
training data, and can predict unphysical results (energy changing with rotation). Networks that
respect them are more data-efficient, generalize better, and satisfy physical laws by construction.

### 3.3 SchNet (continuous-filter convolutional neural network)

SchNet pioneered continuous convolutions for molecular modeling, treating molecules as continuous 3D
objects rather than discrete graphs. Instead of learning fixed filters for discrete bond types, it
learns continuous functions that depend smoothly on interatomic distances — mirroring the physics,
where interactions depend on distance continuously.

**Architecture overview:**

```
Input:
  - Atomic numbers Z = [Z1, Z2, ..., Z_N]
  - 3D coordinates R = [r1, r2, ..., r_N]

Output:
  - Molecular property (energy, HOMO, etc.)
```

**Step 1: Atomic embeddings.** Each atom type is embedded into a feature space:

```
x_i^(0) = Embedding(Z_i)   # atomic number -> d-dimensional vector
```

**Step 2: Continuous filter generation.** Filters are functions of distance, not learned for discrete
bins:

```
d_ij = ||r_i - r_j||                                  # distance
e_ij = [exp(-(d_ij - mu_k)^2 / sigma^2) for k in K]   # RBF expansion
W_ij = MLP(e_ij)                                       # filter weights
```

**Radial basis functions.** RBFs create a smooth representation of distances. With centers spaced
about $0.5$ Å apart, a comparable width $\sigma \approx 0.5$ Å gives smooth, overlapping basis
functions (a width as small as $0.1$ Å would leave large gaps between centers). For example, at a
distance $d_{ij} = 1.8$ Å with $\sigma = 0.5$ Å:

```
RBF(mu=1.0) = exp(-(1.8-1.0)^2 / 0.25) = exp(-2.56) ~ 0.08
RBF(mu=1.5) = exp(-(1.8-1.5)^2 / 0.25) = exp(-0.36) ~ 0.70
RBF(mu=2.0) = exp(-(1.8-2.0)^2 / 0.25) = exp(-0.16) ~ 0.85
RBF(mu=2.5) = exp(-(1.8-2.5)^2 / 0.25) = exp(-1.96) ~ 0.14
```

This smooth, continuous representation captures nuances between bond lengths, interpolates to unseen
distances, and provides smooth gradients.

**Step 3: Interaction blocks (message passing).**

```
x_i^(t+1) = x_i^(t) + sum_{j != i} x_j^(t) (*) W_ij^(t)
```

where $(*)$ is element-wise (Hadamard) multiplication and $W_{ij}^{(t)}$ is the distance-dependent
filter. A smooth cutoff keeps interactions local without introducing discontinuities:

```
f_cutoff(d) = 0.5 * [cos(pi * d / r_cutoff) + 1]   if d < r_cutoff
              0                                     otherwise
```

The filter network expands the distance in RBFs, passes it through a dense layer and a shifted
softplus nonlinearity, produces the filter, and multiplies by the cutoff. Atom-wise updates aggregate
the filtered neighbor features and add them with a residual connection.

**Step 4: Output modules.** After $T$ interaction blocks:

```
E_i = MLP_atom(x_i^(T))     # per-atom scalar
E_total = sum_i E_i         # size-extensive property
```

**Key design choices:**

* **Distances, not coordinates.** $\|r_i - r_j\|$ is rotation- and translation-invariant, which
  guarantees the network is invariant.
* **Continuous filters.** Adapt smoothly to any distance; better than discrete binning.
* **Shifted softplus.** $\text{ssp}(x) = \log(0.5 e^x + 0.5)$ is smooth everywhere (unlike ReLU),
  which matters for force prediction, since forces are $-\nabla E$ and require smooth gradients.

*Strengths:* efficient (linear in atoms with a cutoff), scalable, accurate, end-to-end
differentiable, and physically motivated.

*Limitations:* distance-only (no explicit angles), isotropic (no directionality), and cannot predict
vector properties directly (forces require gradients).

### 3.4 DimeNet (directional message passing)

DimeNet extends SchNet by incorporating angular information. While distances capture bond lengths,
many properties depend on bond angles (the H-O-H angle in water), dihedral angles, and three-body
interactions. Roughly: SchNet is like a pairwise, distance-only potential, while DimeNet is like a
force field with angle terms.

**Directional messages.** DimeNet uses messages that depend on triplets of atoms $(i, j, k)$ with $j$
as the central atom:

```
                k
               /
              / theta_ijk
             /
            j -------- i
          d_jk      d_ij
```

**Step 1: Distance and angle embeddings.**

```
e_dist(d_ij) = RBF_expansion(d_ij)
theta_ijk    = interior bond angle at the central atom j,
               i.e. the angle between (r_i - r_j) and (r_k - r_j)
e_angle      = SphericalBasisFunctions(theta_ijk)
```

Note that the angle is centered on $j$ and is defined between the two vectors pointing *away from*
$j$, namely $(r_i - r_j)$ and $(r_k - r_j)$. Distances use Gaussian RBFs; angles use spherical basis
functions, which respect the periodicity of angles.

**Step 2: Directional message passing.** Messages depend on geometric triplets:

```
m_ij = sum_{k in N(j), k != i} MessageBlock(d_ij, d_jk, theta_ijk) (*) x_k
```

The message block embeds both distances and the angle, then combines them with a bilinear layer that
learns correlations between distances and angles — for example, that a particular combination of
$d_{ij}$, $d_{jk}$, and $\theta$ signals a strong interaction. This is more expressive than treating
the geometric features independently.

**Step 3: Update.** Aggregate the directional messages and update the node features, typically with a
residual network.

**Why it helps:** angular information distinguishes linear from bent from tetrahedral geometries
(CO$_2$ at $180°$ versus H$_2$O at $104.5°$), captures rotational barriers via dihedral sensitivity,
and models three-body interactions. Despite using angles, DimeNet remains rotationally invariant,
because angles and distances are themselves invariant.

**DimeNet++** optimizes the message aggregation for a 2–5$\times$ speedup at the same or better
accuracy, using shared bilinear layers, efficient triplet enumeration, and a lower memory footprint.

*Advantages:* higher accuracy than SchNet on QM9, captures geometry, physics-aware, still rotationally
invariant. *Disadvantages:* higher computational cost ($O(N k^2)$, with $k$ the coordination number),
higher memory (stores triplets), and more hyperparameters.

### 3.5 PaiNN (polarizable atom interaction neural network)

PaiNN maintains both scalar (invariant) and vector (equivariant) representations, rather than
invariant features alone. This matters because many important properties are vectors: forces
($F = -\nabla E$), dipole moments, and polarizability. Earlier models (SchNet, DimeNet) predict only
scalars and obtain forces by differentiating the energy; PaiNN can predict vectors directly and learn
force fields end-to-end.

**Equivariance.** For a rotation matrix $Q$: an invariant scalar satisfies $f(QR) = f(R)$; an
equivariant vector satisfies $f(QR) = Q f(R)$; and a rank-2 tensor transforms as $f(QR) = Q f(R) Q^T$.

**Feature representation.** Each atom $i$ carries two kinds of features: **scalar features** $s_i \in
\mathbb{R}^d$, which are rotation-invariant and transform as $s_i \to s_i$; and **vector features**
$v_i \in \mathbb{R}^{d \times 3}$, which are rotation-equivariant and transform as $v_i \to Q v_i$.

**Message passing.** Scalar messages are built from invariants such as distances and $\|v_j\|^2$.
Vector messages achieve equivariance by multiplying scalar filters with the unit direction vector
$\hat{r}_{ij} = (r_j - r_i)/\|r_j - r_i\|$ and with the vector features $v_j$. The construction is
equivariant because, under a rotation $Q$, both $\hat{r}_{ij}$ and $v_j$ rotate, so the message
transforms as $m_v \to Q m_v$. Scalar and vector channels then mix through invariant combinations
($\|v_i\|$, $v_i \cdot v_i$) and gating.

**Output predictions.** Energy is read from the scalar channel ($E = \sum_i \text{MLP}(s_i)$), while
vector properties such as forces are read directly from the vector channel.

*Advantages:* direct vector prediction (forces without numerical differentiation), true equivariance,
expressive directional representations, and end-to-end force-field learning.

**Training** typically combines energy and force losses:

```python
loss = w_E * ||E_pred - E_true||^2 + w_F * ||F_pred - F_true||^2
# Forces are smaller in magnitude, so w_F is usually >> w_E
```

*Applications:* machine-learned force fields for molecular dynamics (orders of magnitude faster than
ab initio MD at near-DFT accuracy), transition-state search, dipole prediction, and polarizability.

### 3.6 Comparison of approaches

| Model      | Distance | Angles   | Equivariance | Outputs           | Complexity | Use case                                  |
| ---------- | -------- | -------- | ------------ | ----------------- | ---------- | ----------------------------------------- |
| SchNet     | Yes      | No       | Invariant    | Scalars           | O(N·k)     | Fast, general purpose, good baseline      |
| DimeNet    | Yes      | Yes      | Invariant    | Scalars           | O(N·k²)    | High accuracy, angle-dependent properties |
| DimeNet++  | Yes      | Yes      | Invariant    | Scalars           | O(N·k²)*   | DimeNet with a 2–5× speedup               |
| PaiNN      | Yes      | Implicit | Equivariant  | Scalars + vectors | O(N·k)     | Forces, vector properties, MD             |

\*DimeNet++ is optimized but has the same asymptotic complexity.

**Choosing a model:**

* **SchNet** — fast inference, large molecules, when the highest accuracy is not required, and for
  conformational search with many evaluations. Less suited to strongly angle-dependent properties or
  direct force prediction.
* **DimeNet / DimeNet++** — highest accuracy, angle and dihedral effects, small to medium molecules.
  Less suited to tight compute budgets or very large molecules.
* **PaiNN** — force prediction, molecular dynamics, vector properties, with a good accuracy/speed
  trade-off. Overkill if only scalar properties are needed.

**Decision guide:**

```
Do you need vector outputs (forces, dipoles)?
- Yes -> PaiNN
- No  -> Is accuracy critical and the dataset small?
         - Yes -> DimeNet++
         - No  -> SchNet (fastest)
```

**Illustrative benchmarks (QM9, HOMO energy).** These are approximate values that vary by
implementation and hardware; treat them as rough guidance, not exact figures, and cite the source you
take them from.

```
Method      MAE (meV)   Time/molecule   Parameters
SchNet      ~41         ~0.5 ms         ~600K
DimeNet     ~28         ~3.0 ms         ~2M
DimeNet++   ~25         ~1.2 ms         ~2M
PaiNN       ~35         ~0.8 ms         ~800K
```

## 4. Protein-Ligand Interaction Modeling

Understanding how small molecules (ligands) bind to proteins is central to drug discovery. GNNs
provide powerful tools for modeling these interactions, potentially accelerating the development
pipeline.

### 4.1 The drug discovery challenge

Traditional drug discovery proceeds through target identification, hit discovery (experimentally
screening $10^5$–$10^6$ compounds), lead optimization, and clinical trials, and typically takes 10–15
years at a cost often cited around \$1–2 billion per approved drug. GNNs can predict binding without
experiments: virtual screening of $10^6$ compounds in hours, structure-based optimization, and
reduced experimental testing. Several AI-assisted candidates are now in clinical trials.

### 4.2 Problem formulation

**Key tasks:**

1. **Binding affinity prediction** — the strength of protein-ligand binding (IC$_{50}$, $K_i$, $K_d$,
   $\Delta G_{\text{bind}}$), from nM (strong) to mM (weak). Used in virtual screening and lead
   optimization.
2. **Binding pose prediction** — the 3D orientation of the ligand in the pocket, subject to spatial
   constraints and protein flexibility.
3. **Virtual screening** — ranking large libraries, requiring fast inference.
4. **Selectivity prediction** — binding to the target versus off-targets, crucial for safety.

**Input data:**

```
Protein: sequence, 3D structure, and features
         (secondary structure, surface accessibility, physicochemical properties)
Ligand:  SMILES, 3D conformation, atom types, charges, pharmacophore points
Complex: binding pose (if known) and interaction types (H-bonds, hydrophobic, pi-stacking)
```

### 4.3 Representation strategies

**Strategy 1: separate protein and ligand graphs.** Encode each with its own GNN, then concatenate
the pooled representations and predict with an MLP.

```python
# Protein graph (residue-level, coarse-grained)
for residue in protein.residues:
    node_features = [
        residue.amino_acid_type, residue.secondary_structure,
        residue.surface_accessibility, residue.charge, residue.hydrophobicity
    ]
for i, j in combinations(residues, 2):
    if distance(i.CA, j.CA) < 10.0:
        add_edge(i, j, distance=distance(i.CA, j.CA))

# Ligand graph (atom-level)
for atom in ligand.atoms:
    node_features = [
        atom.atomic_number, atom.formal_charge, atom.hybridization,
        atom.is_aromatic, atom.num_hydrogens, atom.degree, atom.chirality
    ]
for bond in ligand.bonds:
    edge_features = [bond.bond_type, bond.is_conjugated, bond.is_in_ring, bond.stereo]
```

*Pros:* simple, modular, separately pre-trainable. *Cons:* no explicit inter-molecular interactions,
and late fusion may miss binding details.

**Strategy 2: joint protein-ligand interaction graph.** Combine both molecules into one graph, with
intra-molecular edges (bonds) and inter-molecular edges (binding interactions).

```python
G = Graph()
G.add_nodes(protein_atoms, type='protein')
G.add_nodes(ligand_atoms, type='ligand')
G.add_edges(protein_bonds)
G.add_edges(ligand_bonds)

for p_atom in protein_atoms:
    for l_atom in ligand_atoms:
        dist = distance(p_atom, l_atom)
        if dist < 5.0:
            interaction_type = classify_interaction(p_atom, l_atom, dist)
            G.add_edge(p_atom, l_atom, distance=dist, interaction=interaction_type)
```

Inter-molecular edges can be typed by interaction (hydrogen bond, hydrophobic, pi-stacking,
electrostatic, salt bridge), each with its own geometric criteria. *Pros:* explicit interaction
modeling, direct message passing across the interface, more interpretable, and better geometry.
*Cons:* larger graphs, a required 3D pose, and more complexity.

**Strategy 3: attention-based cross-attention.** Encode protein and ligand separately, then let each
attend to the other, weighting pairs by both feature similarity and geometric proximity. *Pros:*
flexible, handles multiple binding modes, and interpretable. *Cons:* $O(N_{\text{protein}} \times
N_{\text{ligand}})$ complexity and a risk of overfitting on small datasets.

### 4.4 Architecture patterns

**Separate encoding with late fusion:**

```python
class SeparateFusionModel(nn.Module):
    def __init__(self):
        self.protein_gnn = GNN(protein_features, hidden_dim)
        self.ligand_gnn = GNN(ligand_features, hidden_dim)
        self.fusion_mlp = MLP(2 * hidden_dim, output_dim)

    def forward(self, protein_graph, ligand_graph):
        h_prot = self.protein_gnn(protein_graph)
        h_lig = self.ligand_gnn(ligand_graph)
        z_prot = global_mean_pool(h_prot, protein_graph.batch)
        z_lig = global_mean_pool(h_lig, ligand_graph.batch)
        return self.fusion_mlp(torch.cat([z_prot, z_lig], dim=-1))
```

**Joint encoding:**

```python
class JointGraphModel(nn.Module):
    def __init__(self):
        self.gnn = GNN(node_features, hidden_dim, num_layers=5)
        self.interaction_embedding = nn.Embedding(num_interactions, edge_dim)
        self.readout = Set2Set(hidden_dim)
        self.predictor = MLP(hidden_dim, 1)

    def forward(self, complex_graph):
        edge_attr = self.interaction_embedding(complex_graph.edge_type)
        h = self.gnn(complex_graph.x, complex_graph.edge_index, edge_attr)
        z = self.readout(h, complex_graph.batch)
        return self.predictor(z)
```

**Cross-attention:**

```python
class CrossAttentionModel(nn.Module):
    def __init__(self):
        self.protein_encoder = GNN(protein_features, hidden_dim)
        self.ligand_encoder = GNN(ligand_features, hidden_dim)
        self.cross_attention = CrossAttentionLayer(hidden_dim)
        self.predictor = MLP(hidden_dim, 1)

    def forward(self, protein_graph, ligand_graph, distances):
        h_prot = self.protein_encoder(protein_graph)
        h_lig = self.ligand_encoder(ligand_graph)
        h_prot, h_lig = self.cross_attention(h_prot, h_lig, distances)
        z_prot = global_attention_pool(h_prot)
        z_lig = global_attention_pool(h_lig)
        return self.predictor(z_prot + z_lig)
```

### 4.5 Key considerations

**Geometric information.** 3D coordinates are essential: pocket shape determines complementarity,
hydrogen bonds have distance and angle constraints, hydrophobic interactions are distance-dependent,
and steric clashes prevent binding. Distances are typically encoded with RBFs and, for equivariant
models, unit direction vectors.

**Data challenges.** Structural binding data is limited (PDBbind has roughly 20,000 complexes;
BindingDB and ChEMBL hold millions of affinity values, but many without structures), which is small
by deep-learning standards. Common remedies: **data augmentation** (rotations, conformer sampling,
coordinate noise); **transfer learning** (pre-train on a large molecular dataset, then fine-tune on
binding data with a lower learning rate); and **multi-task learning** (learn related endpoints
jointly). When 3D structures are missing, sequence-based protein representations (e.g. protein
language model embeddings) can substitute for structure-based GNNs.

**Interpretability.** Understanding *why* a compound binds matters as much as predicting *whether* it
binds. Attention weights can highlight key residues, ablations can estimate the contribution of each
interaction type, and gradient-based explanations can rank important ligand atoms. These support
structure-activity analysis, failure analysis, and hypothesis generation — while remaining suggestive
rather than definitive.

### 4.6 State-of-the-art models

* **EquiBind (2021)** — SE(3)-equivariant blind docking with direct coordinate prediction (no
  search), reported around a 38% success rate (below 2 Å RMSD) on PDBbind and roughly three orders of
  magnitude faster than traditional docking.
* **GraphDTA / DeepDTA (2018)** — end-to-end affinity prediction from sequences and ligand graphs,
  competitive on Davis and KIBA and usable without 3D structures.
* **ATOM3D (2021)** — a suite of 3D biomolecular structure benchmarks with baseline models (SchNet,
  DimeNet, 3D transformers).
* **TANKBind (2023)** — equivariant blind docking with trigonometry-aware pocket prediction and
  confidence estimation.

### 4.7 Practical workflow

```python
from biopandas.pdb import PandasPdb

protein = PandasPdb().fetch_pdb('1a2b')
protein_graph = protein_to_graph(protein)
ligand = Chem.MolFromMol2File('ligand.mol2')
ligand_graph = mol_to_graph(ligand)

model = ProteinLigandGNN(protein_features=37, ligand_features=11,
                         hidden_dim=128, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for protein_graph, ligand_graph, affinity in dataloader:
        pred = model(protein_graph, ligand_graph)
        loss = F.mse_loss(pred, affinity)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

candidates = load_library('drugbank.sdf')
predictions = [(lig, model(protein_graph, mol_to_graph(lig))) for lig in tqdm(candidates)]
predictions.sort(key=lambda x: x[1], reverse=True)
top_hits = predictions[:100]
```

### 4.8 Future directions

Physics-informed models (incorporating electrostatics and solvation), generative models for de novo
design, models of allostery and conformational dynamics, and multi-target modeling for selectivity
and side-effect prediction.

## 5. Crystal Structures for Materials Science

GNNs are transforming materials science by predicting properties of crystalline materials from atomic
structure, enabling high-throughput screening of large numbers of hypothetical materials.

### 5.1 Understanding crystalline materials

Unlike molecules, crystals are infinite (periodic in 3D), ordered (regular lattices), defined by
symmetry (space groups), and characterized by bulk properties of the infinite system. Materials with
specific properties enable technology: Li-ion conductors for batteries, high-efficiency photovoltaics,
catalysts, semiconductors, and superconductors.

The scale of the discovery problem is large:

```
Possible stable materials:  ~10^50 (combinatorial estimate)
Known materials:            ~200,000 (Materials Project, ICSD)
DFT calculation:            1-1000 CPU-hours per material
ML prediction:              ~0.001 s per material
```

### 5.2 Crystal representation

A crystal is fully specified by lattice vectors ($\vec{a}, \vec{b}, \vec{c}$), lattice parameters
(lengths $a, b, c$ and angles $\alpha, \beta, \gamma$), a basis of atomic positions within the unit
cell (fractional coordinates $(u, v, w)$ with $0 \le u, v, w < 1$), and a space group (one of the 230
in 3D).

**Example: diamond.**

```
Lattice: face-centered cubic (FCC)
  a = b = c = 3.567 A,   alpha = beta = gamma = 90 deg

Basis: two carbon atoms
  C1: (0, 0, 0)
  C2: (0.25, 0.25, 0.25)

Space group: Fd-3m (227)
```

### 5.3 Periodic boundary conditions

For graph construction we need neighbors, but atoms near cell boundaries have neighbors in adjacent
cells, so periodic images must be accounted for.

```python
def construct_crystal_graph(atoms, lattice, cutoff_radius):
    graph = Graph()
    for atom in atoms:
        graph.add_node(element=atom.element, position=atom.frac_coords,
                       features=get_atom_features(atom))
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    for n3 in [-1, 0, 1]:
                        if i == j and (n1, n2, n3) == (0, 0, 0):
                            continue
                        image = atom_j.frac_coords + [n1, n2, n3]
                        cart = lattice.to_cartesian(image)
                        d = np.linalg.norm(cart - lattice.to_cartesian(atom_i.frac_coords))
                        if d < cutoff_radius:
                            graph.add_edge(i, j, distance=d, cell_offset=(n1, n2, n3))
    return graph
```

**Minimum image convention.** For each pair, consider all periodic images and choose the closest. In
a 1D cell of length 10 Å with atoms at $x = 1$ and $x = 9$, the direct distance is 8 Å, but through
the boundary it is $|9 - 1 - 10| = 2$ Å — the minimum image, which is the one to use.

### 5.4 Property prediction tasks

**Electronic properties.** Band gap $E_g$ (0 for metals to $>10$ eV for insulators; $\approx 1.3$ eV
is optimal for single-junction solar cells; standard DFT functionals underestimate $E_g$, often by
30–50%); formation energy $E_f = E_{\text{compound}} - \sum_i n_i E_{\text{element}(i)}$ (negative for
thermodynamically stable compounds); and energy above the convex hull $E_{\text{hull}}$ (0 on the
hull; below $\sim 0.025$ eV/atom potentially synthesizable).

**Mechanical properties.** Bulk modulus $B$, shear modulus $G$, and the elastic tensor $C_{ij}$ (with
the number of independent constants set by crystal symmetry — 3 for cubic, 5 for hexagonal, 21 for
triclinic).

**Thermodynamic properties.** Phonon spectra, heat capacity, thermal expansion, thermal conductivity,
and free energy (for phase diagrams and high-temperature stability).

### 5.5 Specialized architectures

**CGCNN (Crystal Graph Convolutional Neural Networks).** One of the first GNN architectures designed
for crystals.

```python
class CGCNNConv(nn.Module):
    def __init__(self, node_features, edge_features, hidden):
        self.node_fc = nn.Linear(2*node_features + edge_features, hidden)
        self.bn = nn.BatchNorm1d(hidden)

    def forward(self, x, edge_index, edge_attr):
        i, j = edge_index
        messages = torch.cat([x[i], x[j], edge_attr], dim=-1)
        messages = F.softplus(self.bn(self.node_fc(messages)))
        x_new = scatter_add(messages, i, dim=0)
        return x + x_new   # residual connection
```

CGCNN uses Gaussian distance encodings for edges and pools over atoms for crystal-level properties,
and it handles any stoichiometry because the graph size adapts to composition. Note that CGCNN folds
the central node in through the residual connection rather than through explicit self-loops, so its
formula differs from the self-loop-based normalization used elsewhere in this chapter. It reaches
roughly 0.04 eV/atom MAE for formation energy and is about three orders of magnitude faster than DFT.

**MEGNet (Materials Graph Network).** A multi-level network with atom, bond, and global state. Update
rules pass information between all three levels:

```
e_ij' = phi_e([e_ij || v_i || v_j || g])
v_i'  = phi_v([v_i || sum_j e_ij' || g])
g'    = phi_g([g || sum_i v_i' || sum_ij e_ij'])
```

The global state captures extensive, crystal-level information (volume, space group), which helps for
properties that depend on global structure (e.g. thermal conductivity).

**SchNet for crystals.** Adapt the continuous filters to periodic systems by computing neighbor
distances through periodic images, then proceeding as in the molecular case. Extensive properties are
handled per atom (e.g. energy per atom).

**Allegro / NequIP.** State-of-the-art E(3)-equivariant models built on irreducible representations
(scalars $l=0$, vectors $l=1$, rank-2 tensors $l=2$), used for machine-learned force fields and
stress-tensor prediction (the stress transforms as $\sigma \to Q \sigma Q^T$).

### 5.6 Handling periodicity: technical details

```python
def minimum_image_distance(frac_i, frac_j, lattice):
    d_frac = frac_j - frac_i
    d_frac = d_frac - np.round(d_frac)   # wrap to [-0.5, 0.5]
    d_cart = lattice @ d_frac
    return np.linalg.norm(d_cart)
```

Edge attributes for periodic systems typically store the distance, the unit direction, the cell
offset $(n_1, n_2, n_3)$, and a boundary flag. Lattice parameters (lengths, angles, volume, and log
volume) can be added as global features.

### 5.7 Applications

A materials-discovery workflow generates candidate structures, screens them with a fast GNN (for
example, filtering for a target band-gap range), refines the most promising with DFT, and finally
sends a small number for experimental synthesis. GNNs are also used for phase-stability mapping across
temperature and pressure, for doping and substitution studies, and for catalysis (surface slabs with
adsorbates to predict adsorption energies).

### 5.8 Challenges and future directions

Current limitations include small unit-cell sizes, the accuracy/speed trade-off (GNNs are fast but
typically around 10% error, DFT is slow but accurate), and out-of-distribution generalization to
novel chemistries. Emerging directions include foundation models pre-trained on all known structures,
inverse design (generating structures for target properties), multi-fidelity learning that combines
cheap GNN predictions with occasional DFT, and uncertainty quantification to decide when to trust a
prediction versus run DFT.

## 6. Practical: Building a GAT for the QM9 Dataset

In this practical example, we build a **Graph Attention Network (GAT)** to predict a quantum-chemical
molecular property from the QM9 dataset using PyTorch Geometric.

QM9 contains small organic molecules with atom-level features, bond connectivity, 3D coordinates, and
several regression targets computed from quantum chemistry. Here we predict the **HOMO energy**, which
is target index 2 in PyTorch Geometric's QM9.

The learning problem is:

$$
f_\theta(G) \rightarrow y
$$

where $G$ is a molecular graph and $y$ is the target molecular property.


checked 12 may 2026
```python
# Graph Attention Network for QM9 Property Prediction
# pip uninstall -y rdkit rdkit-pypi

import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# 1. Reproducibility
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# For stricter GPU determinism (at some cost in speed), also set:
#   torch.backends.cudnn.deterministic = True
#   torch.backends.cudnn.benchmark = False
# Note that exact reproducibility is still not guaranteed across
# different GPUs, CUDA versions, or library versions.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load QM9 Dataset

print("Loading QM9 dataset...")

dataset = QM9(root="./data/QM9")

print(f"Total molecules: {len(dataset)}")
print(f"Node feature dimension: {dataset.num_node_features}")
print(f"Number of targets: {dataset[0].y.shape[-1]}")

target_idx = 2
target_name = "HOMO energy"

sample = dataset[0]

print("\nSample molecule:")
print(f"  Number of atoms: {sample.num_nodes}")
print(f"  Number of edges: {sample.num_edges}")
print(f"  Node feature shape: {sample.x.shape}")
print(f"  Edge index shape: {sample.edge_index.shape}")
print(f"  Target shape: {sample.y.shape}")


# 3. Train / Validation / Test Split

num_molecules = len(dataset)
indices = torch.randperm(num_molecules)

train_size = int(0.8 * num_molecules)
val_size = int(0.1 * num_molecules)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_dataset = dataset[train_indices]
val_dataset = dataset[val_indices]
test_dataset = dataset[test_indices]

print("\nDataset split:")
print(f"  Train: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")
print(f"  Test: {len(test_dataset)}")


# 4. Target normalization (train statistics only)
train_targets = torch.cat([
    data.y[:, target_idx] for data in train_dataset
])

target_mean = train_targets.mean()
target_std = train_targets.std()

print("\nTarget normalization:")
print(f"  Target: {target_name}")
print(f"  Mean: {target_mean:.4f}")
print(f"  Std: {target_std:.4f}")


# 5. DataLoaders
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# 6. Define the GAT model
class GATForQM9(nn.Module):
    """
    Graph Attention Network for QM9 molecular property prediction.

    The model uses:
    - an initial linear node embedding,
    - several GATConv layers,
    - residual connections,
    - global mean pooling,
    - and a final MLP regression head.
    """

    def __init__(
        self,
        num_node_features,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.dropout_rate = dropout

        self.node_embedding = nn.Linear(num_node_features, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True
                )
            )

            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim)
            )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(
            self.node_embedding.weight,
            nonlinearity="relu"
        )
        nn.init.zeros_(self.node_embedding.bias)

        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu"
                )
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        for gat_layer, batch_norm in zip(self.gat_layers, self.batch_norms):
            residual = x

            x = gat_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Residual is dimensionally consistent because concat=False
            # keeps the GATConv output at hidden_dim.
            x = x + residual

        # BatchNorm1d operates over nodes; a batch contains many nodes,
        # so there is no single-sample issue here.
        graph_embedding = global_mean_pool(x, batch)

        prediction = self.regressor(graph_embedding)

        return prediction.view(-1)


model = GATForQM9(
    num_node_features=dataset.num_node_features,
    hidden_dim=128,
    num_heads=4,
    num_layers=4,
    dropout=0.2
).to(device)

num_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

print(f"\nTrainable parameters: {num_parameters:,}")


# 7. Training and evaluation functions
def get_normalized_target(data):
    target = data.y[:, target_idx].view(-1)
    target = target.to(device)

    return (target - target_mean.to(device)) / target_std.to(device)


def denormalize_target(y_normalized):
    return y_normalized * target_std.to(y_normalized.device) + target_mean.to(y_normalized.device)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        pred_normalized = model(data)
        target_normalized = get_normalized_target(data)

        loss = criterion(pred_normalized, target_normalized)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs

    return total_loss / total_graphs


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_graphs = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            pred_normalized = model(data)
            target_normalized = get_normalized_target(data)

            loss = criterion(pred_normalized, target_normalized)

            pred = denormalize_target(pred_normalized)
            target = denormalize_target(target_normalized)

            all_predictions.append(pred.cpu())
            all_targets.append(target.cpu())

            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)

    return {
        "loss": total_loss / total_graphs,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "targets": targets
    }


# 8. Train the model
criterion = nn.L1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

num_epochs = 100
early_stop_patience = 20

best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())
patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "val_mae": [],
    "val_rmse": [],
    "val_r2": [],
    "learning_rate": []
}

print("\nStarting training...")
print("=" * 70)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    val_metrics = evaluate(
        model,
        val_loader,
        criterion
    )

    val_loss = val_metrics["loss"]

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_mae"].append(val_metrics["mae"])
    history["val_rmse"].append(val_metrics["rmse"])
    history["val_r2"].append(val_metrics["r2"])
    history["learning_rate"].append(current_lr)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        # The normalization statistics are saved with the checkpoint so
        # the model can be reloaded and used in a fresh session, where the
        # in-memory target_mean / target_std would not otherwise exist.
        torch.save({
            "epoch": epoch,
            "model_state_dict": best_model_state,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_idx": target_idx,
            "val_loss": val_loss,
            "val_mae": val_metrics["mae"]
        }, "best_gat_qm9.pt")

    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:03d}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE:    {val_metrics['mae']:.4f}")
        print(f"  Val RMSE:   {val_metrics['rmse']:.4f}")
        print(f"  Val R2:     {val_metrics['r2']:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print("-" * 70)

    if patience_counter >= early_stop_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print("Training finished.")


# 9. Test set evaluation
# map_location places the checkpoint on the current device regardless of
# where it was saved. (This controls device placement, not load safety;
# the checkpoint stores only tensors, ints, and floats, so the current
# PyTorch default weights_only=True is appropriate.)
checkpoint = torch.load(
    "best_gat_qm9.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nBest model:")
print(f"  Epoch: {checkpoint['epoch'] + 1}")
print(f"  Validation MAE: {checkpoint['val_mae']:.4f}")

test_metrics = evaluate(
    model,
    test_loader,
    criterion
)

print("\nTest results:")
print(f"  MAE:  {test_metrics['mae']:.4f}")
print(f"  RMSE: {test_metrics['rmse']:.4f}")
print(f"  R2:   {test_metrics['r2']:.4f}")


# 10. Plot training curves
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history["train_loss"], label="Train loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Normalized MAE loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(history["val_mae"])
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Validation MAE")
plt.grid(alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(history["val_r2"])
plt.xlabel("Epoch")
plt.ylabel("R2")
plt.title("Validation R2")
plt.grid(alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(history["learning_rate"])
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.yscale("log")
plt.title("Learning rate schedule")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gat_qm9_training_curves.png", dpi=150)
plt.show()


# 11. Plot predictions
predictions = test_metrics["predictions"]
targets = test_metrics["targets"]

plt.figure(figsize=(6, 6))

plt.scatter(targets, predictions, alpha=0.4)

min_value = min(targets.min(), predictions.min())
max_value = max(targets.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(f"True {target_name}")
plt.ylabel(f"Predicted {target_name}")
plt.title(f"GAT predictions on QM9: {target_name}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gat_qm9_predictions.png", dpi=150)
plt.show()
```

Notes on this example:

* The model predicts **normalized targets**, and metrics are computed after denormalization.
* Residual connections are dimensionally consistent because all GAT layers use `concat=False`.
* The best checkpoint is saved from a deep copy of the `state_dict`, together with the target
  normalization statistics, so the model can be reloaded in a fresh session.
* `torch.load(..., map_location=device)` loads the checkpoint onto the current device regardless of
  where it was saved. This controls device placement, not load safety; load safety is governed
  separately by the `weights_only` argument.
* The loss is averaged by the number of graphs, not by the number of batches.
* `num_workers=0` is used for portability across notebooks, Windows, and macOS. If you increase it,
  seed the workers with a `worker_init_fn` to preserve reproducibility.


checked. 14 may 2026
```python
# https://se.mathworks.com/help/deeplearning/ug/node-classification-using-graph-convolutional-network.html
# GCN ON QM7 DATASET: ATOM LABEL PREDICTION FROM COULOMB MATRICES

import os
import urllib.request
import tempfile
import numpy as np
import scipy.io
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# 1. DOWNLOAD AND LOAD QM7 DATA

data_url = "https://quantum-machine.org/data/qm7.mat"
output_folder = os.path.join(tempfile.gettempdir(), "qm7Data")
data_file = os.path.join(output_folder, "qm7.mat")

os.makedirs(output_folder, exist_ok=True)

if not os.path.exists(data_file):
    print("Downloading QM7 data...")
    urllib.request.urlretrieve(data_url, data_file)
    print("Done.")

data = scipy.io.loadmat(data_file)

# QM7 Coulomb matrices
# Shape is usually: (7165, 23, 23)
X_raw = data["X"]

# Atomic numbers
# Shape is usually: (7165, 23)
Z_raw = data["Z"]

print("Raw Coulomb matrix shape:", X_raw.shape)
print("Raw atom data shape:", Z_raw.shape)

# Convert to MATLAB-like format:
# MATLAB used permute(data.X, [2 3 1])
# Python version: (num_molecules, 23, 23) -> (23, 23, num_molecules)
coulomb_data = np.transpose(X_raw, (1, 2, 0)).astype(np.float64)

# Sort atomic numbers in descending order, like MATLAB sort(..., 'descend')
atom_data = -np.sort(-Z_raw, axis=1)

num_molecules = coulomb_data.shape[2]

print("Number of molecules:", num_molecules)
print("Coulomb data shape:", coulomb_data.shape)
print("Atom data shape:", atom_data.shape)


# 2. HELPER FUNCTIONS

def atomic_symbol(atomic_numbers):
    """
    Convert atomic numbers to atomic symbols.
    QM7 mainly contains H, C, N, O, and S.
    """

    symbol_map = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
        16: "S"
    }

    atomic_numbers = np.asarray(atomic_numbers).astype(int)

    return np.array([
        symbol_map.get(z, str(z)) for z in atomic_numbers
    ])


def coulomb_to_adjacency(coulomb_data, atom_data):
    """
    Convert Coulomb matrices into adjacency matrices.

    In a Coulomb matrix:
    - diagonal entries represent atom-specific nuclear terms,
    - off-diagonal entries represent pairwise Coulomb interactions.

    Here, an edge is created whenever an off-diagonal Coulomb entry is nonzero.
    """

    max_nodes = coulomb_data.shape[0]
    num_molecules = coulomb_data.shape[2]

    adjacency_data = np.zeros_like(coulomb_data, dtype=np.float32)

    for i in range(num_molecules):
        atomic_numbers = atom_data[i]
        num_nodes = np.count_nonzero(atomic_numbers)

        C = coulomb_data[:num_nodes, :num_nodes, i]

        A = (np.abs(C) > 0).astype(np.float32)

        # Remove self-loops here.
        # Self-loops are added later during GCN normalization.
        np.fill_diagonal(A, 0.0)

        adjacency_data[:num_nodes, :num_nodes, i] = A

    return adjacency_data


def normalize_adjacency(A):
    """
    Compute normalized adjacency:

        A_norm = D^{-1/2} (A + I) D^{-1/2}

    This is the standard normalization used in many GCN models.
    """

    A = A + sp.eye(A.shape[0], dtype=np.float32)

    degree = np.asarray(A.sum(axis=1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    D_inv_sqrt = sp.diags(degree_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm.tocoo()


def scipy_sparse_to_torch_sparse(matrix, device):
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.
    """

    matrix = matrix.tocoo()

    indices = torch.tensor(
        np.vstack((matrix.row, matrix.col)),
        dtype=torch.long,
        device=device
    )

    values = torch.tensor(
        matrix.data,
        dtype=torch.float32,
        device=device
    )

    shape = torch.Size(matrix.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def preprocess_data(adjacency_data, coulomb_data, atom_data):
    """
    Convert a collection of molecular graphs into one block-diagonal graph.

    Returns:
        A_block:
            Sparse block-diagonal adjacency matrix.

        X:
            Node feature matrix.
            Each node uses the diagonal Coulomb value as its feature.

        labels:
            Atomic-number labels for each node.
    """

    adjacency_blocks = []
    features = []
    labels = []

    num_molecules = adjacency_data.shape[2]

    for i in range(num_molecules):
        atomic_numbers = atom_data[i]
        num_nodes = np.count_nonzero(atomic_numbers)

        if num_nodes == 0:
            continue

        A = adjacency_data[:num_nodes, :num_nodes, i]
        C = coulomb_data[:num_nodes, :num_nodes, i]

        # Node feature: diagonal Coulomb value
        # For atom with nuclear charge Z:
        # diagonal approximately equals 0.5 * Z^2.4
        X = np.diag(C).reshape(-1, 1)

        adjacency_blocks.append(sp.csr_matrix(A))
        features.append(X)
        labels.append(atomic_numbers[:num_nodes])

    A_block = sp.block_diag(adjacency_blocks, format="csr")
    X = np.vstack(features).astype(np.float32)
    labels = np.concatenate(labels).astype(int)

    return A_block, X, labels

# 3. CONVERT COULOMB MATRICES TO GRAPH ADJACENCY MATRICES

adjacency_data = coulomb_to_adjacency(coulomb_data, atom_data)

print("Adjacency data shape:", adjacency_data.shape)


# 4. VISUALIZE A FEW MOLECULES

plt.figure(figsize=(10, 10))

for i in range(9):
    atomic_numbers = atom_data[i]
    num_nodes = np.count_nonzero(atomic_numbers)

    A = adjacency_data[:num_nodes, :num_nodes, i]
    symbols = atomic_symbol(atomic_numbers[:num_nodes])

    G = nx.from_numpy_array(A)

    plt.subplot(3, 3, i + 1)
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={j: symbols[j] for j in range(num_nodes)},
        node_color="lightblue",
        node_size=600,
        font_size=10
    )

    plt.title(f"Molecule {i + 1}")

plt.tight_layout()
plt.show()


# 5. PLOT ATOM LABEL FREQUENCIES

all_atomic_numbers = atom_data[atom_data > 0].astype(int)
all_symbols = atomic_symbol(all_atomic_numbers)

unique_symbols, counts = np.unique(all_symbols, return_counts=True)

plt.figure(figsize=(6, 4))
plt.bar(unique_symbols, counts)
plt.xlabel("Atom Label")
plt.ylabel("Frequency")
plt.title("Atom Label Counts in QM7")
plt.tight_layout()
plt.show()


# 6. TRAIN / VALIDATION / TEST SPLIT

indices = np.arange(num_molecules)

idx_train, idx_temp = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

idx_val, idx_test = train_test_split(
    idx_temp,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

print("Train molecules:", len(idx_train))
print("Validation molecules:", len(idx_val))
print("Test molecules:", len(idx_test))


adjacency_train = adjacency_data[:, :, idx_train]
adjacency_val = adjacency_data[:, :, idx_val]
adjacency_test = adjacency_data[:, :, idx_test]

coulomb_train = coulomb_data[:, :, idx_train]
coulomb_val = coulomb_data[:, :, idx_val]
coulomb_test = coulomb_data[:, :, idx_test]

atom_train = atom_data[idx_train, :]
atom_val = atom_data[idx_val, :]
atom_test = atom_data[idx_test, :]


# 7. PREPROCESS TRAINING AND VALIDATION DATA

A_train, X_train, labels_train = preprocess_data(
    adjacency_train,
    coulomb_train,
    atom_train
)

A_val, X_val, labels_val = preprocess_data(
    adjacency_val,
    coulomb_val,
    atom_val
)

# Normalize features using training statistics only.
mu_x = X_train.mean(axis=0, keepdims=True)
std_x = X_train.std(axis=0, keepdims=True)

std_x[std_x == 0] = 1.0

X_train = (X_train - mu_x) / std_x
X_val = (X_val - mu_x) / std_x


# 8. ENCODE ATOM LABELS AS CLASS INDICES

classes = np.unique(labels_train)
class_to_idx = {atomic_number: i for i, atomic_number in enumerate(classes)}
idx_to_class = {i: atomic_number for atomic_number, i in class_to_idx.items()}

y_train = np.array([class_to_idx[z] for z in labels_train], dtype=np.int64)
y_val = np.array([class_to_idx[z] for z in labels_val], dtype=np.int64)

class_names = atomic_symbol(classes)

print("Classes:", classes)
print("Class names:", class_names)


# 9. DEFINE GCN MODEL

class SimpleGCN(nn.Module):
    """
    Simple Graph Convolutional Network for node classification.

    The model predicts the atom type of each node using:
    - the molecular graph structure,
    - the diagonal Coulomb feature of each atom.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)

    def graph_conv(self, A_norm, X, layer):
        """
        One GCN operation:

            H = A_norm X W

        where:
        - A_norm is the normalized adjacency matrix,
        - X is the node feature matrix,
        - W is a trainable weight matrix.
        """

        X = torch.sparse.mm(A_norm, X)
        X = layer(X)

        return X

    def forward(self, X, A_norm):
        H1 = self.graph_conv(A_norm, X, self.linear1)
        H1 = F.relu(H1)

        H2 = self.graph_conv(A_norm, H1, self.linear2)
        H2 = F.relu(H2)

        logits = self.graph_conv(A_norm, H2, self.linear3)

        return logits


# 10. MOVE DATA TO PYTORCH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

A_train_norm = normalize_adjacency(A_train)
A_val_norm = normalize_adjacency(A_val)

A_train_torch = scipy_sparse_to_torch_sparse(A_train_norm, device)
A_val_torch = scipy_sparse_to_torch_sparse(A_val_norm, device)

X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
X_val_torch = torch.tensor(X_val, dtype=torch.float32, device=device)

y_train_torch = torch.tensor(y_train, dtype=torch.long, device=device)
y_val_torch = torch.tensor(y_val, dtype=torch.long, device=device)


# 11. TRAIN MODEL

input_dim = X_train.shape[1]
hidden_dim = 32
num_classes = len(classes)

model = SimpleGCN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-5
)

criterion = nn.CrossEntropyLoss()

num_epochs = 1500
validation_frequency = 300

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()

    optimizer.zero_grad()

    logits_train = model(X_train_torch, A_train_torch)

    loss_train = criterion(logits_train, y_train_torch)

    loss_train.backward()

    optimizer.step()

    train_losses.append(loss_train.item())

    if epoch == 1 or epoch % validation_frequency == 0:
        model.eval()

        with torch.no_grad():
            logits_val = model(X_val_torch, A_val_torch)
            loss_val = criterion(logits_val, y_val_torch)

        val_losses.append((epoch, loss_val.item()))

        print(
            f"Epoch {epoch:4d}/{num_epochs} | "
            f"Training Loss: {loss_train.item():.4f} | "
            f"Validation Loss: {loss_val.item():.4f}"
        )


# 12. PLOT TRAINING CURVES

plt.figure(figsize=(7, 4))

plt.plot(
    np.arange(1, num_epochs + 1),
    train_losses,
    label="Training Loss"
)

if len(val_losses) > 0:
    val_epochs, val_values = zip(*val_losses)

    plt.plot(
        val_epochs,
        val_values,
        marker="o",
        label="Validation Loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("GCN Training Progress")
plt.legend()
plt.tight_layout()
plt.show()


# 13. TEST SET EVALUATION

A_test, X_test, labels_test = preprocess_data(
    adjacency_test,
    coulomb_test,
    atom_test
)

X_test = (X_test - mu_x) / std_x

# Keep only test labels that are known from training.
# This is usually all labels for QM7.
known_mask = np.isin(labels_test, classes)

X_test = X_test[known_mask]
labels_test = labels_test[known_mask]

y_test = np.array(
    [class_to_idx[z] for z in labels_test],
    dtype=np.int64
)

A_test_norm = normalize_adjacency(A_test)
A_test_torch = scipy_sparse_to_torch_sparse(A_test_norm, device)

X_test_torch = torch.tensor(
    X_test,
    dtype=torch.float32,
    device=device
)

y_test_torch = torch.tensor(
    y_test,
    dtype=torch.long,
    device=device
)

model.eval()

with torch.no_grad():
    logits_test = model(X_test_torch, A_test_torch)
    pred_test = logits_test.argmax(dim=1).cpu().numpy()

accuracy = accuracy_score(y_test, pred_test)

print(f"\nTest accuracy: {accuracy:.4f}")


# 14. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    pred_test,
    labels=np.arange(num_classes)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("GCN QM7 Confusion Matrix")
plt.tight_layout()
plt.show()


# 15. PREDICT ATOM LABELS FOR NEW MOLECULES

def predict_molecule_atoms(model, coulomb_matrix, adjacency_matrix, mu_x, std_x):
    """
    Predict atom labels for a single molecule.
    """

    num_nodes = np.where(adjacency_matrix.any(axis=0))[0]

    if len(num_nodes) == 0:
        return []

    num_nodes = num_nodes[-1] + 1

    A = adjacency_matrix[:num_nodes, :num_nodes]
    C = coulomb_matrix[:num_nodes, :num_nodes]

    X = np.diag(C).reshape(-1, 1)
    X = (X - mu_x) / std_x

    A_sparse = sp.csr_matrix(A)
    A_norm = normalize_adjacency(A_sparse)

    A_torch = scipy_sparse_to_torch_sparse(A_norm, device)

    X_torch = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    model.eval()

    with torch.no_grad():
        logits = model(X_torch, A_torch)
        pred = logits.argmax(dim=1).cpu().numpy()

    atomic_numbers = np.array([idx_to_class[i] for i in pred])

    return atomic_symbol(atomic_numbers)


num_observations_new = 4

plt.figure(figsize=(10, 4))

for i in range(num_observations_new):
    A_full = adjacency_test[:, :, i]
    C_full = coulomb_test[:, :, i]

    predictions = predict_molecule_atoms(
        model,
        C_full,
        A_full,
        mu_x,
        std_x
    )

    num_nodes = len(predictions)
    A = A_full[:num_nodes, :num_nodes]

    G = nx.from_numpy_array(A)

    plt.subplot(1, num_observations_new, i + 1)

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={j: predictions[j] for j in range(num_nodes)},
        node_color="lightgreen",
        node_size=600,
        font_size=10
    )

    plt.title(f"Prediction {i + 1}")

plt.tight_layout()
plt.show()
```

1. Downloads `qm7.mat`.
2. Loads Coulomb matrices and atomic numbers.
3. Converts Coulomb matrices into adjacency matrices.
4. Builds one block-diagonal graph for all molecules in each split.
5. Uses diagonal Coulomb values as node features.
6. Trains a simple GCN for atom-label classification.
7. Evaluates the model with accuracy and a confusion matrix.
8. Predicts atom labels for new test molecules.

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
