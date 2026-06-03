# Day 4: Generative Models for Molecular Design

## Overview

Generative models have revolutionized computational drug discovery and molecular design by enabling the creation of novel molecular structures with desired properties. This session explores state-of-the-art generative approaches and their application to molecular generation.

## Learning Objectives

By the end of this session, you will be able to:
- Understand the principles behind various generative model architectures
- Apply VAEs and GANs to molecular generation tasks
- Implement autoregressive models for sequential molecular design
- Utilize diffusion models for high-quality molecule generation
- Integrate reinforcement learning for property optimization
- Develop conditional generation systems for targeted molecular design
- Build and train a conditional VAE for molecular generation

---

## 1. VAEs and GANs for Molecular Generation

### Variational Autoencoders (VAEs)

Variational Autoencoders learn a continuous latent representation of molecules, enabling smooth interpolation and sampling of chemical space.

#### Architecture

The VAE consists of two main components:

**Encoder**: Maps molecules to a probabilistic latent space
- Input: SMILES string or molecular graph
- Output: Mean (μ) and log-variance (log σ²) of latent distribution
- Typically uses RNN/GRU/LSTM or Graph Neural Networks

**Decoder**: Reconstructs molecules from latent representations
- Input: Sampled latent vector z ~ N(μ, σ²)
- Output: Reconstructed molecular representation
- Generates SMILES character-by-character or reconstructs graph

#### Loss Function

The VAE optimizes two objectives:

```
L_VAE = L_reconstruction + β × L_KL
```

- **Reconstruction Loss**: Ensures accurate molecule reconstruction (e.g., cross-entropy for SMILES)
- **KL Divergence**: Regularizes latent space to follow standard normal distribution N(0, I)
- **β parameter**: Controls the trade-off between reconstruction quality and latent space regularity

#### Molecular Representations

Common representations for VAE-based molecular generation:

1. **SMILES-based**: Character-level sequence modeling
2. **Graph-based**: Node and edge features with graph autoencoders
3. **Fingerprint-based**: Fixed-length binary or count vectors
4. **3D conformations**: Atomic coordinates and features

#### Advantages
- Continuous latent space enables interpolation between molecules
- Sampling generates diverse molecular structures
- Can be trained unsupervised on large molecular databases

#### Challenges
- Reconstruction of valid molecules can be difficult
- Mode collapse in complex chemical spaces
- Balancing reconstruction quality and latent space organization

### Generative Adversarial Networks (GANs)

GANs consist of two competing neural networks that learn to generate realistic molecular structures.

#### Architecture

**Generator (G)**: Creates fake molecules from random noise
- Input: Random noise vector z ~ N(0, I)
- Output: Generated molecular representation (SMILES, graph, etc.)
- Learns to fool the discriminator

**Discriminator (D)**: Distinguishes real molecules from generated ones
- Input: Molecular representation (real or generated)
- Output: Probability that input is real
- Learns to identify fake molecules

#### Training Process

The networks play a minimax game:

```
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

Training alternates between:
1. Update discriminator to better distinguish real from fake
2. Update generator to better fool discriminator

#### Molecular GANs Variants

**MolGAN**: Graph-based GAN for molecular generation
- Generates molecular graphs directly
- Uses permutation-invariant discriminator
- Includes reward network for property optimization

**Objective-Reinforced GAN (ORGAN)**: Combines GAN with RL
- Adds policy gradient for optimizing molecular properties
- Generator receives rewards for desired properties
- Balances diversity with property optimization

#### Advantages
- Can generate highly realistic molecular distributions
- No explicit likelihood computation needed
- Flexible in incorporating domain knowledge

#### Challenges
- Training instability and mode collapse
- Difficulty ensuring chemical validity
- Requires careful hyperparameter tuning

---

## 2. Autoregressive Models (RNNs, Transformers)

Autoregressive models generate molecules sequentially, one token at a time, based on previously generated tokens.

### Recurrent Neural Networks (RNNs)

#### Architecture

RNNs process SMILES strings character-by-character:

```
h_t = f(h_{t-1}, x_t)
p(x_t | x_1, ..., x_{t-1}) = softmax(W_h h_t + b)
```

**LSTM/GRU variants** address vanishing gradient problems and capture long-range dependencies in molecular structures.

#### Training

Models are trained on large databases of molecules using teacher forcing:
- Input: SMILES prefixes (e.g., "CC(=O)")
- Target: Next character (e.g., "O")
- Loss: Cross-entropy between predicted and actual next character

#### Generation

Sample molecules by iteratively predicting the next character:
1. Start with begin token
2. Sample next character from probability distribution
3. Append to sequence and repeat
4. Stop at end token or max length

#### Transfer Learning

Pre-trained RNN language models can be fine-tuned for specific molecular design tasks:
- Pre-train on large molecular databases (ChEMBL, ZINC)
- Fine-tune on smaller target datasets with desired properties
- Enables generation of molecules similar to training distribution

### Transformers for Molecular Generation

Transformers leverage self-attention mechanisms for improved sequence modeling.

#### Architecture

**Self-Attention**: Allows model to attend to all positions simultaneously

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention**: Parallel attention mechanisms capture different relationships

**Positional Encoding**: Injects sequence order information

#### Advantages Over RNNs
- Parallel processing enables faster training
- Better capture of long-range dependencies
- More effective at learning complex molecular patterns

#### Molecular Transformer Applications

**SMILES Generation**: Character-level transformer language models
- Pre-trained on millions of molecules
- Fine-tuned for specific chemical spaces

**Reaction Prediction**: Transform reactants to products
- Trained on reaction datasets (USPTO)
- Can predict reaction outcomes and suggest synthesis routes

**Molecular Translation**: Convert between representations
- SMILES to IUPAC names
- 2D to 3D structure prediction
- Retrosynthesis planning

#### Notable Models

**ChemBERTa**: Transformer pre-trained on SMILES
- Masked language modeling on molecular data
- Transfer learning for downstream tasks

**MolGPT**: GPT-based architecture for molecules
- Autoregressive generation of SMILES
- Can be conditioned on desired properties

---

## 3. Diffusion Models

Diffusion models represent a recent paradigm shift in generative modeling, achieving state-of-the-art results across multiple domains including molecular generation.

### Principles

Diffusion models learn to reverse a gradual noising process:

**Forward Process**: Progressively add Gaussian noise to data
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```

**Reverse Process**: Learn to denoise and recover original data
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### Training

Train a neural network to predict the noise at each timestep:
1. Sample molecule x₀ from dataset
2. Sample timestep t and noise ε ~ N(0, I)
3. Create noisy version x_t
4. Train network to predict ε from x_t and t
5. Loss: MSE between predicted and actual noise

### Generation

Generate molecules by iterative denoising:
1. Start with pure noise x_T ~ N(0, I)
2. Iteratively denoise: x_{t-1} = denoise(x_t, t)
3. Final sample x₀ is generated molecule

### Molecular Diffusion Models

**Continuous Representations**: Operate in latent space or coordinate space
- **E(n) Equivariant Diffusion**: Generates 3D molecular conformations
- Respects rotational and translational symmetries
- Directly outputs atomic positions

**Discrete Representations**: Adapt diffusion to categorical data
- **Discrete Diffusion**: For SMILES or graph generation
- Multinomial diffusion over molecular tokens
- Transition matrices for adding/removing atoms and bonds

**Conditional Diffusion**: Guide generation towards desired properties
- Add property information at each denoising step
- Classifier guidance or classifier-free guidance
- Control molecular properties during generation

### Advantages
- High-quality, diverse samples
- Stable training compared to GANs
- Flexible conditioning mechanisms
- Can generate both 2D and 3D molecular structures

### Challenges
- Slower generation compared to one-shot methods
- Computational cost of iterative sampling
- Ensuring chemical validity in discrete spaces

---

## 4. Reinforcement Learning for Optimization

Reinforcement learning enables goal-directed molecular generation by optimizing molecules for specific properties.

### Framework

Treat molecular generation as a sequential decision-making problem:

**Agent**: Generative model (policy)
**Environment**: Chemical space
**State**: Current partial molecule
**Action**: Add/modify atom or bond
**Reward**: Molecular property score (e.g., drug-likeness, binding affinity)

### Policy Optimization

#### REINFORCE Algorithm

Update policy to maximize expected reward:

```
∇_θ J(θ) = E[∑_t R_t ∇_θ log π_θ(a_t | s_t)]
```

**Baseline subtraction** reduces variance:
```
∇_θ J(θ) = E[∑_t (R_t - b) ∇_θ log π_θ(a_t | s_t)]
```

#### Proximal Policy Optimization (PPO)

Constrains policy updates for stable training:
```
L(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

### Reward Functions

Design reward functions to capture desired properties:

**Single Objective**:
```
R = f(molecule)
```
where f might be:
- Binding affinity prediction
- Synthetic accessibility score
- QED (quantitative estimate of drug-likeness)
- LogP (lipophilicity)

**Multi-Objective**:
```
R = w₁f₁(mol) + w₂f₂(mol) + ... + wₙfₙ(mol)
```

**Reward Shaping**:
- Penalize invalid molecules
- Reward intermediate steps
- Balance exploration and exploitation

### Transfer Learning

Pre-train generative model on molecular databases, then fine-tune with RL:

1. **Pre-training**: Learn general molecular distribution
2. **RL Fine-tuning**: Optimize for specific properties
3. **Constrained Optimization**: Maintain molecular validity while optimizing

### Applications

**De Novo Drug Design**: Generate molecules with:
- High predicted binding affinity to target protein
- Good ADMET properties
- Synthetic accessibility

**Lead Optimization**: Modify existing molecules to:
- Improve potency
- Reduce toxicity
- Optimize pharmacokinetics

**Multi-Parameter Optimization (MPO)**: Balance multiple objectives
- Efficacy vs. safety
- Potency vs. selectivity
- Activity vs. synthetic accessibility

### Challenges
- Reward function design and balancing
- Sparse and delayed rewards
- Maintaining molecular diversity
- Mode collapse to few high-reward molecules

---

## 5. Conditional Generation

Conditional generation enables control over generated molecules by incorporating desired properties or constraints into the generation process.

### Conditioning Strategies

#### Explicit Conditioning

Concatenate property information with latent vector:
```
z_cond = [z, c]
```
where c represents condition (e.g., property value, class label)

#### Cross-Attention Conditioning

Use attention mechanisms to integrate conditions:
- Query: Latent molecular representation
- Key/Value: Condition embeddings
- Allows flexible, learned integration of conditions

#### Classifier Guidance

Steer generation using property prediction model:
```
∇_x log p(x|c) ≈ ∇_x log p(x) + λ∇_x log p(c|x)
```

Guide generation towards higher predicted property values.

### Types of Conditions

**Scalar Properties**:
- Molecular weight
- LogP
- TPSA (topological polar surface area)
- QED score

**Categorical Properties**:
- Compound class (kinase inhibitor, GPCR ligand, etc.)
- Scaffold type
- Functional groups present

**Structural Constraints**:
- Required substructures (pharmacophores)
- Forbidden substructures (toxic moieties)
- Scaffold constraints

**Target-Based**:
- Protein target (e.g., "generate EGFR inhibitor")
- Binding site information
- Target protein sequence/structure

### Conditional VAE Architecture

Modify standard VAE to accept conditions:

**Encoder**:
```
q_φ(z | x, c) = N(μ_φ(x, c), σ_φ(x, c))
```

**Decoder**:
```
p_θ(x | z, c)
```

**Prior**:
```
p(z | c) = N(μ_prior(c), σ_prior(c))
```

### Multi-Conditional Generation

Generate molecules satisfying multiple constraints simultaneously:
- Combine multiple property conditions
- Use hierarchical conditioning
- Balance competing objectives

### Controllable Generation Workflow

1. **Train conditional model** on labeled molecular dataset
2. **Specify desired properties** for new molecule
3. **Generate candidates** with specified properties
4. **Validate and filter** generated molecules
5. **Iterate** by adjusting conditions

---

## 6. Practical: Building a Conditional VAE

This practical exercise guides you through implementing a conditional VAE for molecular generation.

### Dataset Preparation

We'll use a subset of molecules with associated properties:

```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import torch
from torch.utils.data import Dataset, DataLoader

# Load molecular data
def prepare_dataset(smiles_file, max_length=120):
    """Prepare SMILES dataset with properties"""
    df = pd.read_csv(smiles_file)
    
    # Calculate properties
    properties = []
    valid_smiles = []
    
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props = {
                'logp': Descriptors.MolLogP(mol),
                'mw': Descriptors.MolWt(mol),
                'qed': QED.qed(mol)
            }
            properties.append(props)
            valid_smiles.append(smiles)
    
    # Normalize properties
    props_df = pd.DataFrame(properties)
    props_normalized = (props_df - props_df.mean()) / props_df.std()
    
    return valid_smiles, props_normalized

# Create vocabulary
def create_vocabulary(smiles_list):
    """Create character vocabulary from SMILES"""
    chars = set()
    for smiles in smiles_list:
        chars.update(smiles)
    
    chars = sorted(list(chars))
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<START>'] = len(char_to_idx)
    char_to_idx['<END>'] = len(char_to_idx)
    
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

class MolecularDataset(Dataset):
    """PyTorch dataset for molecular SMILES"""
    
    def __init__(self, smiles_list, properties, char_to_idx, max_length=120):
        self.smiles_list = smiles_list
        self.properties = properties
        self.char_to_idx = char_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        props = self.properties.iloc[idx].values
        
        # Encode SMILES
        encoded = [self.char_to_idx['<START>']]
        encoded += [self.char_to_idx[c] for c in smiles]
        encoded += [self.char_to_idx['<END>']]
        
        # Pad sequence
        if len(encoded) < self.max_length:
            encoded += [self.char_to_idx['<PAD>']] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(props, dtype=torch.float32)
```

### Model Architecture

Implement the conditional VAE components:

```python
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEncoder(nn.Module):
    """Encoder: SMILES + properties -> latent distribution"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_properties, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + num_properties, hidden_dim, 
                           num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x, properties):
        # Embed SMILES
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Concatenate properties to each timestep
        batch_size, seq_len = x.size()
        props_expanded = properties.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([embedded, props_expanded], dim=-1)
        
        # LSTM encoding
        _, (hidden, _) = self.lstm(lstm_input)
        
        # Combine forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Latent distribution parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

class ConditionalDecoder(nn.Module):
    """Decoder: latent + properties -> SMILES"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_properties, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + num_properties, hidden_dim, 
                           num_layers=num_layers, batch_first=True)
        
        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim + num_properties, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim + num_properties, hidden_dim * num_layers)
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
    
    def forward(self, z, properties, target_seq):
        batch_size = z.size(0)
        
        # Combine latent and properties
        z_cond = torch.cat([z, properties], dim=1)
        
        # Initialize hidden and cell states
        h0 = self.latent_to_hidden(z_cond).view(batch_size, self.num_layers, -1).transpose(0, 1).contiguous()
        c0 = self.latent_to_cell(z_cond).view(batch_size, self.num_layers, -1).transpose(0, 1).contiguous()
        
        # Embed target sequence
        embedded = self.embedding(target_seq)
        
        # Add properties to each timestep
        seq_len = target_seq.size(1)
        props_expanded = properties.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([embedded, props_expanded], dim=-1)
        
        # LSTM decoding
        output, _ = self.lstm(lstm_input, (h0, c0))
        
        # Project to vocabulary
        logits = self.output(output)
        
        return logits

class ConditionalVAE(nn.Module):
    """Complete Conditional VAE for molecular generation"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 latent_dim=64, num_properties=3, num_layers=2):
        super().__init__()
        
        self.encoder = ConditionalEncoder(vocab_size, embed_dim, hidden_dim, 
                                          latent_dim, num_properties, num_layers)
        self.decoder = ConditionalDecoder(vocab_size, embed_dim, hidden_dim, 
                                          latent_dim, num_properties, num_layers)
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, properties):
        # Encode
        mu, logvar = self.encoder(x, properties)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode (teacher forcing)
        # Input: all tokens except last, Target: all tokens except first
        decoder_input = x[:, :-1]
        logits = self.decoder(z, properties, decoder_input)
        
        return logits, mu, logvar
    
    def generate(self, properties, char_to_idx, idx_to_char, max_length=120, temperature=1.0):
        """Generate SMILES from properties"""
        self.eval()
        with torch.no_grad():
            batch_size = properties.size(0)
            
            # Sample from prior
            z = torch.randn(batch_size, self.latent_dim).to(properties.device)
            
            # Start with <START> token
            generated = torch.tensor([[char_to_idx['<START>']]] * batch_size).to(properties.device)
            
            # Generate sequence
            for _ in range(max_length - 1):
                logits = self.decoder(z, properties, generated)
                
                # Get last token logits
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences generated <END>
                if (next_token == char_to_idx['<END>']).all():
                    break
            
            # Convert to SMILES
            smiles_list = []
            for seq in generated:
                chars = [idx_to_char[idx.item()] for idx in seq]
                # Stop at <END> token
                if '<END>' in chars:
                    chars = chars[:chars.index('<END>')]
                smiles = ''.join([c for c in chars if c not in ['<START>', '<PAD>']])
                smiles_list.append(smiles)
            
            return smiles_list
```

### Training Loop

Train the conditional VAE with reconstruction and KL divergence losses:

```python
def vae_loss(logits, target, mu, logvar, beta=0.1):
    """
    VAE loss = Reconstruction loss + β * KL divergence
    """
    # Reconstruction loss (cross-entropy)
    recon_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                   target.reshape(-1), 
                                   ignore_index=0)  # Ignore padding
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_cvae(model, train_loader, num_epochs=50, lr=1e-3, beta=0.1, device='cuda'):
    """Train conditional VAE"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}
        
        for batch_idx, (smiles_encoded, properties) in enumerate(train_loader):
            smiles_encoded = smiles_encoded.to(device)
            properties = properties.to(device)
            
            # Forward pass
            logits, mu, logvar = model(smiles_encoded, properties)
            
            # Compute loss
            target = smiles_encoded[:, 1:]  # Shift by 1 for target
            total_loss, recon_loss, kl_loss = vae_loss(logits, target, mu, logvar, beta)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
        
        # Average losses
        num_batches = len(train_loader)
        avg_total = epoch_losses['total'] / num_batches
        avg_recon = epoch_losses['recon'] / num_batches
        avg_kl = epoch_losses['kl'] / num_batches
        
        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {avg_total:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
    
    return history

# Example usage
def main():
    # Prepare data
    smiles_list, properties = prepare_dataset('molecules.csv')
    char_to_idx, idx_to_char = create_vocabulary(smiles_list)
    
    dataset = MolecularDataset(smiles_list, properties, char_to_idx)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    vocab_size = len(char_to_idx)
    model = ConditionalVAE(vocab_size, num_properties=3)
    
    # Train
    history = train_cvae(model, train_loader, num_epochs=50, beta=0.1)
    
    # Generate molecules with desired properties
    # Example: Generate molecules with specific LogP, MW, QED
    target_props = torch.tensor([[0.5, 0.0, 0.8]])  # Normalized values
    generated_smiles = model.generate(target_props, char_to_idx, idx_to_char)
    
    print("\nGenerated SMILES:")
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print(f"{smiles} - Valid molecule")
        else:
            print(f"{smiles} - Invalid")
```

### Evaluation and Analysis

Evaluate the conditional VAE's performance:

```python
def evaluate_generation_quality(model, test_loader, char_to_idx, idx_to_char, num_samples=1000):
    """Evaluate generation quality metrics"""
    model.eval()
    
    metrics = {
        'validity': 0,
        'uniqueness': set(),
        'reconstruction_accuracy': 0,
        'property_correlation': []
    }
    
    with torch.no_grad():
        # Test reconstruction
        total_correct = 0
        total_tokens = 0
        
        for smiles_encoded, properties in test_loader:
            logits, mu, logvar = model(smiles_encoded.to(model.device), 
                                       properties.to(model.device))
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            target = smiles_encoded[:, 1:].to(model.device)
            correct = (predictions == target) & (target != 0)  # Exclude padding
            total_correct += correct.sum().item()
            total_tokens += (target != 0).sum().item()
        
        metrics['reconstruction_accuracy'] = total_correct / total_tokens
        
        # Test generation
        property_values = torch.randn(num_samples, 3)  # Sample random properties
        generated_smiles = model.generate(property_values, char_to_idx, idx_to_char)
        
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                metrics['validity'] += 1
                metrics['uniqueness'].add(smiles)
        
        metrics['validity'] /= num_samples
        metrics['uniqueness'] = len(metrics['uniqueness']) / num_samples
    
    return metrics

def visualize_latent_space(model, data_loader, properties_df):
    """Visualize latent space with t-SNE"""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    model.eval()
    latent_vectors = []
    property_values = []
    
    with torch.no_grad():
        for smiles_encoded, properties in data_loader:
            mu, _ = model.encoder(smiles_encoded.to(model.device), 
                                 properties.to(model.device))
            latent_vectors.append(mu.cpu().numpy())
            property_values.append(properties.cpu().numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    property_values = np.vstack(property_values)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot colored by property (e.g., LogP)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=property_values[:, 0], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='LogP')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('Latent Space Visualization')
    plt.show()

def interpolate_molecules(model, smiles1, smiles2, properties, 
                         char_to_idx, idx_to_char, steps=5):
    """Interpolate between two molecules in latent space"""
    model.eval()
    
    # Encode both molecules
    with torch.no_grad():
        encoded1 = torch.tensor([[char_to_idx.get(c, 0) for c in smiles1]]).to(model.device)
        encoded2 = torch.tensor([[char_to_idx.get(c, 0) for c in smiles2]]).to(model.device)
        props = properties.to(model.device)
        
        mu1, _ = model.encoder(encoded1, props)
        mu2, _ = model.encoder(encoded2, props)
        
        # Interpolate
        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            z_interp = alpha * mu1 + (1 - alpha) * mu2
            smiles = model.decoder.generate_from_latent(z_interp, props, 
                                                        char_to_idx, idx_to_char)[0]
            interpolated.append(smiles)
    
    return interpolated
```

### Hyperparameter Tuning

Key hyperparameters to tune:

```python
hyperparameters = {
    'embed_dim': [64, 128, 256],
    'hidden_dim': [128, 256, 512],
    'latent_dim': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'beta': [0.01, 0.1, 1.0],  # KL weight
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128],
    'temperature': [0.7, 1.0, 1.5]  # For generation
}
```

Suggested β annealing schedule for better training:
```python
def get_beta(epoch, warmup_epochs=10, max_beta=1.0):
    """Gradually increase KL weight"""
    if epoch < warmup_epochs:
        return max_beta * (epoch / warmup_epochs)
    return max_beta
```

---

## Summary

In this session, we explored various generative modeling approaches for molecular design:

**VAEs and GANs** provide complementary approaches to learning molecular distributions, with VAEs offering interpretable latent spaces and GANs enabling high-quality generation through adversarial training.

**Autoregressive models** (RNNs and Transformers) generate molecules sequentially, leveraging powerful language modeling techniques adapted for chemical structures.

**Diffusion models** represent the state-of-the-art in many generation tasks, offering stable training and high-quality, diverse molecular samples.

**Reinforcement learning** enables goal-directed optimization, steering generation towards molecules with desired properties through reward-based training.

**Conditional generation** provides fine-grained control over molecular properties, enabling targeted design for specific applications.

The practical implementation demonstrates how these concepts come together in a working system, from data preparation through model training to generation and evaluation.

---

## Additional Resources

### Papers
- **VAEs**: "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" (Gómez-Bombarelli et al., 2018)
- **GANs**: "MolGAN: An implicit generative model for small molecular graphs" (De Cao & Kipf, 2018)
- **Transformers**: "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction" (Schwaller et al., 2019)
- **Diffusion**: "Equivariant Diffusion for Molecule Generation in 3D" (Hoogeboom et al., 2022)
- **RL**: "Optimization of Molecules via Deep Reinforcement Learning" (Zhou et al., 2019)

### Libraries and Tools
- **RDKit**: Cheminformatics toolkit for molecular manipulation
- **PyTorch Geometric**: Graph neural network library
- **Transformers (Hugging Face)**: Pre-trained transformer models
- **GuacaMol**: Benchmarking platform for generative models
- **MOSES**: Molecular sets for evaluation

### Datasets
- **ZINC**: 230M commercially available compounds
- **ChEMBL**: Bioactive molecules with drug-like properties
- **QM9**: 134k molecules with quantum properties
- **USPTO**: Chemical reaction dataset for synthesis planning

---

## Exercises

1. **Extend the conditional VAE** to include additional properties (e.g., number of rotatable bonds, number of aromatic rings)

2. **Implement sampling strategies**: Compare greedy decoding, beam search, and nucleus sampling for molecule generation

3. **Add validity constraints**: Modify the decoder to enforce SMILES grammar rules and improve validity rate

4. **Multi-objective conditioning**: Train a model to generate molecules satisfying multiple property constraints simultaneously

5. **Latent space arithmetic**: Explore property modification by manipulating latent representations (e.g., z_new = z_base + δz_property)

6. **Benchmark your model**: Evaluate on standard metrics (validity, uniqueness, novelty, FCD, KL divergence) and compare with baselines

7. **Incorporate 3D structure**: Extend the model to condition on or generate 3D molecular conformations

8. **Active learning loop**: Implement an iterative generation and selection process where predicted properties guide the next generation round