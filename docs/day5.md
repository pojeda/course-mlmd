# Generative Models for Molecular Science: AlphaFold as a Case Study

Generative machine learning provides a framework for creating new examples from patterns learned in data. In molecular science, the generated objects may be chemical structures, SMILES strings, molecular graphs, amino-acid sequences, or three-dimensional atomic coordinates. Such methods support molecular discovery, protein design, and the exploration of structural alternatives.

This course introduces the mathematical foundations of generative modeling and develops their connection to modern biomolecular structure prediction and design. We begin with probability distributions and sampling, progress through autoregressive models, variational autoencoders (VAEs), generative adversarial networks (GANs), diffusion models, and flow matching, and then examine how these ideas relate to the evolution of **AlphaFold** and related protein-design methods.

A central distinction runs throughout the course: **predicting the structure of a specified protein** is not the same scientific task as **designing a new protein structure**. Nevertheless, a predictive system may contain a generative model internally. AlphaFold 2, AlphaFold 3, RFdiffusion, and SimpleFold provide useful examples of this distinction.

The material is intended for approximately two days of instruction, but it is organized as one continuous sequence of topics rather than as separate daily chapters. Mathematical expressions use LaTeX syntax compatible with MathJax, and the practical exercises use Python code blocks inside collapsible **Material for MkDocs** admonitions.

## Learning Objectives and Prerequisites

After completing the course, participants should be able to explain how a generative model differs from a conventional property-prediction model; describe sampling, likelihood, latent variables, and conditional distributions; compare major generative-model families; explain the forward and reverse processes in diffusion models; and understand why geometric symmetry matters when generating molecular structures.

Participants should also be able to distinguish AlphaFold 2 from AlphaFold 3, explain the roles of RFdiffusion and SimpleFold, and identify the limitations of generated structures and their evaluation metrics.

Basic familiarity with Python, NumPy, neural networks, molecular structures, and protein sequences is helpful. Practical examples use `numpy`, `matplotlib`, `biopython`, and optionally `torch`.

??? note "Python environment"

    Install the packages used in the exercises in a suitable virtual environment:

    ```bash
    python -m pip install numpy matplotlib biopython torch
    ```

    The small numerical examples require only NumPy and Matplotlib. The protein example requires Biopython and a PDB coordinate file. The neural-network exercise requires PyTorch. No AlphaFold model weights or GPU are needed for these educational demonstrations.

## 1. What Is a Generative Model?

A **generative model** learns the statistical structure of a dataset and uses this information to produce new samples. Let the observations be drawn from an unknown data distribution,

$$
\mathbf{x}\sim p_{\mathrm{data}}(\mathbf{x}).
$$

A probabilistic generative model seeks a parameterized approximation,

$$
p_{\theta}(\mathbf{x})\approx p_{\mathrm{data}}(\mathbf{x}),
$$

where $\theta$ represents model parameters learned from training data. Once trained, the model can generate a sample,

$$
\mathbf{x}_{\mathrm{new}}\sim p_{\theta}(\mathbf{x}).
$$

The result may differ from every training example while remaining consistent with patterns in the learned distribution. In practice, not all generative models provide a directly evaluable probability density: some define only a procedure for sampling.

For molecular applications, $\mathbf{x}$ might encode a molecular graph, a SMILES string, a protein sequence, or an array of atomic coordinates. Each representation brings its own constraints. A generated SMILES string must describe a valid molecular graph; a generated protein backbone must obey geometrical and stereochemical constraints.

### Predictive and generative modeling

A conventional supervised prediction model learns a relationship between features and a target,

$$
\hat y=f_{\theta}(\mathbf{x}).
$$

For example, a QSAR regressor maps molecular descriptors to predicted solubility. A generative model instead defines how to sample candidate objects. Conditional generative models may perform generation **given an input** rather than generating freely.

```text
Property prediction:       molecule  --->  predicted solubility
Protein prediction:        sequence  --->  predicted structure
Molecular generation:      noise     --->  candidate molecule
Conditional generation:    noise + constraints  --->  candidate structure
```

These categories describe **tasks**, not necessarily exclusive network architectures. A structure predictor that uses diffusion to generate its output is performing a predictive scientific task through a generative mechanism.

## 2. Probability Distributions and Sampling

For a discrete random variable, the probabilities of all possible outcomes must sum to one,

$$
\sum_{\mathbf{x}}p_{\theta}(\mathbf{x})=1.
$$

For a continuous random variable, the probability density integrates to one,

$$
\int_{\mathbb{R}^{d}}p_{\theta}(\mathbf{x})\,d\mathbf{x}=1.
$$

**Sampling** means producing an observation according to such a distribution. A simple Gaussian distribution is written as

$$
x\sim\mathcal{N}(\mu,\sigma^2),
$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

??? note "Example: sampling from a Gaussian distribution"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    mu = 0.0
    sigma = 1.0

    samples = rng.normal(loc=mu, scale=sigma, size=5000)
    print(f"Sample mean: {samples.mean():.3f}")
    print(f"Sample std:  {samples.std():.3f}")

    plt.figure(figsize=(7, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Samples from a Gaussian Distribution")
    plt.tight_layout()
    plt.savefig("gaussian_samples.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

This is a sampling demonstration, not a **learned** generative model. Its value is to make clear what it means to draw multiple, nonidentical examples from one probability distribution.

## 3. Unconditional and Conditional Generation

An **unconditional model** samples directly from a learned distribution $p_{\theta}(\mathbf{x})$. A **conditional generative model** uses additional information $\mathbf{c}$ and samples from

$$
p_{\theta}(\mathbf{x}\mid\mathbf{c}).
$$

The condition can represent a desired physicochemical property, a protein sequence, a binding partner, a structural motif, a chemical composition, or experimentally measured information. For example, generating a protein backbone around a fixed catalytic motif is a conditional generation task.

```text
Condition: sequence, binding target, or structural motif
                         +
                    random sample
                         |
                         v
              Conditional generator
                         |
                         v
                   3D structure
```

Conditioning does not guarantee that an output satisfies every scientific requirement. Generated candidates still require structural validation, property assessment, and often experimental testing.

## 4. Latent Variables and Learned Representations

A **latent variable** $\mathbf{z}$ is an internal variable that can capture variation in the data. A common starting choice is a standard multivariate normal distribution,

$$
\mathbf{z}\sim p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

A generator can transform this latent input into a sample,

$$
\mathbf{x}=G_{\theta}(\mathbf{z}).
$$

In a probabilistic latent-variable model, the observed-data distribution may be written as

$$
p_{\theta}(\mathbf{x})
=\int p_{\theta}(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}.
$$

A latent space is not necessarily a direct or interpretable coordinate system for physical properties. Its usefulness depends on the model architecture, training data, objective, and learned representations.

## 5. Training Generative Models with Likelihood

One way to train a probabilistic generative model is **maximum likelihood estimation**. Given $N$ independent observations

$$
\mathcal{D}=\{\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N\},
$$

the likelihood of the dataset is

$$
L(\theta)=\prod_{i=1}^{N}p_{\theta}(\mathbf{x}_i).
$$

Taking the logarithm transforms products into sums,

$$
\log L(\theta)=\sum_{i=1}^{N}\log p_{\theta}(\mathbf{x}_i).
$$

Training usually minimizes the **negative log-likelihood**,

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=-\frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(\mathbf{x}_i).
$$

Not every generative model uses this loss explicitly. VAEs optimize a likelihood-related lower bound, GANs use an adversarial objective, and diffusion/flow-matching models are often trained with denoising or velocity-matching losses.

## 6. Autoregressive Models

An autoregressive model generates a sequence one element at a time. For a sequence $\mathbf{x}=(x_1,\ldots,x_T)$, the chain rule gives

$$
p_{\theta}(\mathbf{x})
=\prod_{t=1}^{T}
p_{\theta}(x_t\mid x_1,\ldots,x_{t-1}).
$$

For proteins, each $x_t$ may represent an amino-acid identity. The model predicts a distribution over possible next residues, samples one, and repeats. Real protein language models use learned contextual representations and all 20 standard amino acids (plus any special tokens), not merely the identity of the previous residue.

??? note "Example: a toy sequential amino-acid generator"

    ```python
    import numpy as np

    amino_acids = ["A", "G", "L", "V"]

    # Invented transition probabilities for teaching.
    transitions = {
        "A": [0.10, 0.30, 0.40, 0.20],
        "G": [0.30, 0.20, 0.20, 0.30],
        "L": [0.25, 0.15, 0.40, 0.20],
        "V": [0.30, 0.20, 0.30, 0.20],
    }

    rng = np.random.default_rng(42)
    sequence = ["A"]

    for _ in range(19):
        previous = sequence[-1]
        next_residue = rng.choice(
            amino_acids, p=transitions[previous]
        )
        sequence.append(next_residue)

    print("Generated toy sequence:", "".join(sequence))
    ```

This example is a **first-order Markov chain with hand-chosen probabilities**, not a trained protein language model. It illustrates the sampling process, but it does not produce validated biological sequences.

## 7. Variational Autoencoders

A **Variational Autoencoder (VAE)** combines a probabilistic encoder and a decoder. The encoder estimates a distribution over latent variables conditioned on an observation,

$$
q_{\phi}(\mathbf{z}\mid\mathbf{x}),
$$

while the decoder defines a distribution over reconstructed data,

$$
p_{\theta}(\mathbf{x}\mid\mathbf{z}).
$$

A VAE is commonly trained by maximizing the **evidence lower bound (ELBO)**,

$$
\mathcal{L}_{\mathrm{ELBO}}(\mathbf{x})
=
\mathbb{E}_{q_{\phi}(\mathbf{z}\mid\mathbf{x})}
\left[\log p_{\theta}(\mathbf{x}\mid\mathbf{z})\right]
-
D_{\mathrm{KL}}\!\left(
q_{\phi}(\mathbf{z}\mid\mathbf{x})
\middle\|
p(\mathbf{z})
\right).
$$

The first term rewards reconstruction quality; the Kullback–Leibler divergence encourages consistency between the learned approximate posterior and the chosen latent prior. After training, samples can be generated by drawing $\mathbf{z}$ from the prior and passing it through the decoder.

```text
Training:   molecule --> encoder --> latent z --> decoder --> reconstruction
Generation: random z -------------------------> decoder --> new candidate
```

VAEs have been explored for molecular generation using continuous latent representations of molecules and molecular structures.

## 8. Generative Adversarial Networks

A **Generative Adversarial Network (GAN)** trains a generator $G_{\theta}$ against a discriminator $D_{\phi}$. The generator maps noise to candidate samples,

$$
\mathbf{x}_{\mathrm{fake}}=G_{\theta}(\mathbf{z}),
$$

while the discriminator attempts to distinguish generated samples from real observations. The original minimax objective is

$$
\min_G\max_D\left[
\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}
\log D(\mathbf{x})
+
\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}
\log\bigl(1-D(G(\mathbf{z}))\bigr)
\right].
$$

GAN training can generate realistic samples, but it can also suffer from instability or **mode collapse**, where the generator produces insufficient diversity. GANs are useful historical context before considering diffusion and flow-based methods that are now prominent in geometric generative modeling.

## 9. Diffusion Models: Forward Noising and Reverse Generation

A **diffusion model** learns to reverse a process that progressively corrupts data with noise. It consists conceptually of a **forward noising process** and a learned **reverse generative process**.

For a discrete-time Gaussian diffusion model, the forward transition is

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
=\mathcal{N}\!\left(
\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},
\beta_t\mathbf{I}
\right),
$$

where $0<\beta_t<1$ controls the noise variance at step $t$. Define

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

The noisy sample at arbitrary step $t$ can be constructed directly from a clean example $\mathbf{x}_0$:

$$
\boxed{
\mathbf{x}_t
=\sqrt{\bar\alpha_t}\,\mathbf{x}_0
+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,
\qquad
\boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
}
$$

When $\bar\alpha_t$ is near one, the sample remains close to the original data. When $\bar\alpha_t$ becomes small, the sample is dominated by noise.

```text
Forward (fixed):   clean x0 --> noisier x1 --> ... --> noise xT
Reverse (learned): noise xT --> denoise --> ... --> generated x0
```

During generation, the reverse model begins from an approximately Gaussian sample and gradually transforms it into a sample resembling the training distribution. The reverse transitions are **learned**, rather than being obtained by simply subtracting arbitrary noise.

### Noise-prediction and clean-sample objectives

A common diffusion training strategy predicts the noise added at step $t$,

$$
\boldsymbol\epsilon_{\theta}(\mathbf{x}_t,t).
$$

The corresponding objective is

$$
\mathcal{L}_{\mathrm{noise}}
=\mathbb{E}_{\mathbf{x}_0,t,\boldsymbol\epsilon}
\left[
\left\|
\boldsymbol\epsilon-
\boldsymbol\epsilon_{\theta}(\mathbf{x}_t,t)
\right\|_2^2
\right].
$$

An alternative is to predict the **clean sample** $\hat{\mathbf{x}}_0$ directly. This distinction matters for AlphaFold 3: its diffusion module is trained to take noisy atomic coordinates and predict denoised coordinates, rather than being adequately described as an ordinary noise-prediction network.

??? note "Example: visualize progressive diffusion noise"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[-2, 0], scale=0.35, size=(250, 2))
    x1 = rng.normal(loc=[2, 0], scale=0.35, size=(250, 2))
    clean = np.vstack([x0, x1])

    alpha_bar_values = [1.0, 0.8, 0.3, 0.01]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)

    for ax, alpha_bar in zip(axes, alpha_bar_values):
        noise = rng.normal(size=clean.shape)
        noisy = (
            np.sqrt(alpha_bar) * clean
            + np.sqrt(1.0 - alpha_bar) * noise
        )
        ax.scatter(noisy[:, 0], noisy[:, 1], s=8, alpha=0.5)
        ax.set_title(f"alpha_bar = {alpha_bar:.2f}")
        ax.set_xlabel("x")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)

    axes[0].set_ylabel("y")
    plt.suptitle("Forward Diffusion: Increasing Noise")
    plt.tight_layout()
    plt.savefig("forward_diffusion.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

The two clusters above are synthetic and have no molecular meaning. They are useful because the effect of the noise schedule can be seen clearly in two dimensions.

## 10. Practical Exercise: Train a Small Denoising Model

The preceding example only adds noise. To demonstrate **learning**, we can train a small neural network to predict the added noise for synthetic two-dimensional data. The network receives both the corrupted point $\mathbf{x}_t$ and the diffusion step $t$, and learns a noise estimate $\boldsymbol\epsilon_{\theta}(\mathbf{x}_t,t)$.

After training, the model performs approximate reverse diffusion. The demonstration is deliberately small and is **not** a molecular generator or an implementation of AlphaFold.

??? note "Example: a learned two-dimensional diffusion model (PyTorch)"

    ```python
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    # 1. Two synthetic clusters (training data)
    n = 2000
    left = torch.randn(n // 2, 2) * 0.35 + torch.tensor([-2.0, 0.0])
    right = torch.randn(n // 2, 2) * 0.35 + torch.tensor([2.0, 0.0])
    training_data = torch.cat([left, right], dim=0)

    # 2. Gaussian diffusion schedule
    T = 80
    betas = torch.linspace(0.0005, 0.08, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    # 3. Model input: [x_coordinate, y_coordinate, normalized_time]
    denoiser = nn.Sequential(
        nn.Linear(3, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

    # 4. Learn to predict noise at randomly selected timesteps
    for step in range(800):
        idx = torch.randint(0, n, (256,))
        x0 = training_data[idx]
        t = torch.randint(0, T, (256,))
        epsilon = torch.randn_like(x0)

        a_bar = alpha_bars[t].unsqueeze(1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * epsilon
        time_feature = (t.float() / (T - 1)).unsqueeze(1)

        predicted_noise = denoiser(torch.cat([xt, time_feature], dim=1))
        loss = ((predicted_noise - epsilon) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"Step {step + 1}: loss = {loss.item():.4f}")

    # 5. Generate samples by approximate reverse diffusion
    denoiser.eval()
    with torch.no_grad():
        generated = torch.randn(500, 2)

        for t in reversed(range(T)):
            time_feature = torch.full((len(generated), 1), t / (T - 1))
            predicted_noise = denoiser(
                torch.cat([generated, time_feature], dim=1)
            )

            mean = (
                generated
                - betas[t] / torch.sqrt(1 - alpha_bars[t]) * predicted_noise
            ) / torch.sqrt(alphas[t])

            if t > 0:
                posterior_variance = (
                    betas[t] * (1 - alpha_bars[t - 1])
                    / (1 - alpha_bars[t])
                )
                generated = mean + torch.sqrt(posterior_variance) * torch.randn_like(generated)
            else:
                generated = mean

    # 6. Compare training and generated samples
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(training_data[:, 0], training_data[:, 1], s=5, alpha=0.3)
    axes[0].set_title("Training Distribution")
    axes[1].scatter(generated[:, 0], generated[:, 1], s=5, alpha=0.4)
    axes[1].set_title("Generated Samples")

    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("toy_learned_diffusion.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

The network sees many noisy versions of the training points. During generation, the learned noise estimates guide the reverse trajectory from random noise toward the data distribution. Exact agreement is not expected: the network is small and trained for a limited number of optimization steps. This exercise is intended to illustrate the distinction between **adding noise** and **learning how to generate**.

## 11. Three-Dimensional Molecular Representations

For a molecule or protein with $N$ atoms, let the coordinates be collected in an $N\times 3$ matrix,

$$
\mathbf{R}
=
\begin{bmatrix}
\mathbf{r}_1^{\mathsf T}\\
\mathbf{r}_2^{\mathsf T}\\
\vdots\\
\mathbf{r}_N^{\mathsf T}
\end{bmatrix},
\qquad
\mathbf{r}_i=
\begin{bmatrix}
x_i\\y_i\\z_i
\end{bmatrix}.
$$

Each row of $\mathbf{R}$ represents a point in three-dimensional space. Atomic coordinates preserve detailed geometry, but they also present challenges: meaningful molecular structures occupy a tiny fraction of all possible coordinate arrays. Most arbitrary coordinate samples have unrealistic bond lengths, severe steric clashes, or incorrect stereochemistry.

For protein modeling, a common coarse-grained representation uses the $C_\alpha$ atom of each amino acid, while a more detailed representation retains backbone or all-atom coordinates. A C$\alpha$-only model cannot represent side-chain chemistry or all atomic interactions directly.

## 12. Geometric Symmetry: Invariance and Equivariance

A molecule has no preferred absolute orientation or position. Suppose each coordinate is rotated by $\mathbf{Q}\in SO(3)$ and translated by $\mathbf{t}\in\mathbb{R}^3$:

$$
\mathbf{r}_i'
=\mathbf{Q}\mathbf{r}_i+\mathbf{t}.
$$

A scalar molecular quantity such as potential energy should be **invariant** under a global rigid transformation:

$$
E(\{\mathbf{Q}\mathbf{r}_i+\mathbf{t}\})
=E(\{\mathbf{r}_i\}).
$$

A coordinate-valued prediction should transform **equivariantly**. For a mapping $f$ that outputs vectors,

$$
f(\{\mathbf{Q}\mathbf{r}_i+\mathbf{t}\})
=\{\mathbf{Q}f_i(\mathbf{R})+\mathbf{t}\},
$$

when its outputs are positions. For a force predictor, the translation term is absent:

$$
\mathbf{F}_i(\{\mathbf{Q}\mathbf{r}_j+\mathbf{t}\})
=\mathbf{Q}\mathbf{F}_i(\mathbf{R}).
$$

These distinctions are central in geometry-based molecular machine learning. Importantly, **not every successful protein model hard-codes geometric equivariance into its neural architecture**. In particular, AlphaFold 3 does not impose explicit rotation-equivariant operations in its diffusion module; instead, its training and model formulation encourage geometrically appropriate predictions. It is therefore important to distinguish a desired physical symmetry from the architectural mechanism used to learn or enforce it.

??? info "Advanced material: removing rigid-body degrees of freedom"

    A collection of $N$ points in three dimensions has $3N$ Cartesian coordinates. Overall translation and rotation do not change internal geometry. One can center a structure before comparing or perturbing it,

    $$
    \widetilde{\mathbf{r}}_i
    =\mathbf{r}_i-\frac{1}{N}\sum_{j=1}^{N}\mathbf{r}_j.
    $$

    Centering removes global translation, but not rotational freedom. Shape comparisons often additionally align two structures by finding a best-fit rotation. The appropriate treatment of these degrees of freedom depends on the model and its sampling procedure.

## 13. Python Example: Noising the C$\alpha$ Coordinates of 3MUF

The next exercise connects diffusion mathematics to protein structure. We extract the $C_\alpha$ coordinates of chain A from `3muf.pdb`, center and normalize them, and add Gaussian noise using the forward-diffusion equation. We then compare consecutive C$\alpha$ distances before and after perturbation.

This demonstration deliberately **does not produce valid protein conformations**: it applies generic Gaussian noise without a trained structural prior or stereochemical constraints. The point is to visualize what a denoising model needs to learn.

??? note "Example: forward diffusion of 3MUF C-alpha coordinates"

    ```python
    from pathlib import Path
    from urllib.request import urlretrieve

    import numpy as np
    import matplotlib.pyplot as plt
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    # 1. Download the file if it is not already available
    pdb_file = Path("3muf.pdb")
    if not pdb_file.exists():
        urlretrieve(
            "https://files.rcsb.org/download/3MUF.pdb",
            str(pdb_file),
        )

    # 2. Read chain A and extract C-alpha atoms
    structure = PDBParser(QUIET=True).get_structure("3MUF", str(pdb_file))
    chain = structure[0]["A"]

    residue_names = []
    coordinates = []
    for residue in chain:
        if not is_aa(residue, standard=True) or "CA" not in residue:
            continue
        residue_names.append(residue.get_resname())
        coordinates.append(residue["CA"].get_coord())

    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        raise RuntimeError("Not enough C-alpha atoms were found")
    print("C-alpha coordinate matrix:", coordinates.shape)

    # 3. Normalize to a dimensionless representation
    center = coordinates.mean(axis=0)
    scale = coordinates.std()
    if scale == 0:
        raise RuntimeError("Coordinate scale must be nonzero")
    x0 = (coordinates - center) / scale

    # 4. Add diffusion-style Gaussian noise
    rng = np.random.default_rng(42)
    alpha_bar = 0.50
    epsilon = rng.normal(size=x0.shape)
    xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * epsilon

    # 5. Return noisy coordinates to the original distance scale
    noisy_coordinates = xt * scale + center
    original_distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    noisy_distances = np.linalg.norm(np.diff(noisy_coordinates, axis=0), axis=1)

    print(f"Original mean consecutive CA distance: {original_distances.mean():.2f} Å")
    print(f"Noisy mean consecutive CA distance:    {noisy_distances.mean():.2f} Å")

    # 6. Plot an x-y projection, colored by residue index
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    residue_index = np.arange(len(coordinates))

    axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1],
        c=residue_index, s=20, cmap="viridis"
    )
    axes[0].set_title("Original 3MUF C-alpha Coordinates")

    axes[1].scatter(
        noisy_coordinates[:, 0], noisy_coordinates[:, 1],
        c=residue_index, s=20, cmap="viridis"
    )
    axes[1].set_title("Forward-Diffused Coordinates")

    for ax in axes:
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig("3muf_forward_diffusion.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

The example uses **only the modeled C$\alpha$ atoms**, not the ligands, waters, or side-chain atoms. The normalization is for a convenient numerical illustration: practical biomolecular generative models use more carefully designed coordinate representations, noise schedules, and geometry-aware learning procedures.

## 14. Flow Matching: Learning Continuous Generative Dynamics

**Flow matching** is another approach to generative modeling. It learns a time-dependent velocity field that transports samples from a simple reference distribution toward a target data distribution. A sample follows an ordinary differential equation,

$$
\frac{d\mathbf{x}_t}{dt}
=\mathbf{v}_{\theta}(\mathbf{x}_t,t),
\qquad t\in[0,1].
$$

If $\mathbf{x}_0$ is drawn from a simple distribution, integrating the learned vector field can produce a sample whose distribution resembles the training data at $t=1$.

A generic flow-matching loss compares the learned velocity with a target field $\mathbf{u}_t$:

$$
\mathcal{L}_{\mathrm{FM}}
=\mathbb{E}\left[
\left\|
\mathbf{v}_{\theta}(\mathbf{x}_t,t)
-\mathbf{u}_t
\right\|_2^2
\right].
$$

Flow matching and diffusion are closely related areas of generative modeling, but they are not identical algorithms. Diffusion is commonly framed through stochastic noising and learned denoising, whereas flow matching learns a continuous transport vector field. Either approach may be made conditional on a molecular or protein context $\mathbf{c}$.

??? info "Advanced material: conditional flow matching"

    For a conditioned protein-structure problem, a flow-based model may learn

    $$
    \frac{d\mathbf{R}_t}{dt}
    =\mathbf{v}_{\theta}(\mathbf{R}_t,t,\mathbf{c}),
    $$

    where $\mathbf{c}$ encodes sequence or other structural constraints. The network must learn how the conditioning information changes the trajectories that lead to valid structures.

## 15. Protein Structure as a Learning Problem

A protein sequence containing $L$ amino acids can be represented as

$$
S=(s_1,s_2,\ldots,s_L).
$$

A protein-structure prediction model uses sequence-derived information to estimate a three-dimensional structure,

$$
S\longrightarrow\widehat{\mathbf{R}}.
$$

In practice, the conditioning information may include multiple sequence alignments, structural templates, or information about other molecular components. Protein modeling must account for local geometry, nonlocal contacts, folding constraints, and the existence of multiple possible conformations.

### Folding landscapes and structural ensembles

Protein folding is often illustrated as motion over a free-energy landscape. Different conformations have different free energies, and stable or metastable conformational states may occupy distinct basins.

For a conformation $\mathbf{R}$ at temperature $T$, the equilibrium probability density can be expressed schematically as

$$
p_{\mathrm{eq}}(\mathbf{R})
=\frac{1}{Z}
\exp\left[-\frac{F(\mathbf{R})}{k_{\mathrm B}T}\right],
$$

where $F(\mathbf{R})$ denotes an effective conformational free energy and $Z$ is the normalization constant. The expression illustrates a thermodynamic viewpoint; neither AlphaFold 2 nor AlphaFold 3 directly provides this equilibrium density in general.

Because proteins fluctuate in solution, a single static structure is not a full description of protein dynamics. This distinction becomes particularly important when interpreting multiple samples from a generative structure model.

## 16. AlphaFold 1: Learning Residue-Level Geometry

AlphaFold 1 achieved a major advance in the CASP13 protein-structure prediction assessment. Rather than directly using the later diffusion paradigm, it learned geometrical information about relationships between residues. One important representation is a **distogram**, which describes a probability distribution over residue-pair distances.

For a pair of residues $i$ and $j$ and distance bin $k$,

$$
p_{ij,k}
=P\!\left(d_{ij}\in[b_k,b_{k+1})\mid\text{input features}\right).
$$

This type of output is probabilistic, but **predicting a probability distribution over distances is not by itself the same as a generative model that samples complete protein structures**. The geometric predictions were used to guide the construction of a three-dimensional model.

## 17. AlphaFold 2: A Structure-Prediction Architecture

AlphaFold 2 introduced a substantially redesigned architecture that combines multiple sequence alignments, residue-pair information, optional structural templates, the **Evoformer**, and a geometry-aware **structure module**.

```text
Protein sequence
       |
       +--> sequence search / multiple sequence alignment
       |
       +--> optional structural templates
       |
       v
  MSA and pair representations
       |
       v
     Evoformer
       |
       v
  Structure module
       |
       v
 Predicted 3D protein structure
```

The Evoformer repeatedly updates the MSA and pair representations to capture evolutionary and structural relationships. The structure module produces a coordinate model using learned geometric information. Recycling can refine predictions by feeding previous results back into the model.

### Why AlphaFold 2 is not usually classified as generative

AlphaFold 2 is primarily a conditional **structure predictor**. It does not generate structures by sampling a learned iterative diffusion process from coordinate noise. Although internal representations and the training procedure involve probabilistic ideas, its central inference task is to predict a structure corresponding to the supplied sequence and context.

This is a useful reminder that the presence of neural networks, attention, probability distributions, or random initialization does not automatically make an entire system a generative model.

## 18. The AlphaFold 2 Ecosystem

**ColabFold** makes AlphaFold-style structure prediction more accessible and uses MMseqs2 for rapid homology searches. It is well suited for notebook demonstrations and workflows in which obtaining the MSA efficiently is important.

**AlphaFold-Multimer** extends AlphaFold 2 to protein complexes and places particular emphasis on predicting the arrangement of interacting chains and their interfaces.

**AlphaPulldown** builds workflows for screening potential protein-protein interactions with AlphaFold-based models. It is particularly relevant to batch processing, dataset preparation, and high-performance computing.

**AlphaFold Unmasked** extends AlphaFold 2-related approaches to incorporate resolved experimental structure information and to fill missing structural regions through inpainting. Structural inpainting is closely related to conditional completion, though the use of a reconstruction or completion task does not automatically mean the model is a diffusion generator.

These tools expand the use cases of AlphaFold 2, but should be distinguished from methods specifically trained for generative structure sampling.

## 19. RFdiffusion: Generative Design of Protein Backbones

**RFdiffusion** provides a direct example of generative modeling for proteins. It adapts the RoseTTAFold structure-prediction framework to a diffusion process capable of sampling **new protein backbones**, including structures conditioned on design constraints.

During training, protein structures are corrupted by noise. The model learns to denoise these representations and, during generation, begins from random structural states and constructs protein-like backbones iteratively.

A schematic conditional sampling task is

$$
\mathbf{R}_{\mathrm{new}}
\sim p_{\theta}(\mathbf{R}\mid\mathbf{c}),
$$

where the condition $\mathbf{c}$ could specify a target-binding surface, a structural motif, a topology, or symmetry requirements.

```text
Noise + optional design constraints
                  |
                  v
        RFdiffusion model
                  |
                  v
     Iterative backbone denoising
                  |
                  v
        New protein backbone
                  |
                  v
      Sequence design / testing
```

The original RFdiffusion formulation models residue-level rigid-body frames, including positions and orientations, rather than merely adding independent coordinate noise to C$\alpha$ positions. Designed backbones may subsequently be assigned amino-acid sequences using tools such as **ProteinMPNN**, after which structure prediction and experimental characterization can be used to assess the design.

RFdiffusion therefore illustrates a different scientific objective from conventional AlphaFold structure prediction: instead of asking how a **specified protein sequence** folds, it can propose a **new structural design** compatible with chosen constraints.

## 20. AlphaFold 3: Conditional Generative Structure Prediction

**AlphaFold 3** extends structure prediction to complexes containing proteins, nucleic acids, small molecules, ions, and modified residues. It introduces major changes to the AlphaFold 2 architecture, including the **Pairformer** and a **diffusion-based structure module** that operates on atomic coordinates.

The Pairformer processes representations of molecular tokens and their relationships. The diffusion module receives noisy atomic positions together with the representations produced by the preceding network and iteratively predicts denoised coordinates.

```text
Protein / nucleic-acid / ligand / ion information
                         |
                         v
                  Input features
                         |
                         v
                    Pairformer
                         |
                         v
          Conditional atomic structure model
                         +
                 Noisy coordinates
                         |
                         v
              Iterative denoising
                         |
                         v
             Predicted biomolecular complex
```

The model can be understood schematically as learning a conditional distribution

$$
p_{\theta}(\mathbf{R}\mid\mathbf{c}),
$$

where $\mathbf{R}$ denotes atomic coordinates and $\mathbf{c}$ captures sequence, molecular identities, and other context. Inference uses random initialization followed by recurrent denoising to produce a structural prediction.

### What makes AlphaFold 3 generative?

AlphaFold 3 has a **generative coordinate-production mechanism**: its diffusion module can sample a three-dimensional structure conditioned on the specified molecular system. Its scientific purpose, however, is primarily **conditional biomolecular structure prediction**, not unrestricted invention of new protein sequences.

This distinction can be summarized by the different questions the models answer:

**AlphaFold 2:** What structure is associated with this protein sequence?

**AlphaFold 3:** What joint atomic structure is compatible with these biomolecular components and their context, using diffusion-based coordinate generation?

**RFdiffusion:** What new protein backbone could satisfy these design constraints?

### Important limitations

Multiple samples produced with different random seeds should **not** automatically be interpreted as a Boltzmann-weighted or equilibrium conformational ensemble. AlphaFold 3 primarily models structures represented in its training data, and its authors explicitly caution that repeated sampling does not recover the physical solution ensemble. Generated structures may also be confidently ordered in regions that are actually disordered, or may miss biologically relevant conformational states.

Consequently, generative sampling must be distinguished from molecular dynamics, statistical-mechanical sampling, and experimental characterization of conformational distributions.

## 21. SimpleFold and Flow-Based Protein Structure Prediction

**SimpleFold** is a protein-folding approach that uses a generative **flow-matching objective** with general-purpose transformer blocks. In contrast to architectures centered on specialized pair-representation and triangular-update machinery, SimpleFold explores whether a comparatively general transformer design can perform competitive structure prediction through a learned generative flow.

A conceptual conditional flow for protein coordinates can be written as

$$
\frac{d\mathbf{R}_t}{dt}
=\mathbf{v}_{\theta}(\mathbf{R}_t,t,S),
$$

where $S$ is the protein sequence. Samples are transported along a learned trajectory toward protein-like coordinates that agree with the conditioning information.

SimpleFold is a useful comparison to AlphaFold 3 because it demonstrates that **conditional generative structure prediction** need not be implemented only through conventional discrete diffusion. Its reported results also motivate discussion of alternative architectural designs and the challenge of predicting structural ensembles.

The following comparison is particularly useful for students:

| Model | Primary task | Central architectural idea | Generative mechanism |
|:--|:--|:--|:--|
| AlphaFold 1 | Protein structure prediction | Learned geometrical restraints | Not a modern diffusion generator |
| AlphaFold 2 | Protein structure prediction | Evoformer and structure module | Generally not described as generative |
| AlphaFold 3 | Biomolecular complex prediction | Pairformer and coordinate diffusion module | Conditional diffusion |
| RFdiffusion | De novo protein-backbone design | RoseTTAFold-derived denoising | Diffusion-based generation |
| SimpleFold | Protein structure prediction | General-purpose transformer | Flow matching |

This classification describes the main scientific roles of the systems and should not be interpreted as implying that all their architectures or sampling objectives are equivalent.

## 22. Protein-Structure Prediction and Protein Design

Several related tasks are often grouped together under the broad term *protein modeling*, but they involve different inputs and outputs.

**Structure prediction** seeks the structure corresponding to a sequence and relevant context,

$$
S\longrightarrow\widehat{\mathbf{R}}.
$$

**Inverse folding or sequence design** seeks sequences compatible with a desired structure,

$$
\mathbf{R}_{\mathrm{target}}\longrightarrow S_{\mathrm{candidate}}.
$$

**De novo backbone generation** produces a new structural candidate from a learned model, often conditioned on a functional motif or target,

$$
\mathbf{z},\mathbf{c}\longrightarrow\mathbf{R}_{\mathrm{new}}.
$$

An end-to-end design workflow might therefore involve backbone generation, sequence assignment, structural validation, and experimental assessment. A model that predicts the structure of a proposed sequence may be very useful in such a workflow without itself being the generator of the sequence or backbone.

## 23. Evaluating Predictive and Generative Protein Models

The evaluation of structure-prediction models and generative-design models overlaps, but their main questions differ. In prediction, we may ask whether the proposed coordinates resemble a reference experimental structure. In design, we must also ask whether the new structure is plausible, diverse, novel, functionally appropriate, and experimentally realizable.

### Structural similarity and geometrical quality

For two aligned coordinate sets, a basic structural error is the **root-mean-square deviation**,

$$
\mathrm{RMSD}
=\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
\mathbf{r}_i-\hat{\mathbf{r}}_i
\right\|_2^2
}.
$$

RMSD depends on the atom mapping and alignment procedure. Other structural similarity measures, such as **TM-score**, are often useful because they account for protein-length and fold-similarity considerations differently.

A generated structure should also be checked for reasonable bond lengths and angles, steric clashes, stereochemical consistency, and physically plausible interfaces. For a coarse-grained C$\alpha$ backbone, adjacent C$\alpha$ separations are a useful initial diagnostic, but they are not sufficient to establish structural validity.

### Generative quality

**Validity** asks whether generated structures satisfy chemical and geometric constraints. **Diversity** asks whether repeated samples explore distinct solutions. **Novelty** asks whether samples differ meaningfully from known or training structures. **Conditional consistency** asks whether the output satisfies its specified motif, interface, or target requirements. **Functional performance** asks whether the designed object actually carries out the desired biological role.

Model-provided confidence scores, including AlphaFold's pLDDT-related predictions, are useful diagnostics of expected structural accuracy under the model's assumptions. They are **not** direct measures of binding affinity, thermodynamic stability, experimental function, or the probability that a de novo design will succeed in the laboratory.

??? info "Advanced material: Why many generated structures do not form a thermodynamic ensemble"

    A thermodynamic ensemble at equilibrium assigns conformational probabilities according to physical free-energy differences,

    $$
    p_{\mathrm{eq}}(\mathbf{R})
    \propto
    \exp\left[-\beta F(\mathbf{R})\right],
    \qquad
    \beta=\frac{1}{k_{\mathrm B}T}.
    $$

    A generative model instead samples from the distribution it has learned. Unless the model has been specifically trained or constrained to reproduce equilibrium statistics, there is no general reason for

    $$
    p_{\theta}(\mathbf{R}\mid\mathbf{c})
    =p_{\mathrm{eq}}(\mathbf{R}\mid\mathbf{c}).
    $$

    This is why multiple AlphaFold 3 predictions should not automatically be treated as an equilibrium ensemble.

## 24. Practical Exercises and Discussion Questions

### Exercise A: Prediction or generation?

Classify the role of machine learning in the following applications and justify your answer in one or two sentences:

1. Predict aqueous solubility from molecular descriptors.
2. Predict a structure from a specified protein sequence using AlphaFold 2.
3. Generate a new protein backbone with RFdiffusion under a motif constraint.
4. Produce atomic coordinates from the AlphaFold 3 diffusion module.
5. Generate new amino-acid sequences using an autoregressive language model.
6. Evaluate the expected quality of a proposed protein structure with a confidence model.

??? note "Suggested interpretation"

    The first, second, and sixth examples primarily concern **prediction or evaluation**. The third and fifth concern **generation**. The fourth uses a **conditional generative mechanism within a structure-prediction system**. The important distinction is the scientific task and the mechanism used to achieve it.

### Exercise B: Limiting cases of the diffusion equation

Starting from

$$
\mathbf{x}_t
=\sqrt{\bar\alpha_t}\,\mathbf{x}_0
+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,
$$

explain what happens as $\bar\alpha_t\rightarrow 1$ and as $\bar\alpha_t\rightarrow 0$.

??? note "Suggested solution"

    If $\bar\alpha_t\rightarrow 1$, then $\mathbf{x}_t\rightarrow\mathbf{x}_0$. The data are almost unchanged.

    If $\bar\alpha_t\rightarrow 0$, then $\mathbf{x}_t\rightarrow\boldsymbol\epsilon$, and information about the original example is largely lost.

    This intuition assumes the data have been represented and scaled appropriately for the chosen Gaussian noise process.

### Exercise C: Protein coordinate perturbation

Using the 3MUF example, change `alpha_bar` and compare the mean distance between consecutive C$\alpha$ atoms before and after adding noise. Why does the geometry rapidly cease to resemble a valid protein? What extra structural knowledge would a trained denoiser need?

### Exercise D: Conditional structure modeling

Suppose a model is asked to produce a protein that binds a known target. Identify possible conditioning variables and propose at least two independent ways to assess the generated candidate. Explain why a high predicted confidence score alone is insufficient to establish successful binding.

## 25. Computational Considerations and HPC Workflows

Large protein structure models often separate sequence processing, database searches, neural-network inference, and downstream analysis into distinct computational stages. For some AlphaFold 2-based workflows, generating multiple sequence alignments can be CPU- and I/O-intensive, while the structure-inference stage benefits from GPUs.

```text
Protein sequences / molecular components
                  |
                  v
      Data preparation and, when needed,
      sequence search / MSA generation
                  |
                  v
           Model features
                  |
                  v
     Accelerator-based model inference
                  |
                  v
       Predicted/generated structures
                  |
                  v
      Validation and downstream analysis
```

On an HPC system, computational planning should account for model weights, software environments, database storage, temporary files, GPU memory, job scheduling, and the number of independent prediction or generation jobs. Different tools have different requirements: an AlphaFold 2/ColabFold workflow may depend heavily on MSA preparation, whereas some newer models reduce that dependency.

The Python examples in this chapter demonstrate mathematical operations and lightweight training principles; they do **not** require running the full AlphaFold 3 or RFdiffusion models.

## 26. Summary

A generative model learns statistical structure from observed data and uses it to produce samples. Some models provide explicit probability densities, while others define implicit sampling mechanisms. Conditioning makes it possible to generate structures that depend on sequences, molecules, target properties, or geometric constraints.

Autoregressive models generate sequences step by step, VAEs sample from latent-variable models, GANs learn through an adversarial objective, diffusion models reverse a noising process, and flow matching learns continuous transport between distributions. For molecular structures, the validity of generated coordinates and the treatment of rotational and translational symmetry are central concerns.

The AlphaFold family highlights the difference between a **scientific task** and the **generative mechanism used to perform it**. AlphaFold 2 is primarily a structure-prediction system, AlphaFold 3 uses diffusion-based coordinate generation for conditional biomolecular structure prediction, and RFdiffusion is designed to generate new protein backbones. SimpleFold shows how flow matching can also be applied to protein structure modeling.

The major practical lesson is that producing a plausible structure is not equivalent to establishing chemical validity, physical equilibrium, biological function, or experimental success. Generative modeling is a powerful approach for proposing and exploring candidates, but careful evaluation remains essential.

## References and Suggested Reading

The following references cover both the foundational generative-model methods and the protein structure models discussed in the chapter.

1. Kingma, D. P. and Welling, M. *Auto-Encoding Variational Bayes* (2014). [arXiv:1312.6114](https://arxiv.org/abs/1312.6114).
2. Goodfellow, I. et al. *Generative Adversarial Nets* (2014). [arXiv:1406.2661](https://arxiv.org/abs/1406.2661).
3. Ho, J., Jain, A. and Abbeel, P. *Denoising Diffusion Probabilistic Models* (2020). [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
4. Lipman, Y. et al. *Flow Matching for Generative Modeling* (2023). [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
5. Senior, A. W. et al. *Improved protein structure prediction using potentials from deep learning*. **Nature 577**, 706–710 (2020). [DOI](https://doi.org/10.1038/s41586-019-1923-7).
6. Jumper, J. et al. *Highly accurate protein structure prediction with AlphaFold*. **Nature 596**, 583–589 (2021). [DOI](https://doi.org/10.1038/s41586-021-03819-2).
7. Mirdita, M. et al. *ColabFold: making protein folding accessible to all*. **Nature Methods 19**, 679–682 (2022). [DOI](https://doi.org/10.1038/s41592-022-01488-1).
8. Watson, J. L. et al. *De novo design of protein structure and function with RFdiffusion*. **Nature 620**, 1089–1100 (2023). [DOI](https://doi.org/10.1038/s41586-023-06415-8).
9. Abramson, J. et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3*. **Nature 630**, 493–500 (2024). [DOI](https://doi.org/10.1038/s41586-024-07487-w).
10. Wang, Y. et al. *SimpleFold: Folding Proteins is Simpler than You Think*. **ICLR 2026**. [arXiv:2509.18480](https://arxiv.org/abs/2509.18480).
11. For specialized AlphaFold-Multimer, AlphaPulldown, and AlphaFold Unmasked applications, consult their corresponding papers and official tool documentation.

??? info "Advanced material: score-based diffusion"

    The **score** of a continuously differentiable probability density is

    $$
    \mathbf{s}(\mathbf{x},t)
    =\nabla_{\mathbf{x}}\log p_t(\mathbf{x}).
    $$

    Score-based generative models learn an approximation to the score of noisy data distributions. This quantity is related to denoising and can be used to formulate reverse-time sampling procedures.

??? info "Advanced material: conditional denoising"

    A conditional noise-prediction network has the form

    $$
    \boldsymbol\epsilon_{\theta}
    (\mathbf{x}_t,t,\mathbf{c}),
    $$

    while a model trained to predict clean coordinates may instead take the form

    $$
    \widehat{\mathbf{R}}_0
    =f_{\theta}(\mathbf{R}_t,t,\mathbf{c}).
    $$

    Both are possible parameterizations of a denoising model, but they should not be conflated when describing a particular architecture such as AlphaFold 3.