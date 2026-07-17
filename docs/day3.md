# Foundations of Machine Learning for Molecular Systems

## Learning Objectives

- Understand the fundamental ML concepts relevant to molecular sciences
- Learn multiple molecular representation methods and their trade-offs
- Implement basic ML models for molecular property prediction
- Work with chemical databases and molecular descriptors
- Understand the unique challenges of applying ML to chemistry


## 1. Introduction to ML in Molecular Systems

### 1.1 Why Machine Learning for Molecules?

The chemical space is vast—estimates suggest there are $10^{60}$ possible drug-like molecules, far 
more than atoms in the universe (Med. Res. Rev., 16, 3-50, 1996). Traditional approaches to drug 
discovery and materials design involve:

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

- Explore $10^{60}$ possible molecules intelligently
- Focus experimental resources on most promising candidates
- Find novel scaffolds outside known chemistry

**Accelerate Discovery Pipelines**

- Traditional drug discovery: 10-15 years, $2.6B per drug
- ML-assisted discovery: Potentially 2-3x faster, significantly cheaper
- Example: Insilico Medicine designed a novel drug candidate in 46 days

### 1.2 Success Stories

**COVID-19 Drug Repurposing**

During the COVID-19 pandemic, machine learning models screened more than 6,000
FDA-approved drugs for activity against SARS-CoV-2. This approach identified
Baricitinib, a drug originally approved for rheumatoid arthritis, as a promising
candidate. It was subsequently authorized by the FDA for COVID-19 treatment in
2020, making it one of the earliest examples of ML-assisted drug repurposing
reaching clinical use.

**Antibiotic Discovery**

In 2020, a machine learning model trained on known antibacterial compounds
identified Halicin, a structurally novel antibiotic effective against several
drug-resistant bacteria. Because Halicin was discovered by learning patterns
from data rather than through traditional medicinal chemistry, its structure
is distinct from existing antibiotics; demonstrating that ML can explore
regions of chemical space that conventional approaches tend to overlook.

**Materials Science**

Machine learning has significantly accelerated materials discovery across
several domains. In battery research, ML models have identified promising
solid electrolyte candidates that would have taken far longer to find through
simulation alone. Thermal conductivity predictions that once required
computationally expensive simulations can now be made orders of magnitude
faster, and ML has also been applied to the search for new photovoltaic
materials with improved energy conversion properties.

### 1.3 Key Challenges in Molecular ML

#### High Dimensionality

Molecules exist in complex, high-dimensional spaces: each atom contributes
3D coordinates, electronic structure information, conformational degrees of
freedom, and quantum mechanical properties. Training models directly on such
representations is computationally intractable, so a key challenge is learning
compact representations that retain the essential features of molecular
structure.

#### Data Scarcity

Unlike computer vision, where models can be trained on millions of labeled
images, molecular datasets are typically small: a large drug discovery dataset
may contain between 1,000 and 100,000 compounds, and experimental measurements
are both expensive and time-consuming to obtain. Transfer learning, data
augmentation, and semi-supervised learning are the main strategies used to
compensate for limited labeled data.


#### Physical Constraints

Molecular machine learning models must respect fundamental physical constraints,
including conservation of energy, atomic valence rules, and the symmetries of
3D space — rotation, translation, and permutation of atoms. Models that violate
these constraints produce physically meaningless predictions. Physics-informed
neural networks and equivariant architectures address this by building the
relevant symmetries directly into the model.

#### Interpretability Requirements

A prediction that a molecule has a certain property is only useful if the model
can indicate which structural features drive that prediction, enabling
researchers to generate hypotheses and build confidence in the results.
Unlike many commercial applications, black-box predictions are insufficient in
science. Attention mechanisms, feature importance analysis, and explainable AI
methods are commonly used to make model decisions more transparent.


#### Distribution Shift

Models trained on one region of chemical space often fail when applied to
another; a problem known as distribution shift. Novel scaffolds, unfamiliar
functional groups, or property values outside the training range can all cause
performance to degrade severely. Domain adaptation, uncertainty quantification,
and active learning are the principal approaches for detecting and mitigating
this failure mode.


## 2. Molecular Representations

The choice of molecular representation is crucial; it determines what information is available to
the model and how efficiently it can learn.

### 2.1 SMILES (Simplified Molecular Input Line Entry System)

SMILES is a text-based notation that represents molecular structure as a string (J. Chem. Inf. Comput. Sci. 1988, 28, 1, 31–36).

#### Basic SMILES Syntax

**Simple molecules**:
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

**Double and triple bonds**:
```
Ethene:     C=C
Ethyne:     C#C
CO2:        O=C=O
```

**Rings**:
```
Cyclohexane:    C1CCCCC1
                └─ matching numbers close the ring

Naphthalene:    c1ccc2ccccc2c1
                └─ fused rings
```

**Stereochemistry**:
```
(R)-Alanine:    N[C@@H](C)C(=O)O
                  └─ @ and @@ encode the two tetrahedral configurations
```

#### Working with SMILES in Python

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Draw
    import matplotlib.pyplot as plt

    # Create molecule from SMILES
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)

    # heck validity
    if mol is None:
        print("Invalid SMILES!")
    else:
        print(f"Valid molecule with {mol.GetNumAtoms()} atoms")

    # Visualize
    img = Draw.MolToImage(mol, size=(300, 300))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Aspirin')
    #plt.show()
    plt.savefig('aspirin.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get canonical SMILES (standardized form)
    canonical_smiles = Chem.MolToSmiles(mol)
    print(f"Canonical SMILES: {canonical_smiles}")

    # Generate randomized SMILES (useful for data augmentation).
    # doRandom=True randomizes the atom order on each call, so every
    # iteration yields a different valid string. (canonical=False alone
    # would return the same non-canonical string every time.)
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

- **Not unique**: The same molecule can have multiple valid SMILES strings.

```python
# All represent ethanol:
smiles_variants = ["CCO", "OCC", "C(O)C"]
# Use canonical SMILES to obtain a unique, standardized representation
# for storage and comparison:
canonical = Chem.MolToSmiles(Chem.MolFromSmiles("OCC"))
# -> 'CCO'
```

- **No 3D information**: SMILES encodes connectivity but not geometry. Stereoisomers that differ only in 3D arrangement require explicit stereochemistry notation (@ / @@), and even then the string carries no atomic coordinates.

```python
# Same connectivity, opposite chirality (mirror images).
# The @ / @@ tags encode the two configurations at the stereocenter;
# swapping them inverts the chirality. SMILES still stores no coordinates.
alanine_a = "N[C@@H](C)C(=O)O"
alanine_b = "N[C@H](C)C(=O)O"
# RDKit can assign the CIP (R/S) label:
#   from rdkit import Chem
#   m = Chem.MolFromSmiles(alanine_a)
#   Chem.AssignStereochemistry(m, cleanIt=True, force=True)
#   print(m.GetAtomWithIdx(1).GetPropsAsDict().get("_CIPCode"))
```

- **Sequence-based**: Hard to capture graph structure directly.

- **Fragile**: A single character error invalidates the entire SMILES string.

```python
valid = "CCO"
invalid = "C CO"  # Space breaks it
```

### 2.2 SELFIES (Self-Referencing Embedded Strings)

SELFIES is an alternative to SMILES that guarantees 100% valid molecules (Mach. Learn Sci. Technol. 1, 045024, 2020).

??? note "Example"

    ```python
    import selfies as sf
    from rdkit import Chem

    # Convert SMILES to SELFIES
    smiles = "CCO"
    selfies_str = sf.encoder(smiles)
    print(f"SMILES: {smiles}")
    print(f"SELFIES: {selfies_str}")

    # Convert SELFIES back to SMILES
    smiles_back = sf.decoder(selfies_str)
    print(f"Back to SMILES: {smiles_back}")

    # Verify it's a valid molecule
    mol = Chem.MolFromSmiles(smiles_back)
    print(f"Valid molecule: {mol is not None}")
    ```

### 2.3 Molecular Fingerprints

Fingerprints are fixed-length binary or count vectors that encode molecular structure.

#### Morgan fingerprints (ECFP, Extended-Connectivity Fingerprints)

Morgan fingerprints, also known as Extended-Connectivity Fingerprints (ECFP), are one
of the most widely used molecular representations in cheminformatics and molecular machine
learning. They describe a molecule by examining the local chemical environment around each
atom. Instead of representing the molecule as a whole structure, Morgan fingerprints break
it into many small circular neighborhoods centered on individual atoms.

The main idea is that each atom is first assigned an identifier based on its local properties,
such as atom type, bonding pattern, and connectivity. The algorithm then expands outward step
by step, collecting information from neighboring atoms at increasing radii. These local environments
are converted into numerical identifiers and stored in a fixed-length fingerprint vector. The
final result is a numerical representation that can be used for similarity search, clustering,
classification, regression, or other machine learning tasks.

**Algorithm**:

1. Assign an initial identifier to each atom based on its chemical properties.
2. Expand around each atom to include neighboring atoms within a chosen radius.
3. Update the atom identifiers based on the surrounding chemical environment.
4. Hash the resulting identifiers into a fixed-length fingerprint vector.
5. Use the fingerprint as input for similarity analysis or machine learning models.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    import numpy as np

    # 1. Create molecule from SMILES
    smiles = "CCO"  # Ethanol
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    # 2. Create Morgan fingerprint generator
    morgan_gen = GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    # 3. Generate Morgan fingerprint as a bit vector
    morgan_fp = morgan_gen.GetFingerprint(mol)

    # Convert bit vector to NumPy array
    fp_array = np.array(morgan_fp)

    print("Morgan bit fingerprint")
    print("Fingerprint shape:", fp_array.shape)
    print("Number of set bits:", int(fp_array.sum()))

    # 4. Generate Morgan count fingerprint
    count_fp_array = morgan_gen.GetCountFingerprintAsNumPy(mol)

    print("\nMorgan count fingerprint")
    print("Fingerprint shape:", count_fp_array.shape)
    print("Total feature counts:", int(count_fp_array.sum()))
    print("Number of nonzero features:", np.count_nonzero(count_fp_array))
    ```

A bit fingerprint stores whether a molecular feature is present or absent. A count fingerprint stores
how many times each feature appears. For many introductory examples, bit fingerprints are easier to explain,
while count fingerprints can provide more detailed information for machine learning.

**Parameters**:

- **Radius**: Larger radius captures more context

  - Radius 1 (ECFP2): Immediate neighbors
  - Radius 2 (ECFP4): Common choice, balances local and broader context
  - Radius 3 (ECFP6): Larger substructures

- **fpSize**: Fingerprint length

  - 1024: Fast, but more collisions
  - 2048: Common default
  - 4096: More unique features, slower computationally

#### MACCS Keys

MACCS keys are structural fingerprints based on 166 predefined chemical patterns
commonly found in molecular structures. Each bit indicates the presence or absence of a
specific substructure, functional group, or bonding pattern, making MACCS keys useful for
molecular similarity analysis, clustering, and cheminformatics applications. Note that RDKit
returns a 167-bit vector, since index 0 is unused and the 166 defined keys occupy indices 1–166.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    import numpy as np

    # 1. Create molecule
    mol = Chem.MolFromSmiles("CCO")  # Ethanol

    # 2. Generate MACCS fingerprint
    maccs = MACCSkeys.GenMACCSKeys(mol)

    # RDKit returns 167 bits (index 0 unused; 166 defined keys)
    print(f"MACCS keys length: {len(maccs)}")

    # 3. Structural feature interpretation
    # Each bit flags a predefined substructure pattern (element types,
    # ring systems, functional groups, connectivity motifs, and so on).
    # The exact SMARTS definitions are part of RDKit's MACCS key set.

    # 4. Convert fingerprint to NumPy array
    maccs_array = np.array(
        list(maccs.ToBitString()),
        dtype=int
    )

    print("\nFingerprint shape:", maccs_array.shape)
    print("Number of active bits:", maccs_array.sum())
    ```

**Advantages**:

- Interpretable: Each bit has a defined meaning
- Compact: 167-bit vector (166 defined keys)
- Good for similarity searching

**Limitations**:

- Fixed features: Cannot capture patterns outside the predefined key set
- Less flexible than Morgan fingerprints

#### RDKit Fingerprints

RDKit fingerprints are topological molecular fingerprints that encode structural information
by analyzing atom paths and bond connectivity within a molecule. They are commonly used for
molecular similarity searches, clustering, and cheminformatics machine learning applications.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import RDKFingerprint
    import numpy as np

    # 1. Create molecule
    mol = Chem.MolFromSmiles("CCO")  # Ethanol

    # 2. Generate RDKit fingerprint
    rdkit_fp = RDKFingerprint(
        mol,
        fpSize=2048,
        maxPath=7
    )

    # maxPath:
    # Maximum bond path length considered
    # when generating structural patterns

    # 3. Convert to NumPy array
    fp_array = np.array(
        list(rdkit_fp.ToBitString()),
        dtype=int
    )

    print("Fingerprint shape:", fp_array.shape)
    print("Number of active bits:", fp_array.sum())
    ```

### Atom Pair and Topological Torsion Fingerprints

Atom pair fingerprints encode topological distances between pairs of atoms, while topological
torsion fingerprints capture sequential four-atom connectivity patterns.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    # Create molecule
    mol = Chem.MolFromSmiles("CCO")  # Ethanol

    # Atom pair fingerprint
    atom_pairs = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
        mol,
        nBits=2048
    )

    # Topological torsion fingerprint
    torsions = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        mol,
        nBits=2048
    )

    print("Atom pair fingerprint length:", len(atom_pairs))
    print("Topological torsion fingerprint length:", len(torsions))
    ```

#### Comparing Molecules with Fingerprints

Fingerprint similarity metrics quantify structural similarity between molecules by comparing
shared molecular features encoded in fingerprint vectors.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    # 1. Create molecules
    mol1 = Chem.MolFromSmiles("CCO")        # Ethanol
    mol2 = Chem.MolFromSmiles("CCCO")       # Propanol
    mol3 = Chem.MolFromSmiles("c1ccccc1")   # Benzene

    # 2. Generate Morgan fingerprints

    fp1 = AllChem.GetMorganFingerprintAsBitVect(
        mol1,
        radius=2,
        nBits=2048
    )

    fp2 = AllChem.GetMorganFingerprintAsBitVect(
        mol2,
        radius=2,
        nBits=2048
    )

    fp3 = AllChem.GetMorganFingerprintAsBitVect(
        mol3,
        radius=2,
        nBits=2048
    )

    # 3. Compute similarity metrics

    # Tanimoto similarity
    sim_12 = DataStructs.TanimotoSimilarity(fp1, fp2)
    sim_13 = DataStructs.TanimotoSimilarity(fp1, fp3)

    print(f"Similarity (ethanol, propanol): {sim_12:.3f}")
    print(f"Similarity (ethanol, benzene):  {sim_13:.3f}")

    # Additional similarity metrics
    dice = DataStructs.DiceSimilarity(fp1, fp2)
    cosine = DataStructs.CosineSimilarity(fp1, fp2)

    print(f"\nDice similarity:   {dice:.3f}")
    print(f"Cosine similarity: {cosine:.3f}")
    ```

### 2.4 3D Molecular Representations

While fingerprints and graph-based methods describe molecular connectivity, many molecular
properties also depend strongly on three-dimensional geometry. 3D molecular representations
include spatial information such as atomic coordinates, bond distances, angles, and molecular
conformations. These representations are especially important for applications involving
molecular dynamics, docking, quantum chemistry, protein-ligand interactions, and materials modeling.

Common 3D representations include:

* Cartesian coordinates
* Distance matrices
* Coulomb matrices
* Atomic environments
* Molecular conformations

#### Example: Generating 3D Coordinates with RDKit

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Create molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")  # Ethanol

    # Add hydrogen atoms
    mol = Chem.AddHs(mol)

    # Generate 3D conformation (fixed seed for reproducibility)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    # Optimize geometry
    AllChem.UFFOptimizeMolecule(mol)

    # Print atomic coordinates
    conf = mol.GetConformer()

    print("Atomic coordinates:\n")

    for atom in mol.GetAtoms():

        pos = conf.GetAtomPosition(atom.GetIdx())

        print(
            f"Atom {atom.GetSymbol():2s} "
            f"-> x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}"
        )
    ```

### 2.5 Protein Representations

Proteins are complex biological macromolecules that can be represented in several different
ways for machine learning applications. Depending on the problem, proteins may be described using
amino acid sequences, structural information, residue contact maps, graphs, embeddings, or atomistic
coordinates. Choosing an appropriate representation is essential for tasks such as protein
structure prediction, molecular dynamics, function prediction, and protein-ligand interaction modeling.

Common protein representations include:

* Amino acid sequences
* One-hot encodings
* Protein language model embeddings
* Contact maps
* Graph representations
* Atomic coordinate representations

#### Example: Loading a Protein Structure with ASE

??? note "Example"

    ```python
    from ase.io import read

    # Load protein structure from PDB file
    protein = read("protein.pdb")

    print("Number of atoms:", len(protein))

    print("\nFirst five atoms:\n")

    for atom in protein[:5]:

        print(
            f"{atom.symbol:2s} "
            f"x={atom.position[0]:8.3f} "
            f"y={atom.position[1]:8.3f} "
            f"z={atom.position[2]:8.3f}"
        )
    ```

This example demonstrates how atomistic protein structures can be loaded and manipulated using
[ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase) for
scientific computing and machine learning workflows.

The Coulomb matrix representation is described in PRL 108, 058301 (2012):

??? note "Example"

    ```python
    # Coulomb matrix representation for a small protein-like backbone fragment

    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Define a small protein-like fragment
    # For simplicity, we use backbone atoms from two residues:
    # N, CA, C, O, N, CA, C, O
    #
    # In a real protein, these coordinates would usually come
    # from a PDB file. (CA is a carbon, so it appears as "C" below.)

    atom_symbols = np.array([
        "N", "C", "C", "O",
        "N", "C", "C", "O"
    ])

    coordinates = np.array([
        [0.00, 0.00, 0.00],   # N
        [1.45, 0.10, 0.00],   # CA
        [2.10, 1.45, 0.00],   # C
        [1.55, 2.50, 0.00],   # O

        [3.45, 1.40, 0.00],   # N
        [4.20, 2.65, 0.10],   # CA
        [5.65, 2.30, 0.00],   # C
        [6.10, 1.20, 0.00]    # O
    ])

    # Atomic numbers
    atomic_numbers = {
        "H": 1,
        "C": 6,
        "N": 7,
        "O": 8,
        "S": 16
    }

    Z = np.array([atomic_numbers[symbol] for symbol in atom_symbols])

    # 2. Compute Coulomb matrix

    def coulomb_matrix(Z, coordinates):
        """
        Compute the Coulomb matrix.

        Diagonal terms:
            0.5 * Z_i^2.4

        Off-diagonal terms:
            Z_i * Z_j / distance(i, j)
        """

        n_atoms = len(Z)

        C = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(n_atoms):

                if i == j:
                    C[i, j] = 0.5 * Z[i] ** 2.4

                else:
                    distance = np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )

                    C[i, j] = Z[i] * Z[j] / distance

        return C

    C = coulomb_matrix(Z, coordinates)

    print("Coulomb matrix shape:", C.shape)
    print("\nCoulomb matrix:")
    print(np.round(C, 2))

    # 3. Convert matrix into a machine learning feature vector
    # One common option is to flatten the matrix.
    # For fixed-size systems, this can be used directly as an ML input.

    feature_vector = C.flatten()

    print("\nFeature vector shape:", feature_vector.shape)

    # 4. Optional: use the sorted eigenvalues as a compact representation
    # Eigenvalues are useful because they provide a fixed-length
    # summary of the matrix and are invariant to atom ordering.

    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]

    print("\nSorted Coulomb matrix eigenvalues:")
    print(np.round(eigenvalues, 3))

    # 5. Visualize the Coulomb matrix

    plt.figure(figsize=(6, 5))

    plt.imshow(C)

    plt.colorbar(label="Coulomb matrix value")

    plt.xticks(
        ticks=np.arange(len(atom_symbols)),
        labels=atom_symbols
    )

    plt.yticks(
        ticks=np.arange(len(atom_symbols)),
        labels=atom_symbols
    )

    plt.title("Coulomb matrix for a protein-like fragment")
    plt.xlabel("Atom index")
    plt.ylabel("Atom index")

    plt.tight_layout()
    plt.savefig("coulomb.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

### 2.6 Molecular Descriptors

Numerical features that capture molecular properties.

#### Types of descriptors

**1. Physical descriptors**

Physical descriptors quantify fundamental molecular properties related to size,
polarity, hydrophobicity, and intermolecular interactions in chemical systems.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen

    # 1. Create molecule (Aspirin)
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Molecular weight
    mw = Descriptors.MolWt(mol)
    print(f"Molecular Weight: {mw:.2f} g/mol")

    # 3. Lipophilicity (LogP)
    # Octanol/water partition coefficient
    logp = Crippen.MolLogP(mol)
    print(f"LogP: {logp:.2f}")

    # Lipinski guideline:
    # LogP > 5 may indicate excessive lipophilicity

    # 4. Topological Polar Surface Area (TPSA)
    tpsa = Descriptors.TPSA(mol)
    print(f"TPSA: {tpsa:.2f} Å²")

    # Lower TPSA values are often associated with
    # better membrane permeability

    # 5. Molar Refractivity
    mr = Crippen.MolMR(mol)
    print(f"Molar Refractivity: {mr:.2f}")
    ```

**2. Structural descriptors**

Structural descriptors characterize molecular topology, flexibility, ring systems,
hydrogen bonding capacity, and atomic connectivity patterns influencing chemical behavior.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # 1. Create molecule (Aspirin)
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Hydrogen bond donors and acceptors
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    print(f"H-Bond Donors: {h_donors}")
    print(f"H-Bond Acceptors: {h_acceptors}")

    # 3. Rotatable bonds (measures molecular flexibility)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    print(f"Rotatable Bonds: {rot_bonds}")

    # 4. Ring information
    num_rings = Descriptors.RingCount(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    print(f"Total Rings: {num_rings}")
    print(f"Aromatic Rings: {aromatic_rings}")

    # 5. Fraction of sp3 carbons
    # Indicates molecular saturation and 3D character
    frac_sp3 = Descriptors.FractionCSP3(mol)
    print(f"Fraction Csp3: {frac_sp3:.2f}")
    ```

**3. Topological descriptors**

Topological descriptors quantify molecular connectivity, branching, complexity,
and graph structure independently of three-dimensional molecular geometry or coordinates.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import GraphDescriptors

    # 1. Create molecule (Aspirin)
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Balaban J index (branching and connectivity)
    balaban = GraphDescriptors.BalabanJ(mol)
    print(f"Balaban J Index: {balaban:.3f}")

    # 3. Bertz complexity index (structural complexity)
    bertz = GraphDescriptors.BertzCT(mol)
    print(f"Bertz Complexity Index: {bertz:.3f}")

    # 4. Chi connectivity indices
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)
    print(f"Chi0 Index: {chi0:.3f}")
    print(f"Chi1 Index: {chi1:.3f}")
    ```

**4. 3D descriptors**

3D descriptors characterize molecular shape, spatial distribution, geometry, and
conformational properties using three-dimensional atomic coordinates and optimized
molecular structures.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D

    # 1. Create molecule (Aspirin)
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Generate 3D molecular structure
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol_3d)

    # 3. Compute 3D descriptors

    # Deviation from spherical shape
    asphericity = Descriptors3D.Asphericity(mol_3d)

    # Measure of molecular elongation
    eccentricity = Descriptors3D.Eccentricity(mol_3d)

    # Shape and mass distribution descriptor
    inertial_shape = Descriptors3D.InertialShapeFactor(mol_3d)

    # Spatial extent of the atomic distribution
    radius_of_gyration = Descriptors3D.RadiusOfGyration(mol_3d)

    # 4. Display results
    print(f"Asphericity: {asphericity:.3f}")
    print(f"Eccentricity: {eccentricity:.3f}")
    print(f"Inertial Shape Factor: {inertial_shape:.3f}")
    print(f"Radius of Gyration: {radius_of_gyration:.3f} Å")
    ```

#### Drug-Likeness Metrics

Drug-likeness metrics evaluate whether a molecule possesses physicochemical and
structural properties commonly associated with successful pharmaceutical compounds.

**Lipinski's rule of five**

Lipinski's rule of five estimates oral bioavailability using molecular weight,
hydrogen bonding, and lipophilicity-based physicochemical thresholds.

**QED (Quantitative Estimate of Drug-likeness)**

QED combines multiple molecular descriptors into a single score representing the
overall drug-like character of a compound.

#### Creating Feature Vectors

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, GraphDescriptors, QED
    import pandas as pd

    def calculate_molecular_descriptors(smiles):
        """
        Calculate a collection of common molecular descriptors.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        descriptors = {
            # Physical
            "MW": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "MolMR": Crippen.MolMR(mol),

            # Structural
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "RingCount": Descriptors.RingCount(mol),

            # Complexity
            "BertzCT": GraphDescriptors.BertzCT(mol),
            "NumBridgeheadAtoms": Descriptors.NumBridgeheadAtoms(mol),
            "NumSpiroAtoms": Descriptors.NumSpiroAtoms(mol),

            # Surface-area descriptors
            "LabuteASA": Descriptors.LabuteASA(mol),
            "PEOE_VSA1": Descriptors.PEOE_VSA1(mol),

            # Atom counts
            "NumCarbon": sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()),
            "NumNitrogen": sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
            "NumOxygen": sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
            "NumHalogens": sum(atom.GetAtomicNum() in [9, 17, 35, 53] for atom in mol.GetAtoms()),

            # Saturation
            "FractionCsp3": Descriptors.FractionCSP3(mol),

            # Drug-likeness
            "QED": QED.qed(mol),
        }

        return descriptors


    # Example usage
    smiles_list = [
        "CCO",                              # Ethanol
        "CC(=O)Oc1ccccc1C(=O)O",           # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"     # Caffeine
    ]

    descriptor_list = [
        calculate_molecular_descriptors(smiles)
        for smiles in smiles_list
    ]

    df_descriptors = pd.DataFrame(descriptor_list)
    df_descriptors.insert(0, "SMILES", smiles_list)

    print(df_descriptors)
    ```

### 2.7 Graph Representations

Molecular graphs represent atoms as nodes and chemical bonds as edges, preserving
connectivity and atom-level information.

#### Graph Structure

Molecules can be represented as graphs where atoms are nodes and bonds are edges.

??? note "Example"

    ```python
    from rdkit import Chem
    import networkx as nx

    def mol_to_graph(smiles):
        """Convert a molecule into a NetworkX graph."""

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError("Invalid SMILES string")

        G = nx.Graph()

        # Add nodes: atoms
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

        # Add edges: bonds
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
    print(f"Node 0 features: {G.nodes[0]}")
    ```

#### Adjacency matrix representation

??? note "Example"

    ```python
    from rdkit import Chem
    import numpy as np

    def get_adjacency_matrix(smiles, max_atoms=50):
        """Generate a padded molecular adjacency matrix."""

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError("Invalid SMILES string")

        num_atoms = mol.GetNumAtoms()

        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule contains {num_atoms} atoms "
                f"but max_atoms={max_atoms}"
            )

        # Initialize adjacency matrix
        adj_matrix = np.zeros(
            (max_atoms, max_atoms),
            dtype=int
        )

        # Fill connectivity
        for bond in mol.GetBonds():

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Symmetric matrix

        return adj_matrix, num_atoms

    # Example
    adj, n_atoms = get_adjacency_matrix("CCO")

    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Actual atoms: {n_atoms}")

    print("\nAdjacency matrix:")
    print(adj[:n_atoms, :n_atoms])
    ```

#### Node and Edge Features

Node and edge features encode atom and bond properties as numerical vectors
for graph-based molecular machine learning models.

??? note "Example"

    ```python
    from rdkit import Chem
    import numpy as np

    def get_node_features(atom):
        """Extract numerical features for one atom."""

        hybridization_map = {
            Chem.rdchem.HybridizationType.SP: 1,
            Chem.rdchem.HybridizationType.SP2: 2,
            Chem.rdchem.HybridizationType.SP3: 3,
            Chem.rdchem.HybridizationType.SP3D: 4,
            Chem.rdchem.HybridizationType.SP3D2: 5,
        }

        return np.array([
            atom.GetAtomicNum(),                     # Atomic number
            atom.GetDegree(),                        # Number of bonded neighbors
            atom.GetFormalCharge(),                  # Formal charge
            atom.GetNumRadicalElectrons(),           # Radical electrons
            hybridization_map.get(
                atom.GetHybridization(), 0
            ),                                       # Hybridization state
            int(atom.GetIsAromatic()),               # Aromaticity
            atom.GetTotalNumHs(),                    # Attached hydrogens
        ], dtype=float)


    def get_edge_features(bond):
        """Extract numerical features for one bond."""

        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }

        return np.array([
            bond_type_map.get(
                bond.GetBondType(), 0
            ),                                       # Bond type
            int(bond.GetIsConjugated()),             # Conjugation
            int(bond.GetIsAromatic()),               # Aromaticity
        ], dtype=float)


    # Example
    mol = Chem.MolFromSmiles("CCO")

    print("Node features:\n")

    for atom in mol.GetAtoms():
        print(
            atom.GetSymbol(),
            get_node_features(atom)
        )

    print("\nEdge features:\n")

    for bond in mol.GetBonds():
        print(
            f"{bond.GetBeginAtomIdx()} - "
            f"{bond.GetEndAtomIdx()}:",
            get_edge_features(bond)
        )
    ```

The following example builds a protein graph from residue C$^{\alpha}$ atoms. Note that it uses a
4.5 Å cutoff, which here mainly connects sequential residues; residue-level contact maps
more commonly use a cutoff of around 8 Å between C$^{\alpha}$ atoms.

??? note "Example"

    ```python
    # Protein graph example using residue C-alpha atoms

    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    # 1. Example protein residues
    # Each residue has:
    # - residue name
    # - residue index
    # - C-alpha 3D coordinates

    residues = [
        {"name": "ALA", "index": 1, "ca_coord": np.array([0.0, 0.0, 0.0])},
        {"name": "GLY", "index": 2, "ca_coord": np.array([3.8, 0.2, 0.0])},
        {"name": "SER", "index": 3, "ca_coord": np.array([7.5, 0.1, 0.3])},
        {"name": "VAL", "index": 4, "ca_coord": np.array([5.0, 3.5, 0.2])},
        {"name": "LYS", "index": 5, "ca_coord": np.array([1.5, 3.2, 0.1])},
    ]

    # 2. Create graph
    G = nx.Graph()

    # Add residues as nodes
    for residue in residues:
        G.add_node(
            residue["index"],
            residue_name=residue["name"],
            ca_coord=residue["ca_coord"]
        )

    # 3. Add edges based on distance cutoff
    distance_cutoff = 4.5

    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):

            coord_i = residues[i]["ca_coord"]
            coord_j = residues[j]["ca_coord"]

            distance = np.linalg.norm(coord_i - coord_j)

            if distance <= distance_cutoff:
                G.add_edge(
                    residues[i]["index"],
                    residues[j]["index"],
                    distance=distance
                )

    # 4. Print graph information
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    print("\nNodes:")
    for node, features in G.nodes(data=True):
        print(node, features)

    print("\nEdges:")
    for u, v, features in G.edges(data=True):
        print(f"{u} -- {v}, distance = {features['distance']:.2f} Å")

    # 5. Visualize graph
    pos = {
        residue["index"]: residue["ca_coord"][:2]
        for residue in residues
    }

    labels = {
        residue["index"]: residue["name"]
        for residue in residues
    }

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=900
    )

    plt.title("Protein graph representation")
    plt.savefig("graph-protein.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

**Advantages of Graph Representations**:

- Natural for molecules (atoms connected by bonds)
- Permutation invariant (atom order doesn't matter)
- Captures topology and local structure
- Enables Graph Neural Networks

**Limitations**:

- More complex to implement
- Computationally expensive for large molecules
- Requires specialized neural network architectures


## 3. Traditional Machine Learning Methods

Before deep learning, these methods were (and still are) workhorses of molecular machine learning.

### 3.1 Feature engineering principles

**Domain knowledge is key**:

- Choose descriptors relevant to the property being predicted
- For solubility: polarity, surface area, H-bond capacity
- For toxicity: reactive functional groups, lipophilicity

**Feature scaling**:

??? note "Example"

    ```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split

    # Split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Standardization: zero mean, unit variance
    standard_scaler = StandardScaler()

    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    # Min-Max scaling: range [0, 1]
    minmax_scaler = MinMaxScaler()

    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    # Robust scaling: uses median and interquartile range
    robust_scaler = RobustScaler()

    X_train_robust = robust_scaler.fit_transform(X_train)
    X_test_robust = robust_scaler.transform(X_test)
    ```

Note that the scaler is always fit on the training set only and then applied to
the test set. Fitting on the full dataset would leak information from the test
set into the model.

**Feature selection**:

??? note "Example"

    ```python
    from sklearn.feature_selection import (
        VarianceThreshold,
        SelectKBest,
        f_regression,
        RFE,
        SelectFromModel
    )

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.01)

    X_train_high_var = variance_selector.fit_transform(X_train)
    X_test_high_var = variance_selector.transform(X_test)

    # Select top 10 features using univariate regression scores
    kbest_selector = SelectKBest(
        score_func=f_regression,
        k=10
    )

    X_train_selected = kbest_selector.fit_transform(X_train, y_train)
    X_test_selected = kbest_selector.transform(X_test)

    # Recursive Feature Elimination
    rfe_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rfe_selector = RFE(
        estimator=rfe_model,
        n_features_to_select=10
    )

    X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
    X_test_rfe = rfe_selector.transform(X_test)

    # Feature importance from a fitted model
    importance_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    importance_model.fit(X_train, y_train)

    # With prefit=True, the estimator is already fitted, so we call
    # transform directly (without fitting the selector again).
    importance_selector = SelectFromModel(
        importance_model,
        prefit=True,
        threshold="median"
    )

    X_train_important = importance_selector.transform(X_train)
    X_test_important = importance_selector.transform(X_test)
    ```

### 3.2 Random Forests

An ensemble of decision trees, each trained on a random subset of the data and features.

#### How it works

1. **Bootstrap sampling**: Create N random subsets of the training data (with replacement)
2. **Random feature selection**: At each split, consider a random subset of features
3. **Tree building**: Build deep trees without pruning
4. **Prediction**: Average the predictions from all trees (regression) or take a majority vote (classification)

Averaging many decorrelated trees is what reduces variance: individual deep
trees overfit, but their errors partially cancel when combined.

??? note "Example"

    ```python
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Create example regression dataset
    X, y = make_regression(
        n_samples=500,
        n_features=12,
        n_informative=6,
        noise=15,
        random_state=42
    )

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Define random forest regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,      # out-of-bag estimate (see note below)
        random_state=42,
        n_jobs=-1
    )

    # 4. Train model
    rf_reg.fit(X_train, y_train)

    # 5. Predict and evaluate
    y_pred = rf_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test R²: {r2:.3f}")
    print(f"Test MSE: {mse:.3f}")
    print(f"Out-of-bag R²: {rf_reg.oob_score_:.3f}")

    # 6. Cross-validation
    cv_scores = cross_val_score(
        rf_reg,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 7. Feature importance
    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i + 1}. Feature {idx}: {importances[idx]:.4f}")

    # 8. Visualize feature importance
    plt.figure(figsize=(10, 6))

    plt.bar(
        range(len(importances)),
        importances[indices]
    )

    plt.xticks(
        range(len(importances)),
        indices
    )

    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()
    plt.savefig("random-forest.png", dpi=300, bbox_inches="tight")
    plt.show()
    ```

**Out-of-bag estimation**: Because each tree is trained on a bootstrap sample,
roughly one-third of the training data is left out ("out of bag") for each tree.
These held-out samples provide a built-in validation estimate at no extra cost,
enabled with `oob_score=True` and read from `rf_reg.oob_score_`.

**A note on feature importance**: The `feature_importances_` attribute reports
impurity-based (Gini) importance, which is fast but biased toward high-cardinality
and continuous features. For a more reliable ranking, use permutation importance,
which measures the drop in performance when a feature's values are shuffled:

??? note "Example"

    ```python
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        rf_reg,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    for idx in result.importances_mean.argsort()[::-1][:10]:
        print(
            f"Feature {idx}: "
            f"{result.importances_mean[idx]:.4f} "
            f"± {result.importances_std[idx]:.4f}"
        )
    ```

**Advantages**:

- Handles non-linear relationships
- Robust to outliers
- Provides feature importance
- Little hyperparameter tuning needed
- Reduces variance relative to a single deep tree

**Limitations**:

- Can be slow for very large datasets
- Poor at extrapolation beyond the range of the training data
- Hard to interpret individual predictions

**Hyperparameter Tuning**:

??? note "Example"

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

#### Regression metrics

For $n$ samples with true values $y_i$ and predictions $\hat{y}_i$, the most common
regression metrics are defined as follows.

The mean absolute error (MAE) is the average absolute deviation, in the same units
as the target:

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

The root mean squared error (RMSE) also has the units of the target, but penalizes
large errors more heavily because of the square:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
$$

The coefficient of determination $R^2$ measures the fraction of variance explained,
relative to a baseline that always predicts the mean $\bar{y}$:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}
                {\sum_{i=1}^{n} \left( y_i - \bar{y} \right)^2}
$$

An $R^2$ of 1 indicates perfect predictions, while $R^2 = 0$ corresponds to a model
no better than predicting the mean. Note that $R^2$ can be negative when the model
performs worse than this baseline.

??? note "Example"

    ```python
    import numpy as np
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

### 3.4 QSAR (Quantitative Structure–Activity Relationship)

QSAR models relate molecular structure to biological or chemical activity using
numerical descriptors. Because they operate on molecular descriptors or fingerprints
rather than on raw structures, QSAR is considered a classical machine learning
approach in cheminformatics, typically combining molecular features with regression
or classification algorithms.

Typical QSAR applications include:

* predicting drug activity,
* toxicity prediction,
* solubility estimation,
* binding affinity prediction,
* environmental risk assessment.

A QSAR workflow generally proceeds in three steps:

1. molecules are converted into descriptors or fingerprints,
2. the descriptors are used as input features,
3. a regression or classification model predicts the target property.

??? note "Example"

    ```python
    # QSAR regression example.
    # Note: this dataset is tiny and purely illustrative — with only a
    # handful of molecules, the reported metrics are not statistically
    # meaningful and serve only to demonstrate the workflow.

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    # 1. Example molecular descriptor dataset
    data = pd.DataFrame({
        "molecular_weight": [46.07, 60.05, 78.11, 180.16, 194.19, 151.16],
        "logP": [-0.31, -0.17, 2.13, 1.19, -0.07, 1.35],
        "h_bond_donors": [1, 1, 0, 1, 0, 1],
        "h_bond_acceptors": [1, 2, 0, 4, 6, 2],
        "activity": [1.2, 1.6, 0.4, 2.8, 3.1, 2.2]
    })

    X = data.drop(columns="activity")
    y = data["activity"]

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=42
    )

    # 3. Train QSAR model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. Predict and evaluate
    y_pred = model.predict(X_test)

    print("Predicted activity:", y_pred)
    print("True activity:", y_test.values)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    ```

## 4. Working with Chemical Databases

### 4.1 Public Databases

#### PubChem

??? note "Example"

    ```python
    import pubchempy as pcp

    # 1. Search compound by name
    results = pcp.get_compounds(
        "aspirin",
        "name"
    )

    if len(results) == 0:
        raise ValueError("No compound found")

    compound = results[0]

    # 2. Display basic information

    print("Compound Information\n")
    print(f"IUPAC Name:        {compound.iupac_name}")
    print(f"SMILES:            {compound.smiles}")
    print(f"Molecular Formula: {compound.molecular_formula}")
    print(f"Molecular Weight:  {compound.molecular_weight}")

    # 3. Retrieve selected properties
    # Look up by CID to reuse the compound we already found
    # (avoids a second name search).
    properties = pcp.get_properties(
        [
            "MolecularWeight",
            "XLogP",
            "TPSA",
            "Complexity"
        ],
        compound.cid,
        "cid"
    )

    # 4. Display properties
    print("\nSelected Properties\n")

    for key, value in properties[0].items():
        print(f"{key}: {value}")
    ```

#### ChEMBL

??? note "Example"

    ```python
    # ChEMBL example:
    # Search for aspirin and retrieve related bioactivity data

    import pandas as pd
    from chembl_webresource_client.new_client import new_client

    # 1. Search for a molecule by name
    molecule = new_client.molecule
    results = molecule.search("aspirin")

    if not results:
        raise ValueError("No molecule found")

    aspirin = results[0]

    print("Molecule information")
    print("ChEMBL ID:", aspirin["molecule_chembl_id"])
    print("Preferred name:", aspirin["pref_name"])
    print("Molecular formula:", aspirin["molecule_properties"]["full_molformula"])
    print("Molecular weight:", aspirin["molecule_properties"]["full_mwt"])

    # 2. Retrieve bioactivity data for aspirin
    activity = new_client.activity

    activities = activity.filter(
        molecule_chembl_id=aspirin["molecule_chembl_id"]
    ).only(
        "target_chembl_id",
        "target_pref_name",
        "standard_type",
        "standard_value",
        "standard_units"
    )

    # Convert first 10 records to DataFrame
    df = pd.DataFrame(list(activities[:10]))

    print("\nBioactivity data")
    print(df)
    ```

### 4.2 Data preprocessing

#### Molecular standardization

A basic first step is to remove salts and counterions and return a canonical SMILES.
Note that `SaltRemover` only strips a predefined list of common salts: it does not
neutralize charges or normalize functional groups, so an isolated acetate anion or a
zwitterion will remain charged.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import SaltRemover

    def standardize_molecule(smiles):
        """Remove salts and return canonical SMILES."""

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        # Remove common salts or counterions
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        # Return canonical SMILES
        canonical_smiles = Chem.MolToSmiles(mol)

        return canonical_smiles


    # Example molecules
    smiles_list = [
        "CC(=O)[O-].[Na+]",  # Sodium acetate
        "CCO",               # Ethanol
        "[NH3+]CC[O-]"       # Zwitterionic form
    ]

    standardized = [
        standardize_molecule(smiles)
        for smiles in smiles_list
    ]

    print(standardized)
    ```

For a more complete and reproducible pipeline, RDKit's `rdMolStandardize` module
handles fragment selection, functional-group normalization, and charge
neutralization in a principled way. This is the recommended approach for preparing
molecules pulled from public databases.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    def standardize(smiles):
        """Normalize, keep the largest fragment, and neutralize charges."""

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        # 1. Normalize functional groups and charge representations
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        # 2. Keep only the largest fragment (removes salts/counterions)
        fragment_chooser = rdMolStandardize.LargestFragmentChooser()
        mol = fragment_chooser.choose(mol)

        # 3. Neutralize charges where chemically reasonable
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        return Chem.MolToSmiles(mol)


    smiles_list = [
        "CC(=O)[O-].[Na+]",  # Sodium acetate -> acetic acid
        "CCO",               # Ethanol
        "[NH3+]CC[O-]"       # Zwitterion -> neutral form
    ]

    print([standardize(smiles) for smiles in smiles_list])
    ```

#### Removing duplicates

Databases frequently contain the same compound under several different SMILES
strings. Hashing each molecule to its InChIKey provides a canonical identifier
that makes deduplication straightforward.

??? note "Example"

    ```python
    from rdkit import Chem

    def deduplicate(smiles_list):
        """Return unique molecules, identified by InChIKey."""

        seen = set()
        unique = []

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            inchikey = Chem.MolToInchiKey(mol)

            if inchikey not in seen:
                seen.add(inchikey)
                unique.append(smiles)

        return unique


    # "CCO", "OCC", and "C(O)C" all represent ethanol
    smiles_list = ["CCO", "OCC", "C(O)C", "c1ccccc1"]

    print(deduplicate(smiles_list))
    # -> ['CCO', 'c1ccccc1']
    ```

#### Train/Test Splitting

??? note "Example"

    ```python
    from rdkit import Chem
    from sklearn.model_selection import train_test_split

    # Random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaffold split for molecules
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict

    def scaffold_split(smiles_list, test_size=0.2):
        """Split molecules by Bemis-Murcko scaffold."""
        scaffolds = defaultdict(list)

        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")

            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds[scaffold].append(idx)

        # Sort scaffold groups from largest to smallest
        scaffold_sets = sorted(
            scaffolds.values(), key=len, reverse=True
        )

        n_total = len(smiles_list)
        n_test = int(n_total * test_size)

        train_idx, test_idx = [], []
        train_count = 0

        # Greedily fill the training set; overflow goes to the test set.
        # Whole scaffold groups are kept together, so no scaffold appears
        # in both splits.
        for scaffold_set in scaffold_sets:
            if train_count + len(scaffold_set) <= n_total - n_test:
                train_idx.extend(scaffold_set)
                train_count += len(scaffold_set)
            else:
                test_idx.extend(scaffold_set)

        return train_idx, test_idx
    ```

## 5. Practical Exercise: Solubility Prediction

### Complete Workflow

??? note "Example"

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
            'FractionCSP3': Descriptors.FractionCSP3(mol)
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
    #plt.show()
    plt.savefig('solubilities.png', dpi=300, bbox_inches='tight')
    plt.close()

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


## 6. Example with SELFIES

??? note "Example"

    ```python
    import numpy as np
    import selfies as sf
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen

    # Convert SELFIES to molecule
    selfies_str = sf.encoder("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin in SELFIES
    print(f"SELFIES representation: {selfies_str}")

    smiles = sf.decoder(selfies_str)
    mol = Chem.MolFromSmiles(smiles)

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

    ####

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
    frac_sp3 = Descriptors.FractionCSP3(mol)
    print(f"Fraction Csp3: {frac_sp3:.2f}")

    ####

    from rdkit.Chem import GraphDescriptors

    # Balaban J index (molecular branching)
    balaban = GraphDescriptors.BalabanJ(mol)

    # Bertz complexity index
    bertz = GraphDescriptors.BertzCT(mol)

    # Chi indices (connectivity)
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)

    ####

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

    ####

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

    ####

    from rdkit.Chem import QED

    qed_score = QED.qed(mol)
    print(f"QED Score: {qed_score:.3f}")
    # Range: [0, 1], higher is more drug-like
    # Based on 8 molecular properties

    ####

    from rdkit.Chem import RDConfig
    import sys
    sys.path.append(f'{RDConfig.RDContribDir}/SA_Score')
    import sascorer

    sa_score = sascorer.calculateScore(mol)
    print(f"SA Score: {sa_score:.2f}")
    # Range: [1, 10]
    # 1: Easy to synthesize
    # 10: Difficult to synthesize

    ####

    def calculate_molecular_descriptors(selfies_str):
        """
        Comprehensive descriptor calculation from SELFIES
        """
        # Convert SELFIES to SMILES then to molecule
        smiles = sf.decoder(selfies_str)
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
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            
            # Drug-likeness
            'QED': QED.qed(mol),
        }
        
        return descriptors

    # Example usage - Convert SMILES to SELFIES first
    smiles_list = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    selfies_list = [sf.encoder(s) for s in smiles_list]

    print("\nSELFIES representations:")
    for smiles, selfies_str in zip(smiles_list, selfies_list):
        print(f"SMILES:  {smiles}")
        print(f"SELFIES: {selfies_str}")
        print()

    import pandas as pd

    descriptor_list = [calculate_molecular_descriptors(s) for s in selfies_list]
    df_descriptors = pd.DataFrame(descriptor_list)
    df_descriptors['SELFIES'] = selfies_list
    df_descriptors['SMILES'] = smiles_list  # Optional: keep original SMILES for reference

    print(df_descriptors)

    ####

    import networkx as nx

    def mol_to_graph(selfies_str):
        """Convert SELFIES molecule to NetworkX graph"""
        smiles = sf.decoder(selfies_str)
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
    selfies_ethanol = sf.encoder("CCO")
    G = mol_to_graph(selfies_ethanol)
    print(f"\nGraph from SELFIES: {selfies_ethanol}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Node features: {G.nodes[0]}")


    ####

    def get_adjacency_matrix(selfies_str, max_atoms=50):
        """Get adjacency matrix with padding from SELFIES"""
        smiles = sf.decoder(selfies_str)
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

    selfies_ethanol = sf.encoder("CCO")
    adj, n_atoms = get_adjacency_matrix(selfies_ethanol)
    print(f"\nAdjacency matrix from SELFIES: {selfies_ethanol}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Actual atoms: {n_atoms}")
    ```

## 6. Practical Exercise: Complete ML Pipeline

### Task
Build a complete machine learning pipeline to predict molecular solubility. The dataset can be downloaded from:
J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005 (https://pubs.acs.org/doi/10.1021/ci034243x)

### Dataset

??? note "Example"

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

??? note "Example"

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

??? note "Example"

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

??? note "Example"

    ```python
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    ```

### Step 4: Preprocessing

??? note "Example"

    ```python
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

### Step 5: Model Training with Cross-Validation

??? note "Example"

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

??? note "Example"

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

??? note "Example"

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

??? note "Example"

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

??? note "Example"

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

## 7. Practical example: QM9

??? note "Example"

    ```python
    # Classical machine learning with QM9
    # Model: Random Forest Regressor
    # Target: HOMO energy

    import numpy as np
    import pandas as pd

    import deepchem as dc

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # 1. Load QM9 dataset

    tasks, datasets, transformers = dc.molnet.load_qm9(
        featurizer="ECFP",
        splitter="random"
    )

    train_dataset, valid_dataset, test_dataset = datasets

    print("QM9 tasks:")
    print(tasks)

    # 2. Select target property

    target_name = "homo"
    target_index = tasks.index(target_name)

    # DeepChem datasets store:
    # X = molecular features
    # y = target values

    X_train = train_dataset.X
    y_train = train_dataset.y[:, target_index]

    X_valid = valid_dataset.X
    y_valid = valid_dataset.y[:, target_index]

    X_test = test_dataset.X
    y_test = test_dataset.y[:, target_index]

    print("\nTraining shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print("Test shape:", X_test.shape)

    # 3. Train classical ML model

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 4. Validate model

    y_valid_pred = model.predict(X_valid)

    valid_mae = mean_absolute_error(y_valid, y_valid_pred)
    valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    valid_r2 = r2_score(y_valid, y_valid_pred)

    print("\nValidation performance")
    print(f"MAE:  {valid_mae:.4f}")
    print(f"RMSE: {valid_rmse:.4f}")
    print(f"R²:   {valid_r2:.4f}")

    # 5. Final test evaluation

    y_test_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTest performance")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²:   {test_r2:.4f}")

    # 6. Compare true vs predicted values

    results = pd.DataFrame({
        "true_homo": y_test[:10],
        "predicted_homo": y_test_pred[:10]
    })

    print("\nExample predictions:")
    print(results)
    ```

## 8. Key Takeaways

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


## 9. Resources and Further Reading

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


