# Foundations of Machine Learning for Molecular Systems

## Learning Objectives

* Develop an understanding of the key machine learning concepts used in molecular science
* Become familiar with different approaches for representing molecules and their advantages and limitations
* Build basic machine learning models for predicting molecular properties
* Learn how to work with chemical databases, molecular descriptors, and computational representations
* Recognize the main challenges associated with applying machine learning methods to chemical problems

## 1. Introduction to ML in Molecular Systems

### 1.1 Why Machine Learning for Molecules?

Chemical space is extraordinarily large, with estimates suggesting that approximately $10^{60}$ drug-like 
molecules could potentially exist (Med. Res. Rev., 16, 3-50, 1996). Exploring such an enormous number of possible 
compounds using conventional methods is impractical. Traditional drug discovery and materials research often 
rely on approaches such as:

* Synthesizing and experimentally evaluating compounds individually, which can be costly and time-consuming
* Performing quantum mechanical calculations for candidate molecules, which may require substantial computational resources
* Using iterative trial-and-error experiments, which can involve many unsuccessful candidates

Machine learning provides new approaches for exploring molecular systems more efficiently.

**Predict Molecular Properties Computationally**

* Estimate properties such as solubility, toxicity, and binding affinity before experimental testing
* Screen large collections of candidate molecules computationally prior to synthesis
* Reduce the number of expensive experiments or simulations required during early-stage discovery

**Identify Structure-Property Relationships**

* Determine which molecular characteristics are associated with particular properties
* Help reveal relationships between molecular structure and biological or chemical behavior
* Apply knowledge learned from one group of molecules to related chemical families

**Explore Chemical Space More Efficiently**

* Search extremely large molecular spaces using data-driven strategies
* Prioritize promising candidates for further computational or experimental investigation
* Identify new molecular scaffolds and chemical structures beyond previously studied compounds

**Accelerate Discovery Workflows**

* Conventional drug discovery can require 10–15 years and substantial financial investment to bring a new drug to market
* Machine learning can help reduce the time and resources required for tasks such as virtual screening, molecular optimization, and candidate prioritization
* Data-driven molecular design approaches have demonstrated the potential to generate and evaluate new candidate structures on considerably shorter timescales

### 1.2 Success Stories

**COVID-19 Drug Repurposing**

During the COVID-19 pandemic, machine learning and knowledge-based computational methods were used 
to rapidly evaluate existing drugs as potential treatments for SARS-CoV-2 infection. One notable 
example was Baricitinib, originally developed for rheumatoid arthritis, which was identified as a 
promising candidate for repurposing. It was later authorized for the treatment of COVID-19, 
illustrating how data-driven approaches can help prioritize existing compounds for new therapeutic 
applications.

**Antibiotic Discovery**

In 2020, researchers used a machine learning model trained on compounds with known antibacterial 
activity to identify Halicin as a promising antibiotic candidate. Halicin has a chemical structure 
that differs substantially from those of many conventional antibiotics and showed activity against 
several drug-resistant bacterial species. This study demonstrated how machine learning can search 
chemical space for candidates that may be difficult to identify using traditional screening strategies.

**Materials Science**

Machine learning has also become an important tool for accelerating materials discovery. In battery 
research, data-driven models can help prioritize promising electrolyte and electrode materials from 
large candidate spaces. ML methods can also provide rapid approximations of properties such as thermal 
conductivity that might otherwise require expensive simulations. Similar approaches are being applied 
to the discovery and optimization of photovoltaic, catalytic, and other functional materials.

### 1.3 Key Challenges in Molecular ML

#### High Dimensionality

Molecular systems can be described by large numbers of variables, including atomic identities, 
three-dimensional coordinates, conformational degrees of freedom, electronic information, and other 
physicochemical properties. Using all of this information directly can lead to complex and computationally 
demanding models. An important challenge is therefore to construct compact molecular representations 
that preserve the structural and chemical information required for accurate predictions.

#### Data Scarcity

Compared with fields such as computer vision, many molecular machine learning problems have relatively 
limited amounts of labeled data. Experimental measurements can be expensive, slow, or difficult to obtain, 
and available datasets may contain only thousands or tens of thousands of compounds. Approaches such as 
transfer learning, self-supervised learning, data augmentation, and semi-supervised learning can help 
make better use of limited labeled datasets.

#### Physical Constraints

Machine learning models for molecular systems should be consistent with important chemical and physical 
principles. Depending on the application, these may include energy conservation, atomic valence constraints, 
and symmetries associated with rotations, translations, and permutations of equivalent atoms. Incorporating 
these principles into model architectures can improve physical consistency and data efficiency. Equivariant 
neural networks and other physics-aware approaches are commonly used for this purpose.

#### Interpretability Requirements

In scientific applications, understanding why a model produces a particular prediction can be almost as 
important as the prediction itself. Identifying the molecular features or structural regions associated 
with a predicted property can support hypothesis generation, model validation, and experimental design. 
Methods such as feature importance analysis, attribution techniques, attention visualization, and other 
explainable machine learning approaches can help researchers examine model behavior.

#### Distribution Shift

Molecular models may perform well on compounds similar to those used during training but become less reliable 
when applied to unfamiliar regions of chemical space. New scaffolds, uncommon functional groups, different 
molecular sizes, or target values outside the training distribution can all lead to reduced predictive performance. 
Strategies such as uncertainty quantification, active learning, domain adaptation, and carefully designed 
dataset splits can help identify and reduce the effects of distribution shift.


## 2. Molecular Representations

The choice of molecular representation is crucial; it determines what information is available to
the model and how efficiently it can learn.

### 2.1 SMILES (Simplified Molecular Input Line Entry System)

SMILES is a text-based notation that represents molecular structure as a string (J. Chem. Inf. Comput. Sci. 1988, 28, 1, 31–36).

#### Basic SMILES Syntax

**Simple molecules**:

```text
Methane:    C
Ethanol:    CCO
Benzene:    c1ccccc1   (lowercase c indicates aromatic carbon)
Water:      O
````

**Branches**:

```text
Isobutane:    CC(C)C
                └─ atoms inside parentheses represent a branch
```

**Double and triple bonds**:

```text
Ethene:     C=C
Ethyne:     C#C
CO2:        O=C=O
```

**Rings**:

```text
Cyclohexane:    C1CCCCC1
                └─ matching ring numbers indicate a ring-closure bond

Naphthalene:    c1ccc2ccccc2c1
                └─ fused aromatic rings
```

**Stereochemistry**:

```text
Chiral alanine:    N[C@@H](C)C(=O)O
                     └─ @ and @@ specify tetrahedral chirality
```

The symbols `@` and `@@` describe the orientation of atoms around a tetrahedral
stereocenter relative to their order in the SMILES string. They should not be
interpreted directly as `R` and `S`; the absolute configuration must be determined
from the molecular connectivity and CIP priority rules.

Note that in:

```text
c1ccccc1
```

lowercase `c` means **aromatic carbon**. Other aromatic atoms can also appear in 
lowercase, such as `n` for aromatic nitrogen.

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

- **No 3D information**: SMILES encodes connectivity but not geometry. Stereoisomers that differ 
only in 3D arrangement require explicit stereochemistry notation (@ / @@), and even then the 
string carries no atomic coordinates.

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

Molecular fingerprints are numerical representations that encode structural information 
about a molecule, typically as fixed-length binary or count vectors. They are widely used 
in cheminformatics because they provide compact representations that can be efficiently 
compared or used as input features for machine learning models.

#### Morgan Fingerprints and ECFP

Morgan fingerprints are circular molecular fingerprints based on the Morgan algorithm. 
Extended-Connectivity Fingerprints (ECFPs) are a widely used implementation of this general 
approach. They represent a molecule by describing the local chemical environments surrounding 
its atoms at progressively larger distances.

The procedure begins by assigning each atom an initial identifier derived from properties such 
as atomic number, bonding, charge, and connectivity. During successive iterations, information 
from neighboring atoms is incorporated into each identifier. Each iteration therefore describes 
a larger circular environment around the central atom.

The resulting identifiers represent molecular substructures. In practical implementations, these 
identifiers are commonly hashed or folded into a fixed-length fingerprint vector. A binary 
fingerprint records whether particular environments are present, whereas a count fingerprint 
can record how many times they occur.

Morgan fingerprints are commonly used for molecular similarity searches, clustering, virtual 
screening, classification, regression, and other machine learning applications.

**Algorithm:**

1. Assign an initial identifier to each atom using selected atomic and bonding properties.
2. Examine the neighborhood surrounding each atom within a specified radius.
3. Iteratively update the identifiers by incorporating information from neighboring atoms and bonds.
4. Convert the resulting molecular-environment identifiers into a fixed-length binary or count fingerprint, typically using hashing.
5. Use the fingerprint as a molecular representation for similarity calculations or machine learning models.


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

**Bit and Count Fingerprints**

A bit fingerprint records whether particular molecular environments are present, whereas 
a count fingerprint retains information about how many times those environments occur. 
Bit fingerprints are often convenient for similarity calculations and introductory examples, 
while count fingerprints can preserve additional frequency information that may be useful for machine learning.

**Important parameters:**

* **Radius:** controls the size of the local atomic environments included in the fingerprint

  * **Radius 1 (ECFP2):** captures environments extending approximately one bond from each atom
  * **Radius 2 (ECFP4):** captures larger local environments and is a commonly used choice
  * **Radius 3 (ECFP6):** represents still broader molecular neighborhoods

* **`fpSize`:** determines the length of the hashed fingerprint vector

  * **1024 bits:** compact and computationally efficient, but more susceptible to hash collisions
  * **2048 bits:** commonly used and provides a good balance between compactness and collision frequency
  * **4096 bits:** provides more available positions and can reduce collisions, at the cost of increased memory and computational requirements

#### MACCS Keys

MACCS keys are predefined structural fingerprints that represent molecules using a fixed collection 
of 166 chemical patterns. Each key indicates whether a particular substructure, functional group, 
or bonding motif is present in the molecule. Because the features are predefined and chemically 
interpretable, MACCS keys are commonly used for molecular similarity searches, clustering, and other 
cheminformatics tasks. In RDKit, the fingerprint has 167 positions because bit 0 is unused, 
while the 166 defined MACCS keys occupy positions 1 through 166.

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

RDKit fingerprints are topological molecular representations that capture structural information 
by examining paths of connected atoms and bonds within a molecule. They are frequently used in 
cheminformatics for tasks such as molecular similarity comparison, clustering, virtual screening, 
and machine learning based on molecular structure.

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

**Atom pair fingerprints** describe molecules using pairs of atoms together with the 
topological distance between them, measured as the number of bonds along the shortest path. 
The atoms are characterized using properties such as atomic number, connectivity, and 
bonding environment.

**Topological torsion fingerprints** represent sequences of connected atoms along molecular 
paths. By default, RDKit uses paths containing four atoms. Despite the name, these 
fingerprints describe molecular connectivity rather than three-dimensional torsion angles.

Both representations can be generated as count fingerprints or fixed-length bit vectors and 
are useful for molecular similarity analysis and machine learning.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    # Create a molecule
    mol = Chem.MolFromSmiles("CCCCO")  # 1-butanol

    # Create fingerprint generators
    atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=2048
    )

    torsion_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=2048
    )

    # Generate bit fingerprints
    atom_pairs = atom_pair_generator.GetFingerprint(mol)
    torsions = torsion_generator.GetFingerprint(mol)

    # Display fingerprint information
    print("Atom pair fingerprint length:", len(atom_pairs))
    print("Topological torsion fingerprint length:", len(torsions))

    print("Atom pair bits set:", atom_pairs.GetNumOnBits())
    print("Topological torsion bits set:", torsions.GetNumOnBits())
    ```

#### Comparing Molecules with Fingerprints

Molecular fingerprints can be compared using similarity coefficients that measure the 
overlap between the structural features encoded in two fingerprint vectors. Molecules 
sharing many fingerprint features generally receive higher similarity scores, while 
structurally different molecules tend to produce lower values. The resulting similarity 
depends on both the fingerprint representation and the metric used for comparison.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs

    # 1. Create molecules
    mol1 = Chem.MolFromSmiles("CCO")        # Ethanol
    mol2 = Chem.MolFromSmiles("CCCO")       # 1-Propanol
    mol3 = Chem.MolFromSmiles("c1ccccc1")   # Benzene

    # 2. Create a Morgan fingerprint generator
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    # 3. Generate Morgan bit fingerprints
    fp1 = morgan_generator.GetFingerprint(mol1)
    fp2 = morgan_generator.GetFingerprint(mol2)
    fp3 = morgan_generator.GetFingerprint(mol3)

    # 4. Calculate Tanimoto similarities
    sim_12 = DataStructs.TanimotoSimilarity(fp1, fp2)
    sim_13 = DataStructs.TanimotoSimilarity(fp1, fp3)

    print(
        f"Tanimoto similarity "
        f"(ethanol, 1-propanol): {sim_12:.3f}"
    )

    print(
        f"Tanimoto similarity "
        f"(ethanol, benzene):    {sim_13:.3f}"
    )

    # 5. Compare additional similarity metrics
    dice = DataStructs.DiceSimilarity(fp1, fp2)
    cosine = DataStructs.CosineSimilarity(fp1, fp2)

    print(f"\nDice similarity:   {dice:.3f}")
    print(f"Cosine similarity: {cosine:.3f}")
    ```

The **Tanimoto coefficient** is one of the most commonly used similarity measures 
for binary molecular fingerprints. For two fingerprints $A$ and $B$, it can be written as

$$
T(A,B)
=
\frac{|A \cap B|}
{|A| + |B| - |A \cap B|},
$$

where $|A|$ and $|B|$ represent the numbers of active bits and $|A \cap B|$ represents the 
number of bits shared by both fingerprints.

Similarity values typically range from $0$ to $1$, where values closer to $1$ indicate greater 
fingerprint similarity. However, these scores should be interpreted relative to the fingerprint type, 
its parameters, and the similarity metric rather than as an absolute measure of chemical similarity.

### 2.4 3D Molecular Representations

While fingerprints and connectivity-based molecular graphs primarily describe molecular topology, 
many chemical and physical properties also depend on the three-dimensional arrangement of atoms. 
**3D molecular representations** incorporate geometric information such as atomic coordinates, 
interatomic distances, bond angles, and molecular conformations.

These representations are particularly important for applications in molecular dynamics, molecular 
docking, quantum chemistry, protein-ligand modeling, interatomic potentials, and materials science.

Common types of 3D molecular representations include:

* Cartesian atomic coordinates
* Interatomic distance matrices
* Coulomb matrices
* Local atomic environments
* Molecular conformations or conformer ensembles

Cartesian coordinates provide a direct description of atomic positions in three-dimensional space, 
although the numerical coordinates change when the molecule is translated or rotated. Other representations, 
such as interatomic distances, can provide geometric information that is naturally invariant to 
global rotations and translations.

#### Example: Generating 3D Coordinates with RDKit

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 1. Create a molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")  # Ethanol

    # Add explicit hydrogen atoms
    mol = Chem.AddHs(mol)

    # 2. Generate a 3D conformation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    conf_id = AllChem.EmbedMolecule(
        mol,
        params
    )

    if conf_id == -1:
        raise RuntimeError(
            "3D conformer generation failed."
        )

    # 3. Optimize the molecular geometry
    status = AllChem.UFFOptimizeMolecule(
        mol,
        confId=conf_id
    )

    if status == 0:
        print("UFF optimization converged.")
    else:
        print("UFF optimization did not fully converge.")

    # 4. Extract atomic coordinates
    conf = mol.GetConformer(conf_id)

    print("\nAtomic coordinates:\n")

    for atom in mol.GetAtoms():

        atom_index = atom.GetIdx()
        pos = conf.GetAtomPosition(atom_index)

        print(
            f"Atom {atom_index:2d} "
            f"{atom.GetSymbol():2s} -> "
            f"x={pos.x:8.3f}, "
            f"y={pos.y:8.3f}, "
            f"z={pos.z:8.3f}"
        )
    ```

The resulting coordinates describe one possible three-dimensional conformation of 
ethanol. For flexible molecules, several conformations may be energetically accessible, 
so a single generated structure does not necessarily represent the complete conformational 
behavior of the molecule.

### 2.5 Protein Representations

Proteins are complex biological macromolecules that can be represented at different levels 
of detail depending on the machine learning task. A protein may be described by its amino 
acid sequence, residue-level features, three-dimensional structure, molecular graph, learned 
embedding, or complete set of atomic coordinates.

The choice of representation strongly influences what information is available to the model. 
Sequence-based representations are useful for tasks such as function prediction, whereas 
three-dimensional and atomistic representations are particularly important for structural 
modeling, molecular dynamics, protein-ligand interactions, and geometry-based machine learning.

Common protein representations include:

* Amino acid sequences
* One-hot encoded residue sequences
* Protein language model embeddings
* Residue contact or distance maps
* Graph-based representations
* Three-dimensional atomic coordinates

#### Example: Loading an Atomistic Protein Structure with ASE

??? note "Example"

    ```python
    from urllib.request import urlretrieve
    from ase.io import read

    # Get the 3MUF structure
    pdb_id = "3MUF"

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    filename = f"{pdb_id}.pdb"

    # Download structure
    urlretrieve(url, filename)

    # Read with ASE
    protein = read(
        filename,
        format="proteindatabank"
    )

    print("Number of atoms:", len(protein))

    for index, atom in enumerate(protein[:5]):

        x, y, z = atom.position

        print(
            f"Atom {index:2d} "
            f"{atom.symbol:2s} -> "
            f"x={x:8.3f} "
            f"y={y:8.3f} "
            f"z={z:8.3f}"
        )
    ```

This example demonstrates how a protein structure stored in PDB format can be converted into 
an atomistic representation using the [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase). 
Each atom is described by its chemical element and Cartesian coordinates,

$$
\mathbf{r}_i =
\begin{bmatrix}
x_i \
y_i \
z_i
\end{bmatrix},
$$

where $\mathbf{r}_i$ represents the position of atom $i$ in three-dimensional space.

Such coordinate-based representations can serve as the starting point for calculating interatomic 
distances, constructing molecular graphs, performing atomistic simulations, or preparing geometric 
features for machine learning models.

**Coulomb matrix representation**

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

Molecular descriptors are numerical quantities that summarize different chemical, structural, 
topological, or geometric characteristics of a molecule. They provide fixed numerical features 
that can be used in statistical analysis, similarity studies, and machine learning models.

Descriptor classifications are not always strict, and some descriptors can reasonably belong to 
more than one category. A practical classification includes physicochemical, structural, topological, 
and three-dimensional descriptors.

#### 1. Physicochemical Descriptors

Physicochemical descriptors describe properties related to molecular size, polarity, lipophilicity, 
and intermolecular interactions. Some are calculated directly from molecular composition, whereas 
others are empirical estimates derived from molecular structure.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen

    # 1. Create a molecule: aspirin
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Average molecular weight
    mw = Descriptors.MolWt(mol)

    print(
        f"Molecular Weight: {mw:.2f} g/mol"
    )

    # 3. Estimated lipophilicity
    # Wildman-Crippen estimate of LogP
    logp = Crippen.MolLogP(mol)

    print(
        f"LogP: {logp:.2f}"
    )

    # 4. Topological polar surface area
    tpsa = Descriptors.TPSA(mol)

    print(
        f"TPSA: {tpsa:.2f} Å²"
    )

    # 5. Molar refractivity
    # Wildman-Crippen estimate
    mr = Crippen.MolMR(mol)

    print(
        f"Molar Refractivity: {mr:.2f}"
    )
    ```

`MolWt` provides the average molecular weight, while `MolLogP` estimates the octanol/water 
partition coefficient and is commonly used as a measure of lipophilicity. `TPSA` estimates the 
molecular polar surface area from topological information and is frequently used when studying 
properties such as permeability. `MolMR` provides an estimate of molar refractivity.

#### 2. Structural Descriptors

Structural descriptors summarize characteristics such as hydrogen-bonding capacity, molecular 
flexibility, ring composition, and carbon hybridization. These descriptors are calculated 
primarily from the molecular connectivity rather than from an explicit three-dimensional conformation.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # 1. Create a molecule: aspirin
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Hydrogen-bond donors and acceptors
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    print(
        f"H-Bond Donors: {h_donors}"
    )

    print(
        f"H-Bond Acceptors: {h_acceptors}"
    )

    # 3. Rotatable bonds
    rot_bonds = Descriptors.NumRotatableBonds(
        mol
    )

    print(
        f"Rotatable Bonds: {rot_bonds}"
    )

    # 4. Ring information
    num_rings = Descriptors.RingCount(mol)
    aromatic_rings = Descriptors.NumAromaticRings(
        mol
    )

    print(
        f"Total Rings: {num_rings}"
    )

    print(
        f"Aromatic Rings: {aromatic_rings}"
    )

    # 5. Fraction of sp3-hybridized carbon atoms
    frac_sp3 = Descriptors.FractionCSP3(mol)

    print(
        f"Fraction Csp3: {frac_sp3:.2f}"
    )
    ```

The number of hydrogen-bond donors and acceptors provides information about 
potential intermolecular interactions. Rotatable-bond counts are commonly used 
as a simple measure of molecular flexibility, while ring counts describe cyclic 
structure. `FractionCSP3` represents the fraction of carbon atoms that are $sp^3$ hybridized.


#### 3. Topological Descriptors

Topological descriptors characterize the connectivity and organization of the molecular 
graph without requiring three-dimensional coordinates. They can describe branching, molecular 
complexity, connectivity patterns, and graph structure.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import GraphDescriptors

    # 1. Create a molecule: aspirin
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Balaban J index
    balaban = GraphDescriptors.BalabanJ(
        mol
    )

    print(
        f"Balaban J Index: {balaban:.3f}"
    )

    # 3. Bertz complexity index
    bertz = GraphDescriptors.BertzCT(
        mol
    )

    print(
        f"Bertz Complexity Index: {bertz:.3f}"
    )

    # 4. Chi connectivity indices
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)

    print(
        f"Chi0 Index: {chi0:.3f}"
    )

    print(
        f"Chi1 Index: {chi1:.3f}"
    )
    ```

The **Balaban J index** is a graph-based connectivity descriptor derived from molecular 
distances and connectivity. The **Bertz complexity index** provides a measure of 
molecular structural complexity. **Chi connectivity indices** describe aspects of 
molecular connectivity using atom degrees and graph paths.

#### 4. 3D Descriptors

Three-dimensional descriptors characterize molecular shape and the spatial distribution of 
atoms using an explicit molecular conformation. Unlike purely topological descriptors, their 
values depend on the particular three-dimensional geometry used in the calculation.

For flexible molecules, different conformations may therefore produce different 3D descriptor values.

??? note "Example"

    ```python
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D


    # 1. Create a molecule: aspirin
    mol = Chem.MolFromSmiles(
        "CC(=O)Oc1ccccc1C(=O)O"
    )

    # 2. Add explicit hydrogens
    mol_3d = Chem.AddHs(mol)

    # 3. Generate a 3D conformation
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    conf_id = AllChem.EmbedMolecule(
        mol_3d,
        params
    )

    if conf_id == -1:
        raise RuntimeError(
            "3D conformer generation failed."
        )

    # 4. Optimize the geometry using MMFF94
    if not AllChem.MMFFHasAllMoleculeParams(
        mol_3d
    ):
        raise RuntimeError(
            "MMFF parameters are not available "
            "for all atoms."
        )

    status = AllChem.MMFFOptimizeMolecule(
        mol_3d,
        confId=conf_id
    )

    if status == 0:
        print("MMFF optimization converged.")
    elif status == 1:
        print(
            "MMFF optimization did not fully converge."
        )
    else:
        print(
            "MMFF force field could not be initialized."
        )

    # 5. Calculate 3D descriptors
    asphericity = Descriptors3D.Asphericity(
        mol_3d,
        confId=conf_id
    )

    eccentricity = Descriptors3D.Eccentricity(
        mol_3d,
        confId=conf_id
    )

    inertial_shape = (
        Descriptors3D.InertialShapeFactor(
            mol_3d,
            confId=conf_id
        )
    )

    radius_of_gyration = (
        Descriptors3D.RadiusOfGyration(
            mol_3d,
            confId=conf_id
        )
    )

    # 6. Display results
    print(
        f"Asphericity: "
        f"{asphericity:.3f}"
    )

    print(
        f"Eccentricity: "
        f"{eccentricity:.3f}"
    )

    print(
        f"Inertial Shape Factor: "
        f"{inertial_shape:.3f}"
    )

    print(
        f"Radius of Gyration: "
        f"{radius_of_gyration:.3f} Å"
    )
    ```

**Asphericity** describes how strongly a molecular shape deviates from spherical symmetry. 
**Eccentricity** describes molecular elongation based on the principal moments of inertia. 
The **inertial shape factor** is another descriptor derived from those principal moments, 
while the **radius of gyration** measures the overall spatial extent of the molecular structure.

Because these descriptors depend on atomic coordinates, conformer generation and geometry 
optimization should be performed before they are calculated.

#### Drug-Likeness Metrics

Drug-likeness metrics provide simple quantitative measures of whether a molecule has physicochemical 
and structural characteristics commonly observed among drug-like compounds. These metrics are 
useful for prioritizing molecules during early-stage screening, but they do not directly predict 
biological activity, safety, efficacy, or clinical success.

#### Lipinski's Rule of Five

**Lipinski's Rule of Five** is a widely used heuristic for evaluating whether a small 
molecule has physicochemical properties compatible with oral absorption and permeability.

The conventional criteria are:

* Molecular weight

$$
MW \leq 500\ \mathrm{Da}
$$

* Lipophilicity

$$
\mathrm{cLogP} \leq 5
$$

* Number of hydrogen-bond donors

$$
HBD \leq 5
$$

* Number of hydrogen-bond acceptors

$$
HBA \leq 10
$$

A molecule violating more than one of these criteria may have an increased likelihood 
of poor absorption or permeability. However, the Rule of Five is a guideline rather than 
a strict requirement, and many successful drugs fall outside these limits.

#### QED (Quantitative Estimate of Drug-Likeness)

The **Quantitative Estimate of Drug-Likeness (QED)** combines several molecular 
properties into a continuous score describing how closely a compound resembles the 
physicochemical characteristics commonly observed in drug-like molecules.

The QED calculation includes information such as:

* Molecular weight
* Lipophilicity
* Hydrogen-bond donors
* Hydrogen-bond acceptors
* Polar surface area
* Rotatable bonds
* Aromatic ring count
* Structural alerts

The resulting QED score ranges approximately from

$$
0 \leq \mathrm{QED} \leq 1,
$$

where larger values indicate a more favorable combination of drug-like molecular properties. 
QED should be interpreted as a relative drug-likeness measure rather than as a probability 
that a molecule will become a successful drug.

#### Creating Molecular Feature Vectors

Multiple molecular descriptors can be combined into a numerical feature vector f
or use in machine learning. For a molecule $i$, a descriptor vector may be written as

$$
\mathbf{x}_i
=
[
MW,
\mathrm{LogP},
TPSA,
HBD,
HBA,
\ldots
].
$$

Each molecule is therefore represented by a fixed set of numerical properties that 
can be assembled into a feature matrix for statistical analysis or machine learning.

??? note "Example"

    ```python
    import pandas as pd

    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors,
        Crippen,
        GraphDescriptors,
        QED
    )

    def calculate_molecular_descriptors(smiles):
        """
        Calculate a collection of molecular descriptors
        and simple drug-likeness metrics.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES: {smiles}"
            )

        # Physicochemical properties
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)

        tpsa = Descriptors.TPSA(mol)
        mol_mr = Crippen.MolMR(mol)

        # Hydrogen bonding
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)

        # Lipinski Rule-of-Five violations
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            h_donors > 5,
            h_acceptors > 10
        ])

        lipinski_pass = (
            lipinski_violations <= 1
        )

        # Construct descriptor dictionary
        descriptors = {

            # Physicochemical descriptors
            "MW": mw,
            "LogP": logp,
            "TPSA": tpsa,
            "MolMR": mol_mr,

            # Structural descriptors
            "NumHDonors": h_donors,
            "NumHAcceptors": h_acceptors,
            "NumRotatableBonds":
                Descriptors.NumRotatableBonds(mol),

            "NumHeteroatoms":
                Descriptors.NumHeteroatoms(mol),

            "NumAromaticRings":
                Descriptors.NumAromaticRings(mol),

            "NumSaturatedRings":
                Descriptors.NumSaturatedRings(mol),

            "NumAliphaticRings":
                Descriptors.NumAliphaticRings(mol),

            "RingCount":
                Descriptors.RingCount(mol),

            # Molecular complexity
            "BertzCT":
                GraphDescriptors.BertzCT(mol),

            "NumBridgeheadAtoms":
                Descriptors.NumBridgeheadAtoms(mol),

            "NumSpiroAtoms":
                Descriptors.NumSpiroAtoms(mol),

            # Surface-area-related descriptors
            "LabuteASA":
                Descriptors.LabuteASA(mol),

            # Partial-charge-weighted VSA bin
            "PEOE_VSA1":
                Descriptors.PEOE_VSA1(mol),

            # Atom counts
            "NumCarbon": sum(
                atom.GetAtomicNum() == 6
                for atom in mol.GetAtoms()
            ),

            "NumNitrogen": sum(
                atom.GetAtomicNum() == 7
                for atom in mol.GetAtoms()
            ),

            "NumOxygen": sum(
                atom.GetAtomicNum() == 8
                for atom in mol.GetAtoms()
            ),

            "NumHalogens": sum(
                atom.GetAtomicNum()
                in {9, 17, 35, 53}
                for atom in mol.GetAtoms()
            ),

            # Carbon saturation
            "FractionCsp3":
                Descriptors.FractionCSP3(mol),

            # Drug-likeness metrics
            "LipinskiViolations":
                lipinski_violations,

            "LipinskiPass":
                lipinski_pass,

            "QED":
                QED.qed(mol)
        }

        return descriptors

    # Example molecules
    molecules = {
        "Ethanol":
            "CCO",

        "Aspirin":
            "CC(=O)Oc1ccccc1C(=O)O",

        "Caffeine":
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }

    # Calculate descriptors
    rows = []

    for name, smiles in molecules.items():

        row = {
            "Molecule": name,
            "SMILES": smiles
        }

        row.update(
            calculate_molecular_descriptors(
                smiles
            )
        )

        rows.append(row)

    # Create DataFrame
    df_descriptors = pd.DataFrame(rows)

    print(df_descriptors)

    # Display selected drug-likeness properties
    print(
        "\nDrug-likeness summary:\n"
    )

    print(
        df_descriptors[
            [
                "Molecule",
                "MW",
                "LogP",
                "NumHDonors",
                "NumHAcceptors",
                "LipinskiViolations",
                "LipinskiPass",
                "QED"
            ]
        ]
    )
    ```

In this example, the molecular descriptors form numerical features that can later be supplied to a machine 
learning model. The `LipinskiViolations` variable counts how many of the four Rule-of-Five thresholds 
are exceeded, while `QED` provides a continuous measure of overall drug-likeness.

A molecule can satisfy Lipinski's Rule of Five and still be unsuitable as a drug. These metrics describe 
selected physicochemical characteristics and should therefore be considered screening tools rather than 
definitive measures of pharmaceutical suitability.

### 2.7 Graph Representations

Graph representations describe molecular systems as collections of **nodes** connected by **edges**. 
For small molecules, nodes usually correspond to atoms and edges to chemical bonds. For proteins, 
graphs can instead be constructed at the residue level, where each node represents an amino acid and 
edges describe sequence connectivity, spatial proximity, or other interactions.

Graphs are particularly useful for molecular machine learning because they preserve connectivity 
while allowing chemical and structural information to be associated with individual nodes and edges.

A graph can be written as

$$
G=(V,E),
$$

where (V) is the set of nodes and (E) is the set of edges.

#### Graph Structure

For a molecular graph, each atom becomes a node and each chemical bond becomes an edge. 
Additional chemical information can be stored as node and edge attributes.

??? note "Example"

    ```python
    from rdkit import Chem
    import networkx as nx


    def mol_to_graph(smiles):
        """
        Convert a molecule from SMILES
        into a NetworkX graph.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES: {smiles}"
            )

        G = nx.Graph()

        # Add atoms as nodes
        for atom in mol.GetAtoms():

            G.add_node(
                atom.GetIdx(),

                atomic_num=atom.GetAtomicNum(),
                symbol=atom.GetSymbol(),

                # Number of explicit neighboring atoms
                degree=atom.GetDegree(),

                formal_charge=atom.GetFormalCharge(),

                # Explicit + implicit hydrogen count
                num_h=atom.GetTotalNumHs(),

                hybridization=str(
                    atom.GetHybridization()
                ),

                is_aromatic=atom.GetIsAromatic()
            )

        # Add chemical bonds as edges
        for bond in mol.GetBonds():

            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),

                bond_type=str(
                    bond.GetBondType()
                ),

                is_conjugated=
                    bond.GetIsConjugated(),

                is_aromatic=
                    bond.GetIsAromatic(),

                is_in_ring=
                    bond.IsInRing()
            )

        return G

    # Example: ethanol
    G = mol_to_graph("CCO")

    print(
        "Number of nodes:",
        G.number_of_nodes()
    )

    print(
        "Number of edges:",
        G.number_of_edges()
    )

    print(
        "\nNode 0 features:"
    )

    print(
        G.nodes[0]
    )

    print(
        "\nEdges:"
    )

    for u, v, features in G.edges(data=True):
        print(
            u,
            "--",
            v,
            features
        )
    ```

    For ethanol, the graph contains three heavy-atom nodes corresponding to two carbon atoms and one 
    oxygen atom. Because hydrogens are implicit in the SMILES representation, they are not separate 
    graph nodes in this example.

#### Adjacency Matrix Representation

The connectivity of a graph can also be represented using an **adjacency matrix** (A). For an 
unweighted molecular graph,

$$
A_{ij}
=
\begin{cases}
1, & \text{if atoms } i \text{ and } j \text{ are bonded},\\
0, & \text{otherwise}.
\end{cases}
$$

For an undirected molecular graph,

$$
A=A^\mathrm{T}.
$$

A binary adjacency matrix describes connectivity only. Chemical information such as bond order 
should therefore be stored separately as edge features or explicitly incorporated into a weighted 
adjacency matrix.

??? note "Example"

    ```python
    import numpy as np

    from rdkit import Chem
    from rdkit.Chem import rdmolops


    def get_adjacency_matrix(
        smiles,
        max_atoms=50
    ):
        """
        Generate a padded binary molecular
        adjacency matrix.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES: {smiles}"
            )

        num_atoms = mol.GetNumAtoms()

        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule contains "
                f"{num_atoms} atoms, "
                f"but max_atoms={max_atoms}"
            )

        # Generate the molecular adjacency matrix
        # useBO=False:
        #   bonded     -> 1
        #   nonbonded  -> 0
        adjacency = rdmolops.GetAdjacencyMatrix(
            mol,
            useBO=False
        )

        # Pad to a fixed matrix size
        padded_adjacency = np.zeros(
            (max_atoms, max_atoms),
            dtype=float
        )

        padded_adjacency[
            :num_atoms,
            :num_atoms
        ] = adjacency

        return padded_adjacency, num_atoms

    # Example: ethanol
    adjacency, n_atoms = (
        get_adjacency_matrix("CCO")
    )

    print(
        "Padded adjacency matrix shape:",
        adjacency.shape
    )

    print(
        "Actual number of atoms:",
        n_atoms
    )

    print(
        "\nMolecular adjacency matrix:"
    )

    print(
        adjacency[
            :n_atoms,
            :n_atoms
        ]
    )
    ```

    For ethanol, the unpadded connectivity matrix is

    $$
    A=
    \begin{bmatrix}
    0 & 1 & 0\\
    1 & 0 & 1\\
    0 & 1 & 0
    \end{bmatrix}.
    $$

    The first carbon is connected to the second carbon, which is connected to oxygen.

    Self-connections are not chemical bonds and are therefore absent from this matrix. Some 
    graph neural network architectures add self-loops later as part of the message-passing procedure.

#### Node and Edge Features

Graph topology alone does not contain all of the chemical information required for molecular 
machine learning. Each node and edge can therefore be associated with a numerical feature vector.

For atom $i$, the node feature vector may be written as

$$
\mathbf{x}_i
=
[
x_{i1},
x_{i2},
\ldots,
x_{iF}
],
$$

while a bond between atoms $i$ and $j$ can have an edge feature vector

$$
\mathbf{e}_{ij}
=
[
e_{ij1},
e_{ij2},
\ldots,
e_{ijK}
].
$$

Categorical properties such as hybridization and bond type are generally better 
represented using one-hot encodings or learned embeddings rather than arbitrary 
integer labels, because integer labels can incorrectly imply an ordinal relationship 
between categories.

??? note "Example"

    ```python
    import numpy as np
    from rdkit import Chem

    def get_node_features(atom):
        """
        Create a simple numerical feature
        vector for one atom.
        """

        # One-hot hybridization
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]

        hybridization = [
            int(
                atom.GetHybridization() == h
            )
            for h in hybridization_types
        ]

        hybridization_other = int(
            atom.GetHybridization()
            not in hybridization_types
        )

        # Construct atom feature vector
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),

            *hybridization,
            hybridization_other
        ]

        return np.array(
            features,
            dtype=float
        )

    def get_edge_features(bond):
        """
        Create a simple numerical feature
        vector for one chemical bond.
        """

        # One-hot bond type
        bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]

        bond_type_features = [
            int(
                bond.GetBondType() == bond_type
            )
            for bond_type in bond_types
        ]

        # Construct bond feature vector
        features = [
            *bond_type_features,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]

        return np.array(
            features,
            dtype=float
        )

    # Example: ethanol
    mol = Chem.MolFromSmiles("CCO")

    print("Node features:\n")

    for atom in mol.GetAtoms():

        print(
            f"Atom {atom.GetIdx()} "
            f"({atom.GetSymbol()}):",
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

    This is a simple example that is useful for teaching but in production molecular GNNs often 
    include additional node features such as chirality, valence, atomic mass, and charge, as 
    well as edge features such as stereochemistry and bond order.

#### Protein Graph Representations

Proteins can also be represented as graphs. One common residue-level representation assigns each 
amino acid residue to a node and uses the position of its C$^\alpha$ atom as the representative 
three-dimensional coordinate.

Two different types of relationships are especially useful:

* **Sequence edges** connect residues that are adjacent in the protein sequence.
* **Spatial-contact edges** connect residues whose C$$\alpha$ atoms lie within a chosen 
three-dimensional cutoff.

For two residues $i$ and $j$, their C$^\alpha$ distance is

$$
d_{ij}
=
\left|
\mathbf{r}_i
-
\mathbf{r}_j
\right|.
$$

A simple contact criterion is

$$
(i,j)\in E
\quad\text{if}\quad
d_{ij}\leq r_{\mathrm{cut}}.
$$

A cutoff around 8 Å is often used for illustrative C$^\alpha$ contact graphs, although 
the appropriate definition depends on the scientific problem.

The following example constructs a residue-level graph directly from the C$^\alpha$ atoms of 
chain A in the **3MUF** PDB structure.

??? note "Example"

    ```python
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    # 1. Load the PDB structure
    pdb_file = "3MUF.pdb"

    parser = PDBParser(
        QUIET=True
    )

    structure = parser.get_structure(
        "3MUF",
        pdb_file
    )

    # 2. Select model 0 and protein chain A
    model = structure[0]
    chain = model["A"]

    # 3. Extract amino-acid residues containing C-alpha
    residues = []

    for residue in chain:

        # Ignore water, ligands, and other
        # non-amino-acid residues
        if not is_aa(
            residue,
            standard=True
        ):
            continue

        if "CA" not in residue:
            continue

        ca_atom = residue["CA"]

        residues.append({
            "residue_name":
                residue.get_resname(),

            "residue_number":
                residue.id[1],

            "insertion_code":
                residue.id[2].strip(),

            "ca_coord":
                ca_atom.get_coord().astype(float)
        })


    print(
        "Number of C-alpha residues:",
        len(residues)
    )

    # 4. Create the residue graph
    G = nx.Graph()

    # Add one node per residue
    for i, residue in enumerate(residues):

        G.add_node(
            i,

            residue_name=
                residue["residue_name"],

            residue_number=
                residue["residue_number"],

            insertion_code=
                residue["insertion_code"],

            ca_coord=
                residue["ca_coord"]
        )

    # 5. Add sequence-neighbor edges
    for i in range(
        len(residues) - 1
    ):

        coord_i = residues[i][
            "ca_coord"
        ]

        coord_j = residues[i + 1][
            "ca_coord"
        ]

        distance = np.linalg.norm(
            coord_i - coord_j
        )

        G.add_edge(
            i,
            i + 1,

            distance=distance,
            sequence_neighbor=True,
            spatial_contact=False
        )

    # 6. Add spatial-contact edges
    distance_cutoff = 8.0

    for i in range(len(residues)):

        for j in range(
            i + 1,
            len(residues)
        ):

            coord_i = residues[i][
                "ca_coord"
            ]

            coord_j = residues[j][
                "ca_coord"
            ]

            distance = np.linalg.norm(
                coord_i - coord_j
            )

            if distance <= distance_cutoff:

                # Edge may already exist because
                # residues are sequence neighbors.
                if G.has_edge(i, j):

                    G.edges[i, j][
                        "spatial_contact"
                    ] = True

                    G.edges[i, j][
                        "distance"
                    ] = distance

                else:

                    G.add_edge(
                        i,
                        j,

                        distance=distance,
                        sequence_neighbor=False,
                        spatial_contact=True
                    )

    # 7. Display graph information
    print("Number of nodes:",G.number_of_nodes())

    print("Number of edges:",G.number_of_edges())

    print("\nFirst five residues:\n")

    for node in list(G.nodes)[:5]:

        data = G.nodes[node]

        print(
            f"Node {node:3d}: "
            f"{data['residue_name']} "
            f"{data['residue_number']} "
            f"CA = {data['ca_coord']}"
        )

    print("\nFirst ten edges:\n")

    for u, v, data in list(
        G.edges(data=True)
    )[:10]:

        print(
            f"{u:3d} -- {v:3d} "
            f"distance="
            f"{data['distance']:.2f} Å "
            f"sequence="
            f"{data['sequence_neighbor']} "
            f"contact="
            f"{data['spatial_contact']}"
        )

    # 8. Visualize the graph using C-alpha x-y coordinates
    positions = {
        i: residue["ca_coord"][:2]
        for i, residue
        in enumerate(residues)
    }

    plt.figure(figsize=(10, 8))

    nx.draw(
        G,
        positions,
        with_labels=False,
        node_size=30,
        width=0.5
    )

    plt.title("Residue-Level Cα Graph of 3MUF")

    plt.savefig(
        "graph-protein-3muf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    ```


    In this representation, each graph node corresponds to one amino acid residue and stores 
    the three-dimensional coordinate of its C$^\alpha$ atom. Sequence-neighbor edges preserve 
    the primary protein sequence, while spatial-contact edges introduce information about 
    the folded three-dimensional structure.

    The graph therefore combines two different forms of protein information:

    ```text
    Amino-acid sequence
            |
            v
    Adjacent residues
    (i, i+1)
            |
            +--------------------+
                                 |
                                 v
                            Residue graph
                                 ^
                                 |
            +--------------------+
            |
            v
    C-alpha coordinates
            |
            v
    Pairwise distances
            |
            v
    3D contact edges
    (d_ij <= cutoff)   
    ```

    This type of residue-level graph is useful for graph neural networks because it provides 
    a compact representation of protein structure while preserving both local sequence 
    relationships and long-range contacts created by protein folding.


**Advantages of Graph Representations**:

- Natural for molecules (atoms connected by bonds)
- Permutation invariant (atom order doesn't matter)
- Captures topology and local structure
- Enables Graph Neural Networks

**Limitations**:

- More complex to implement
- Computationally expensive for large molecules
- Requires specialized neural network architectures

#### Atomic Cluster Expansion

The **Atomic Cluster Expansion (ACE)** (Phys. Rev. B 99, 014104, 2019) is a systematic representation of 
the local three-dimensional environment surrounding an atom. It describes neighboring atoms using radial 
and angular basis functions and combines these contributions into symmetry-adapted features.

ACE is particularly useful for molecular and atomistic machine learning because it can systematically 
incorporate interactions involving pairs, triplets, and higher-order groups of atoms. Increasing the body 
order and basis resolution provides progressively more detailed information about the local atomic environment.

For proteins, ACE can be applied to an all-atom representation or to a coarse-grained representation based on 
selected atoms such as the C$^\alpha$ atoms of amino acid residues.

??? info "Advanced material"

    *Local Atomic Environment*

    For a central atom $i$, the local environment is defined as the set of neighboring atoms lying within 
    a cutoff radius $r_{\mathrm{cut}}$:

    $$
    \mathcal{N}_i
    =
    \left{
    j ; | ;
    r_{ij} < r_{\mathrm{cut}}
    \right},
    $$

    where

    $$
    r_{ij}
    =
    \left|
    \mathbf{r}_j-\mathbf{r}_i
    \right|
    $$

    is the distance between atoms $i$ and $j$.

    The relative position vector is

    $$
    \mathbf{r}_{ij}=\mathbf{r}_j-\mathbf{r}_i.
    $$

    Because ACE is constructed from relative atomic positions, a global translation of the 
    molecule does not change the representation.

    *Atomic Density Expansion*

    The local environment is expanded using radial basis functions and spherical harmonics. 
    A one-particle basis function can be written schematically as

    $$
    \phi_{znlm}
    \left(
    \mathbf{r}_{ij}
    \right)
    =
    \delta_{z,z_j}
    R_{nl}
    \left(
    r_{ij}
    \right)
    Y_l^m
    \left(
    \hat{\mathbf{r}}_{ij}
    \right),
    $$

    where $z_j$ represents the chemical species of neighboring atom $j$, $R_{nl}(r)$ is a 
    radial basis function, and $Y_l^m$ is a spherical harmonic describing angular information.

    The contributions from all neighbors are summed to obtain the atomic-density projections

    $$
    A_{znlm}^{(i)}=
    \sum_{j\in\mathcal{N}*i}
    \delta*{z,z_j}
    R_{nl}
    \left(
    r_{ij}
    \right)
    Y_l^m
    \left(
    \hat{\mathbf{r}}_{ij}
    \right).
    $$

    These coefficients provide the fundamental building blocks of the Atomic Cluster Expansion.

    *Many-Body Information*

    Products of the density projections introduce higher-order information about groups of 
    neighboring atoms. Schematically,

    $$
    A_{\nu_1}^{(i)}
    A_{\nu_2}^{(i)}
    \cdots
    A_{\nu_p}^{(i)}
    $$

    contains information involving increasingly large atomic clusters.
    For example, low-order terms primarily describe radial relationships between the central atom 
    and its neighbors, while higher-order terms encode angular and many-body correlations involving 
    several atoms simultaneously.
    This creates a systematic hierarchy:

    ```text
    central atom
        |
        +---- neighbor
        |        |
        |        +------ radial information
        |
        +---- neighbor pair
        |        |
        |        +------ angular information
        |
        +---- larger clusters
                |
                +------ higher-body correlations
    ```

    The complexity of the representation can therefore be increased systematically by including 
    higher body orders.

    *Rotationally Invariant ACE Features*

    The spherical-harmonic components can be coupled so that the final features are invariant 
    under rotation. A rotationally invariant ACE basis function can be written schematically as

    $$
    B_{\alpha}^{(i)}
    =
    \sum_{\mathbf{m}}
    C_{\mathbf{m}}^{\alpha}
    \prod_{t=1}^{p}
    A_{z_t n_t l_t m_t}^{(i)},
    $$

    where $C_{\mathbf{m}}^{\alpha}$ represents angular-coupling coefficients.
    The resulting descriptor for atomic environment $i$ is

    $$
    \mathbf{B}_i
    =
    \left[
    B_1^{(i)},
    B_2^{(i)},
    \ldots,
    B_D^{(i)}
    \right].
    $$

    For invariant ACE descriptors, rotating or translating the entire molecular structure leaves these 
    scalar features unchanged. Permutations of equivalent neighboring atoms also do not change the representation.
    ACE can additionally be formulated to generate equivariant vector and tensor features when the 
    predicted quantity must transform under rotation.

    *ACE for Proteins*

    For an atomistic protein structure, each atom can have its own local ACE representation:

    $$
    \mathbf{B}_1,
    \mathbf{B}_2,
    \ldots,
    \mathbf{B}_N.
    $$

    The chemical species channels can distinguish atoms such as carbon, nitrogen, oxygen, sulfur, and 
    hydrogen. These descriptors can then be used as input to machine learning models for atomistic 
    properties or interatomic potentials.
    A simpler coarse-grained protein representation can instead assign one point to each amino acid 
    using its C$^\alpha$ coordinate:

    $$
    \mathbf{r}*i^{,C*\alpha}
    =
    \begin{bmatrix}
    x_i\
    y_i\
    z_i
    \end{bmatrix}.
    $$

    Local environments are then constructed from neighboring C$^\alpha$ atoms:

    $$
    \mathcal{N}*i^{C*\alpha}
    =
    \left{
    j :
    \left|
    \mathbf{r}*j^{C*\alpha}
    -
    \mathbf{r}*i^{C*\alpha}
    \right|
    <
    r_{\mathrm{cut}}
    \right}.
    $$

    This coarse-grained representation captures aspects of local protein geometry but, 
    if only C$^\alpha$ coordinates are used, it does not contain the detailed atomic chemistry 
    of the amino-acid side chains.

#### Example: Low-Order ACE Representation of 3MUF

??? note "Example"

    The following example illustrates the basic ideas behind ACE using the C$^\alpha$ atoms from chain A of `3MUF.pdb`.

    For clarity, the example implements a **small low-order ACE-style invariant basis directly in Python** 
    rather than relying on a specialized ACE potential package. It includes radial two-body information 
    and angular three-body information.

    A smooth cutoff function is first defined as

    $$
    f_{\mathrm{cut}}(r)
    =
    \frac{1}{2}
    \left[
    \cos
    \left(
    \frac{\pi r}{r_{\mathrm{cut}}}
    \right)
    +1
    \right],
    $$

    for

    $$
    r < r_{\mathrm{cut}},
    $$

    and zero otherwise.

    Simple radial basis functions are then constructed as

    $$
    R_n(r)
    =
    f_{\mathrm{cut}}(r)
    \left(
    \frac{r}{r_{\mathrm{cut}}}
    \right)^n.
    $$

    A low-order radial invariant can be calculated as

    $$
    B_n^{(2)}
    =
    \sum_{j\in\mathcal{N}*i}
    R_n(r*{ij}).
    $$

    Angular information involving the central atom $i$ and two neighbors $j$ and $k$ 
    can be incorporated using Legendre polynomials:

    $$
    B_{nl}^{(3)}
    =
    \sum_{j<k}
    R_n(r_{ij})
    R_n(r_{ik})
    P_l
    \left(
    \cos\theta_{jik}
    \right),
    $$

    where

    $$
    \cos\theta_{jik}
    =
    \frac{
    \mathbf{r}*{ij}\cdot\mathbf{r}*{ik}
    }{
    r_{ij}r_{ik}
    }.
    $$

    Because these quantities depend on distances and angles rather than the absolute orientation 
    of the protein, they are invariant to global rotations.

    ```python
    import numpy as np

    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    from numpy.polynomial.legendre import Legendre

    # 1. Load protein structure
    pdb_file = "3MUF.pdb"

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "3MUF",
        pdb_file
    )

    model = structure[0]
    chain = model["A"]

    # 2. Extract C-alpha coordinates
    residues = []

    for residue in chain:

        if not is_aa(
            residue,
            standard=True
        ):
            continue

        if "CA" not in residue:
            continue

        residues.append({
            "name":
                residue.get_resname(),

            "number":
                residue.id[1],

            "coord":
                residue["CA"]
                .get_coord()
                .astype(float)
        })

    coordinates = np.array([
        residue["coord"]
        for residue in residues
    ])

    print("Number of C-alpha atoms:",len(coordinates))

    # 3. Define smooth cutoff function
    def cutoff_function(r,r_cut):

        if r >= r_cut:
            return 0.0

        return 0.5 * (np.cos(np.pi * r / r_cut) + 1.0)

    # 4. Define radial basis
    def radial_basis(r,n,r_cut):

        return (
            cutoff_function(r,r_cut)*(r / r_cut) ** n
        )

    # 5. Calculate low-order ACE-style features
    def calculate_ace_features(
        center_index,
        coordinates,
        r_cut=8.0,
        n_max=4,
        l_max=3
    ):

        center = coordinates[center_index]
        neighbor_vectors = []

        # Find neighbors
        for j, coord in enumerate(
            coordinates
        ):

            if j == center_index:
                continue

            vector = coord - center

            distance = np.linalg.norm(vector)

            if distance < r_cut:

                neighbor_vectors.append(
                    (vector,distance)
                )

        features = []

        # Two-body radial features
        for n in range(
            n_max
        ):

            value = 0.0

            for vector, distance in (neighbor_vectors):

                value += radial_basis(distance,n,r_cut)

            features.append(value)

        # Three-body angular features
        for n in range(n_max):

            for l in range(l_max + 1):

                value = 0.0
                P_l = Legendre.basis(l)

                for j in range(
                    len(
                        neighbor_vectors
                    )
                ):

                    vector_j, r_j = (
                        neighbor_vectors[j]
                    )

                    for k in range(
                        j + 1,
                        len(neighbor_vectors)
                    ):

                        vector_k, r_k = (
                            neighbor_vectors[k]
                        )

                        cos_theta = np.dot(
                            vector_j,
                            vector_k
                        ) / (r_j * r_k)

                        # Numerical protection
                        cos_theta = np.clip(
                            cos_theta,
                            -1.0,
                            1.0
                        )

                        value += (
                            radial_basis(r_j,n,r_cut)
                            *
                            radial_basis(r_k,n,r_cut)
                            *
                            P_l(cos_theta)
                        )

                features.append(
                    value
                )

        return np.array(features,dtype=float)

    # 6. Calculate ACE-style descriptors
    #    for every C-alpha environment
    ace_features = np.array([

        calculate_ace_features(
            i,
            coordinates,
            r_cut=8.0,
            n_max=4,
            l_max=3
        )

        for i in range(len(coordinates))
    ])

    print("ACE feature matrix shape:",ace_features.shape)

    # 7. Examine one residue environment
    residue_index = 50

    residue = residues[residue_index]

    print("\nResidue:",residue["name"],residue["number"])

    print("First 10 ACE-style features:")

    print(ace_features[residue_index,:10])

    # 8. Compare two local environments
    i = 20
    j = 50

    descriptor_i = ace_features[i]
    descriptor_j = ace_features[j]

    similarity = np.dot(
        descriptor_i,
        descriptor_j
    ) / (
        np.linalg.norm(descriptor_i)
        *
        np.linalg.norm(descriptor_j)
    )

    print("\nResidue 1:",residues[i]["name"],residues[i]["number"])
    print("Residue 2:",residues[j]["name"],residues[j]["number"])

    print("Descriptor similarity:",similarity)
    ```

    The resulting matrix has the form

    $$
    B
    =
    \begin{bmatrix}
    \mathbf{B}_1\\
    \mathbf{B}_2\\
    \vdots\\
    \mathbf{B}_N
    \end{bmatrix},
    $$

    where each row represents the local structural environment surrounding one C$^\alpha$ atom.

    The example combines two levels of structural information. The radial terms 
    describe the distribution of neighboring residues around the central residue, while 
    the angular terms describe the relative arrangement of pairs of neighboring residues.

    Consequently, two residues can have similar ACE descriptors when their surrounding 
    three-dimensional environments have similar geometries, even if they occur at different positions in the protein sequence.

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

A **Random Forest** is an ensemble learning method that combines the predictions of many 
decision trees. Instead of relying on a single tree, which can be highly sensitive to variations 
in the training data, a Random Forest introduces randomness during tree construction and combines 
the resulting predictions. This generally produces a model with lower variance and better 
generalization than an individual deep decision tree.

#### How Random Forests Work

Each tree in the forest is typically trained using a **bootstrap sample** of the training dataset. 
Bootstrap sampling draws observations randomly with replacement, meaning that some training samples 
may appear multiple times while others are not selected for a particular tree.

Additional randomness is introduced when splitting nodes. Rather than considering every available 
feature at each split, the algorithm evaluates only a randomly selected subset of features. This 
encourages the individual trees to learn somewhat different patterns and reduces the correlation 
between them.

The individual decision trees are usually grown independently and can be relatively deep. Their 
predictions are then combined. For a regression problem containing $M$ trees, the Random Forest prediction 
is the average

$$
\hat{y}(\mathbf{x})
=
\frac{1}{M}
\sum_{m=1}^{M}
T_m(\mathbf{x}),
$$

where $T_m(\mathbf{x})$ is the prediction produced by tree $m$.

For classification, the predictions of the individual trees are combined to determine the final 
class prediction, commonly through aggregated class probabilities or voting.

The main advantage of the ensemble comes from averaging many partially decorrelated trees. A 
single deep decision tree may fit noise or small variations in the training data, whereas 
averaging many different trees tends to reduce these unstable fluctuations.

#### Example: Predicting Experimental Band Gaps of Inorganic Materials

The following example uses the **Matbench experimental band-gap dataset**, which contains 
inorganic compositions together with experimentally measured band gaps. Each chemical 
composition is converted into numerical elemental-property descriptors using the Magpie 
descriptor set provided by Matminer.

Random Forests do not normally require feature standardization because decision-tree splits 
depend on feature thresholds rather than distances between samples.

??? note "Example"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    from pymatgen.core import Composition

    from matminer.datasets import load_dataset
    from matminer.featurizers.composition import ElementProperty

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import (
        train_test_split,
        cross_val_score
    )
    from sklearn.metrics import (
        r2_score,
        mean_squared_error,
        mean_absolute_error
    )

    # 1. Load the materials-science dataset
    df = load_dataset(
        "matbench_expt_gap"
    )

    print(df.head())

    print(
        "\nNumber of materials:",
        len(df)
    )

    # 2. Ensure compositions are pymatgen objects
    df["composition"] = df[
        "composition"
    ].apply(
        lambda x:
        x if isinstance(x, Composition)
        else Composition(x)
    )

    # 3. Generate composition-based descriptors
    featurizer = ElementProperty.from_preset(
        "magpie",
        impute_nan=True
    )

    df = featurizer.featurize_dataframe(
        df,
        col_id="composition",
        ignore_errors=False
    )

    # Descriptor names
    feature_names = (
        featurizer.feature_labels()
    )

    # 4. Construct input features and target
    X = df[feature_names]

    # Experimental band gap in eV
    y = df["gap expt"]

    print(
        "\nNumber of features:",
        X.shape[1]
    )

    # 5. Split into training and test sets
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # 6. Define the Random Forest model
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    # 7. Train the model
    rf_reg.fit(
        X_train,
        y_train
    )

    # 8. Predict the test data
    y_pred = rf_reg.predict(
        X_test
    )

    # 9. Evaluate model performance
    r2 = r2_score(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    print(
        f"\nTest R²:   {r2:.3f}"
    )

    print(
        f"Test MAE:  {mae:.3f} eV"
    )

    print(
        f"Test RMSE: {rmse:.3f} eV"
    )

    print(
        f"OOB R²:    "
        f"{rf_reg.oob_score_:.3f}"
    )

    # 10. Cross-validation on the training set
    cv_scores = cross_val_score(
        rf_reg,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print(
        "\nCross-validation R²:"
    )

    print(
        f"{cv_scores.mean():.3f} "
        f"± {cv_scores.std():.3f}"
    )

    # 11. Impurity-based feature importance
    importances = (
        rf_reg.feature_importances_
    )

    indices = np.argsort(
        importances
    )[::-1]

    print(
        "\nTop 10 features:"
    )

    for rank, idx in enumerate(
        indices[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{feature_names[idx]}: "
            f"{importances[idx]:.4f}"
        )

    # 12. Visualize the 10 most important features
    top_indices = indices[:10]

    top_names = [
        feature_names[i]
        for i in top_indices
    ]

    top_importances = importances[
        top_indices
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        range(len(top_names)),
        top_importances
    )

    plt.yticks(
        range(len(top_names)),
        top_names
    )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Impurity-Based Importance"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "random-forest-materials.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    ```

In this example, each material is initially represented by its chemical composition. 
The Magpie featurizer transforms that composition into numerical descriptors containing 
statistics of elemental properties such as atomic number, atomic mass, electronegativity, 
and related quantities. The Random Forest then learns relationships between these 
composition-derived descriptors and the experimental band gap.

Because the target is continuous,

$$
y = E_{\mathrm{gap}},
$$

the problem is formulated as a regression task. The final model predicts the experimental 
band gap of previously unseen inorganic compositions.

#### Out-of-Bag Estimation

Bootstrap sampling also provides a convenient internal estimate of model performance. 
When a bootstrap sample containing $N$ draws is constructed from $N$ training 
observations, some samples are not selected for that particular tree. On average, 
approximately (36.8%) of the observations are excluded and are referred to as **out-of-bag (OOB) samples**.

These observations can be passed through trees for which they were not included during 
training, providing an internal estimate of predictive performance without creating an additional validation subset.

In scikit-learn, OOB estimation can be enabled using

```python
oob_score=True
```

and the resulting score can be accessed through

```python
rf_reg.oob_score_
```

OOB evaluation is useful, but it does not replace a final independent test set when 
an unbiased final performance estimate is required.

#### Feature Importance

Random Forests provide an impurity-based feature importance through

```python
rf_reg.feature_importances_
```

This quantity measures how much each feature contributes, on average, to reducing the 
splitting criterion across the trees in the forest. For regression, this is commonly related 
to reductions in squared-error impurity.

Although fast and convenient, impurity-based feature importance can be misleading, 
particularly when features differ substantially in cardinality or when several variables are strongly correlated.

**Permutation importance** provides an alternative approach. After the model has been 
trained, one feature is randomly shuffled while all other features remain unchanged. If model 
performance decreases substantially, the model was relying strongly on that feature.

??? note "Example"

    ```python
    import numpy as np

    from sklearn.inspection import (
        permutation_importance
    )

    # Calculate permutation importance
    # using the held-out test set
    result = permutation_importance(
        rf_reg,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    # Rank features
    indices = np.argsort(
        result.importances_mean
    )[::-1]

    print(
        "Top permutation importances:\n"
    )

    for rank, idx in enumerate(
        indices[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{feature_names[idx]}: "
            f"{result.importances_mean[idx]:.4f} "
            f"± "
            f"{result.importances_std[idx]:.4f}"
        )
    ```

Permutation importance measures the importance of a feature for the 
**specific trained model and evaluation dataset**. It should therefore not 
be interpreted as proof that a descriptor has a direct causal influence on 
the material property.

#### Advantages

Random Forests can capture nonlinear relationships and interactions between multiple 
descriptors without requiring an explicit mathematical form for those relationships. 
They generally require little preprocessing, and numerical features do not need to be 
standardized before training. Combining many decision trees usually provides substantially 
more stable predictions than using a single unrestricted decision tree.

Random Forests can also work effectively with mixtures of informative and weakly 
informative variables and provide several approaches for investigating feature importance. 
Their parallel tree structure makes training and prediction suitable for multicore computation.

#### Limitations

Random Forests can become computationally and memory intensive when very large numbers of 
trees, samples, or descriptors are used. Their predictions are also less directly interpretable 
than those of a single decision tree because the final result combines many individual models.

A particularly important limitation for scientific applications is **extrapolation**. 
Decision-tree predictions are constructed from target values observed in regions of the 
training data, so Random Forests generally perform poorly when asked to predict behavior 
far outside the domain represented during training.

Feature importance must also be interpreted carefully. Correlated materials descriptors may 
contain overlapping information, making the importance assigned to any individual variable difficult to interpret.

#### Hyperparameter Tuning

The behavior of a Random Forest is controlled by several important hyperparameters. `n_estimators` 
specifies the number of trees in the ensemble. `max_depth`, `min_samples_split`, and `min_samples_leaf` 
control tree complexity, while `max_features` determines how many descriptors can be considered 
when searching for a split. These parameters influence the balance between model flexibility, variance, 
computational cost, and generalization.

Rather than selecting these values manually, they can be explored systematically using cross-validation.

??? note "Example"

    ```python
    from sklearn.ensemble import (
        RandomForestRegressor
    )

    from sklearn.model_selection import (
        RandomizedSearchCV
    )

    # Hyperparameter search space
    param_dist = {

        "n_estimators": [100,200,500],

        "max_depth": [10,20,30,None],

        "min_samples_split": [2,5,10],

        "min_samples_leaf": [1,2,4],

        "max_features": ["sqrt","log2",0.5,1.0]
    }

    # Define randomized search
    random_search = RandomizedSearchCV(

        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),

        param_distributions=param_dist,

        n_iter=20,

        cv=5,

        scoring="r2",

        random_state=42,

        n_jobs=-1
    )

    # Tune using training data only
    random_search.fit(
        X_train,
        y_train
    )

    print(
        "Best parameters:"
    )

    print(
        random_search.best_params_
    )


    print(
        "\nBest cross-validation R²:"
    )

    print(
        f"{random_search.best_score_:.3f}"
    )

    # Evaluate the optimized model
    # on the untouched test set
    best_model = (
        random_search.best_estimator_
    )

    y_pred_best = best_model.predict(
        X_test
    )

    test_r2 = r2_score(
        y_test,
        y_pred_best
    )

    print(
        f"\nOptimized test R²: "
        f"{test_r2:.3f}"
    )
    ```

The hyperparameter search is performed using only the training set. The test 
set remains untouched until the final model has been selected, ensuring that it 
provides an independent estimate of generalization performance.

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

**Quantitative Structure–Activity Relationship (QSAR)** modeling aims to establish mathematical 
relationships between molecular structure and a measured biological activity or physicochemical 
property. Molecules are typically converted into numerical representations, such as molecular 
descriptors or fingerprints, which are then used as input to regression or classification models.

Traditional QSAR models commonly rely on engineered molecular features, although the predictive 
algorithm itself may range from simple linear regression to methods such as support vector 
machines, Random Forests, or neural networks.

Typical QSAR applications include:

* prediction of biological activity,
* toxicity assessment,
* aqueous solubility prediction,
* binding affinity estimation,
* environmental fate and risk assessment.

A typical QSAR workflow consists of three main stages:

1. Convert each molecule into numerical descriptors or fingerprints.
2. Assemble these molecular features into a feature matrix.
3. Train a regression or classification model to predict the target activity or property.

For a dataset containing $N$ molecules and $F$ molecular descriptors, the input matrix 
can be written as

$$
X
=
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1F}\\
x_{21} & x_{22} & \cdots & x_{2F}\\
\vdots & \vdots & \ddots & \vdots\\
x_{N1} & x_{N2} & \cdots & x_{NF}
\end{bmatrix},
$$

where each row represents one molecule and each column corresponds to a molecular descriptor.

For a regression problem, the model learns a mapping

$$
\hat{y}
=
f(\mathbf{x}),
$$

where $\mathbf{x}$ is the molecular feature vector and $\hat{y}$ is the predicted molecular property.

#### Example: QSAR Model for Aqueous Solubility

The following example uses the **Delaney ESOL dataset**, which contains molecular 
structures represented as SMILES strings together with experimentally measured aqueous solubilities.

RDKit is used to calculate a small set of molecular descriptors, and a Random Forest 
regression model is trained to predict the measured logarithmic solubility.

??? note "Example"

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    # 1. Load the Delaney ESOL dataset
    url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/"
        "datasets/delaney-processed.csv"
    )

    df = pd.read_csv(url)

    print(df.head())

    print(
        "\nNumber of molecules:",
        len(df)
    )

    # 2. Calculate molecular descriptors
    def calculate_descriptors(smiles):
        """
        Calculate a small set of molecular descriptors
        from a SMILES string.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return {
            "MW":
                Descriptors.MolWt(mol),

            "LogP":
                Crippen.MolLogP(mol),

            "TPSA":
                Descriptors.TPSA(mol),

            "HBD":
                Descriptors.NumHDonors(mol),

            "HBA":
                Descriptors.NumHAcceptors(mol),

            "RotatableBonds":
                Descriptors.NumRotatableBonds(mol),

            "RingCount":
                Descriptors.RingCount(mol),

            "FractionCSP3":
                Descriptors.FractionCSP3(mol)
        }

    descriptor_rows = [
        calculate_descriptors(smiles)
        for smiles in df["smiles"]
    ]

    X = pd.DataFrame(
        descriptor_rows
    )

    # 3. Define the target property
    y = df[
        "measured log solubility in mols per litre"
    ]

    # Remove invalid rows if necessary
    valid_rows = X.notna().all(axis=1)

    X = X.loc[valid_rows].reset_index(
        drop=True
    )

    y = y.loc[valid_rows].reset_index(
        drop=True
    )

    print(
        "\nDescriptor matrix shape:",
        X.shape
    )

    # 4. Split into training and test sets
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    # 5. Define the QSAR regression model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # 6. Train the model
    model.fit(
        X_train,
        y_train
    )

    # 7. Predict solubility for the test molecules
    y_pred = model.predict(
        X_test
    )

    # 8. Evaluate predictive performance
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )


    print(f"\nTest MAE:  {mae:.3f}")

    print(f"Test RMSE: {rmse:.3f}")

    print(f"Test R²:   {r2:.3f}")

    # 9. Compare predicted and experimental values
    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    # Ideal prediction line
    minimum = min(
        y_test.min(),
        y_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )


    plt.xlabel("Experimental Log Solubility")

    plt.ylabel("Predicted Log Solubility")

    plt.title("QSAR Prediction of Aqueous Solubility")

    plt.tight_layout()

    plt.savefig(
        "qsar-solubility.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    ```


In this example, each molecule is converted from its SMILES representation into a 
descriptor vector containing molecular weight, lipophilicity, polar surface area, 
hydrogen-bonding properties, molecular flexibility, ring count, and carbon saturation.

For example,

$$
\mathbf{x}_i
=
[MW,LogP,TPSA,HBD,HBA,
N_{\mathrm{rot}},
N_{\mathrm{rings}},
f_{sp^3}
].
$$

The Random Forest learns a relationship between these molecular descriptors 
and the experimentally measured logarithmic aqueous solubility.

The target variable is

$$
y
=
\log_{10}
\left(
S_{\mathrm{mol/L}}
\right),
$$

where $S_{\mathrm{mol/L}}$ represents aqueous solubility in moles per liter.
Model performance is evaluated using several regression metrics:
**Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, or the coefficient of determination $R^2$. 

This example illustrates the central QSAR workflow:

```text
SMILES
   |
   v
Molecular descriptors
   |
   v
Feature matrix
   |
   v
QSAR regression model
   |
   v
Predicted molecular property
```

In practical QSAR studies, additional steps are usually required, including descriptor selection, 
cross-validation, hyperparameter optimization, applicability-domain analysis, and careful selection 
of training and test compounds. Scaffold-based splitting can also provide a more demanding 
evaluation when the objective is to predict properties for structurally novel molecules.

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

## 5. Practical Example: Solubility Prediction

This example ties together the descriptors, models, and evaluation metrics from the
previous sections in a complete workflow. The task is aqueous solubility prediction on
the ESOL (Delaney) dataset, a standard benchmark of 1128 small molecules
(Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000–1005). The target is the base-10
logarithm of aqueous solubility,

$$
y = \log_{10} S,
$$

where $S$ is the solubility in mol/L. Working in log space is standard, since solubility
spans many orders of magnitude.

### Complete workflow

??? note "Example"

    ```python
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import matplotlib.pyplot as plt

    # Step 1: Load data
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
            'FractionCSP3': Descriptors.FractionCSP3(mol),
        }

    # Build features and targets together so that they stay aligned even if
    # some SMILES fail to parse. (Slicing y with [:len(X)] would silently
    # mispair rows whenever a molecule in the middle is dropped.)
    target_col = 'measured log solubility in mols per litre'

    records, targets, valid_smiles = [], [], []
    for smiles, solubility in zip(df['smiles'], df[target_col]):
        descriptors = calculate_descriptors(smiles)
        if descriptors is not None:
            records.append(descriptors)
            targets.append(solubility)
            valid_smiles.append(smiles)

    X_desc = pd.DataFrame(records)
    y = np.array(targets)

    print(f"Valid molecules: {len(X_desc)}")

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_desc, y, test_size=0.2, random_state=42
    )

    # Step 4: Train model
    # Note: feature scaling is intentionally omitted here. Random forests
    # are invariant to monotonic per-feature rescaling, so standardization
    # has no effect. Scaling matters for linear models, SVMs, k-NN, and
    # neural networks, but not for tree-based models.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Evaluate
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nResults (random split):")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    # Step 6: Cross-validation (more stable than a single split)
    cv_scores = cross_val_score(model, X_desc, y, cv=5, scoring="r2")
    print(f"\n5-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Step 7: Visualize
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', lw=2
    )
    plt.xlabel('True solubility (log M)')
    plt.ylabel('Predicted solubility (log M)')
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

    print("\nTop 5 most important features:")
    print(feature_importance_df.head())
    ```

**Expected results (random split)**:

- Test RMSE: ~0.7-0.9 log units
- R²: ~0.75-0.85
- Key features: LogP, molecular weight, polar surface area

### A more realistic evaluation: scaffold split

A random split lets molecules with the same scaffold appear in both training and test
sets, which inflates performance. Evaluating on a scaffold split (where whole scaffold
groups are held out) estimates how well the model generalizes to genuinely new chemical
series. Scaffold-split scores are typically lower than random-split scores, and are the
more honest number to report.

??? note "Example"

    ```python
    # Reuses the scaffold_split function from Section 4.2.
    from rdkit import Chem  # noqa: F401  (used inside scaffold_split)

    train_idx, test_idx = scaffold_split(valid_smiles, test_size=0.2)

    X_arr = X_desc.to_numpy()
    X_train_s, X_test_s = X_arr[train_idx], X_arr[test_idx]
    y_train_s, y_test_s = y[train_idx], y[test_idx]

    model_s = RandomForestRegressor(n_estimators=100, random_state=42)
    model_s.fit(X_train_s, y_train_s)

    y_pred_s = model_s.predict(X_test_s)

    print("Results (scaffold split):")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_s, y_pred_s)):.3f}")
    print(f"MAE:  {mean_absolute_error(y_test_s, y_pred_s):.3f}")
    print(f"R²:   {r2_score(y_test_s, y_pred_s):.3f}")
    ```

## 6. Practical Example: working with SELFIES

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
    # LogP > 5: too lipophilic (Lipinski's rule of five)

    # Topological polar surface area
    tpsa = Descriptors.TPSA(mol)
    print(f"TPSA: {tpsa:.2f} Å²")
    # TPSA <= 140 Å²: associated with good oral bioavailability (Veber rule).
    # Blood-brain barrier penetration typically requires TPSA below ~90 Å².

    # Molar Refractivity
    mr = Crippen.MolMR(mol)
    print(f"Molar Refractivity: {mr:.2f}")

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

    from rdkit.Chem import GraphDescriptors

    # Balaban J index (molecular branching)
    balaban = GraphDescriptors.BalabanJ(mol)

    # Bertz complexity index
    bertz = GraphDescriptors.BertzCT(mol)

    # Chi indices (connectivity)
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)

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

    # Radius of gyration is a length, so its unit is Å (not Å²)
    print(f"Radius of Gyration: {radius_of_gyration:.2f} Å")

    def lipinski_rule_of_five(mol):
        """
        Estimate oral drug-likeness using Lipinski's rule of five.
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
    print(f"Passes Lipinski's rule: {is_druglike}")


    from rdkit.Chem import QED

    qed_score = QED.qed(mol)
    print(f"QED Score: {qed_score:.3f}")
    # Range: [0, 1], higher is more drug-like
    # Combines several molecular properties into a single score


    # Synthetic accessibility score.
    # Note: sascorer lives in RDKit's Contrib directory, which must be
    # present in the installation (it ships with the standard conda build).
    from rdkit.Chem import RDConfig
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    sa_score = sascorer.calculateScore(mol)
    print(f"SA Score: {sa_score:.2f}")
    # Range: [1, 10]
    # 1: easy to synthesize
    # 10: difficult to synthesize


    def calculate_molecular_descriptors(selfies_str):
        """
        Comprehensive descriptor calculation from a SELFIES string.
        """
        # Convert SELFIES to SMILES, then to a molecule
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Note: explicit hydrogens are not added here. The 2D descriptors
        # below (MW, TPSA, H-bond counts, ring counts, QED, ...) already
        # account for implicit hydrogens, so AddHs would not change them.

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

            # Surface area and charge (VSA) descriptors
            'LabuteASA': Descriptors.LabuteASA(mol),
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),

            # Counts
            'NumCarbon': sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()),
            'NumNitrogen': sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
            'NumOxygen': sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
            'NumHalogens': sum(atom.GetAtomicNum() in [9, 17, 35, 53] for atom in mol.GetAtoms()),

            # Saturation
            'FractionCsp3': Descriptors.FractionCSP3(mol),

            # Drug-likeness
            'QED': QED.qed(mol),
        }

        return descriptors

    # Example usage - convert SMILES to SELFIES first
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

    import networkx as nx

    def mol_to_graph(selfies_str):
        """Convert a SELFIES molecule to a NetworkX graph."""
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Could not parse molecule from SELFIES: {selfies_str}")

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

    def get_adjacency_matrix(selfies_str, max_atoms=50):
        """Get a padded adjacency matrix from a SELFIES string."""
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Could not parse molecule from SELFIES: {selfies_str}")

        num_atoms = mol.GetNumAtoms()

        if num_atoms > max_atoms:
            raise ValueError(
                f"Molecule contains {num_atoms} atoms but max_atoms={max_atoms}"
            )

        # Initialize matrix
        adj_matrix = np.zeros((max_atoms, max_atoms), dtype=int)

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

## 7. Practical Example: A complete ML Pipeline

### Task

Build a complete machine learning pipeline to predict molecular solubility. The dataset is
the ESOL (Delaney) set (J. Chem. Inf. Comput. Sci. 2004, 44, 1000–1005,
[link](https://pubs.acs.org/doi/10.1021/ci034243x)).

### Dataset

In this example, we construct a complete machine learning workflow for predicting 
**aqueous solubility** from molecular descriptors.
The dataset is the processed **Delaney ESOL dataset**, which contains molecular structures 
represented as SMILES strings together with experimentally measured logarithmic aqueous solubilities.

The target property is

$$
y =
\log_{10}
\left(
S_{\mathrm{mol/L}}
\right),
$$

where $S_{\mathrm{mol/L}}$ is the aqueous solubility expressed in moles per liter.

??? note "Example"

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor
    )
    from sklearn.svm import SVR

    from sklearn.model_selection import (
        train_test_split,
        cross_val_score,
        KFold,
        RandomizedSearchCV
    )

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    # Load the ESOL dataset
    url = (
        "https://raw.githubusercontent.com/"
        "deepchem/deepchem/master/datasets/"
        "delaney-processed.csv"
    )

    data = pd.read_csv(url)

    # Column names in the processed ESOL dataset
    smiles_col = "smiles"

    target_col = (
        "measured log solubility in mols per litre"
    )

    print(f"Dataset size: {len(data)}")

    print(data.head())

    print("\nColumn names:")

    print(data.columns.tolist())
    ```

The dataset contains both molecular structures and several previously calculated properties. 
In this example, however, molecular descriptors are recalculated directly from the SMILES 
strings using RDKit.

### Step 1: Feature Engineering

Machine learning algorithms require numerical inputs. The molecular SMILES strings are therefore 
converted into a collection of physicochemical and structural descriptors.
For molecule $i$, the resulting feature vector can be represented as

$$
\mathbf{x}_i
=
[
MW,
LogP,
HBD,
HBA,
N_{\mathrm{rot}},
N_{\mathrm{arom}},
TPSA,
\ldots
].
$$

These descriptor vectors are combined into a feature matrix

$$
X
\in
\mathbb{R}^{N\times F},
$$

where $N$ is the number of molecules and $F$ is the number of molecular descriptors.

??? note "Example"

    ```python
    def calculate_molecular_features(smiles):
        """
        Calculate molecular descriptors
        from a SMILES string.
        """

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        features = {

            "MolWt":
                Descriptors.MolWt(mol),

            "LogP":
                Descriptors.MolLogP(mol),

            "NumHDonors":
                Descriptors.NumHDonors(mol),

            "NumHAcceptors":
                Descriptors.NumHAcceptors(mol),

            "NumRotatableBonds":
                Descriptors.NumRotatableBonds(mol),

            "NumAromaticRings":
                Descriptors.NumAromaticRings(mol),

            "TPSA":
                Descriptors.TPSA(mol),

            "NumHeteroatoms":
                Descriptors.NumHeteroatoms(mol),

            "NumRings":
                Descriptors.RingCount(mol),

            "NumSaturatedRings":
                Descriptors.NumSaturatedRings(mol),
        }

        return features

    # Calculate descriptors for all molecules
    features_list = []
    valid_indices = []

    for idx, smiles in enumerate(
        data[smiles_col]
    ):

        features = (
            calculate_molecular_features(
                smiles
            )
        )

        if features is not None:

            features_list.append(features)

            valid_indices.append(idx)

    # Construct feature matrix
    X = pd.DataFrame(features_list)

    # Experimental solubility target
    y = data.loc[
        valid_indices,
        target_col
    ].reset_index(
        drop=True
    )

    print(f"\nFeature matrix shape: {X.shape}")

    print(f"Target shape: {y.shape}")

    print(
        f"Valid molecules: "
        f"{len(valid_indices)} / {len(data)}"
    )

    print("\nFeatures:")

    print(X.columns.tolist())
    ```

Tracking the valid molecule indices ensures that the descriptor matrix $X$ and 
target vector $y$ remain aligned if any SMILES string fails to parse.

### Step 2: Data Validation and Exploration

Before training a model, the generated descriptors and target values should be 
inspected for missing values, unusual ranges, and potential errors.
Simple exploratory analysis can also reveal relationships between individual 
descriptors and the target property.
The Pearson correlation coefficient between a descriptor $x$ and the target $y$ is

$$
r_{xy}
=
\frac{
\operatorname{cov}(x,y)
}{
\sigma_x \sigma_y
}.
$$

Correlation measures linear association and should not be interpreted as a measure of causality or as a complete description of predictive importance.

??? note "Example"

    ```python
    # Check missing values
    print("\nMissing values:")

    print(X.isnull().sum())

    # Check for non-finite values
    print(
        "\nNon-finite feature values:",
        np.sum(
            ~np.isfinite(
                X.to_numpy()
            )
        )
    )

    # Feature statistics
    print("\nFeature statistics:")

    print(X.describe())

    # Target statistics
    print("\nTarget statistics:")

    print(f"Mean: {y.mean():.3f}")

    print(f"Std:  {y.std():.3f}")

    print(f"Min:  {y.min():.3f}")

    print(f"Max:  {y.max():.3f}")

    # Feature-target correlations
    correlations = X.corrwith(
        y
    )

    correlations = correlations.sort_values()

    plt.figure(
        figsize=(10, 8)
    )

    correlations.plot(
        kind="barh"
    )

    plt.xlabel(
        "Pearson Correlation with Solubility"
    )

    plt.title(
        "Feature-Target Correlations"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_correlations.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    ```

A strong individual correlation may indicate that a descriptor contains 
useful information about solubility. However, nonlinear models can also 
learn relationships from descriptors that show relatively weak linear correlations.

### Step 3: Train-Test Split

The dataset is divided into a **training set** and an independent **test set**. 
The training set is used for model fitting, cross-validation, model selection, and 
hyperparameter optimization.

The test set is kept separate until the final evaluation.

??? note "Example"

    ```python
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    print(
        f"\nTraining set size: "
        f"{len(X_train)}"
    )

    print(
        f"Test set size: "
        f"{len(X_test)}"
    )
    ```

This example uses a random train-test split for simplicity. In realistic QSAR applications, 
a scaffold-based split can provide a more demanding evaluation because structurally related 
molecules are prevented from appearing in both the training and test sets.

### Step 4: Preprocessing

Different machine learning algorithms have different preprocessing requirements.

Models such as Ridge regression, Lasso regression, and Support Vector Regression are 
sensitive to the scale of the input features. For these models, the descriptors are 
standardized according to

$$
x_j'
=
\frac{
x_j-\mu_j
}{
\sigma_j
}.
$$

Tree-based models such as Random Forests and Gradient Boosting do not generally 
require feature standardization because their splits are based on feature thresholds.

To prevent **data leakage**, preprocessing is incorporated into a scikit-learn `Pipeline`. 
This ensures that the scaler is fitted only to the training portion of each cross-validation fold.

??? note "Example"

    ```python
    # Define model pipelines

    models = {

        "Ridge":

            Pipeline([
                ("scaler",StandardScaler()),
                ("model",Ridge(alpha=1.0))
            ]),

        "Lasso":

            Pipeline([
                ("scaler",StandardScaler()),
                ("model",Lasso(alpha=0.1,max_iter=10000))
            ]),

        "Random Forest":

            Pipeline([
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=1
                    )
                )
            ]),

        "Gradient Boosting":

            Pipeline([
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        random_state=42
                    )
                )
            ]),

        "SVR":

            Pipeline([
                ("scaler",StandardScaler()),
                (
                    "model",
                    SVR(
                        kernel="rbf",
                        C=1.0
                    )
                )
            ])
    }
    ```

The important advantage of this design is that every model receives the 
preprocessing appropriate for that algorithm while retaining a common 
interface for training and prediction.

### Step 5: Model Training with Cross-Validation

Rather than selecting a model from a single train-test result, several algorithms can be 
compared using cross-validation on the training set.
For $K$-fold cross-validation, the training data are divided into $K$ folds. Each model is 
trained on $K-1$ folds and evaluated on the remaining fold. This process is repeated until 
every fold has been used for validation.

For regression, the coefficient of determination is

$$
R^2
=
1-
\frac{
\sum_i
(y_i-\hat{y}_i)^2
}{
\sum_i
(y_i-\bar{y})^2
}.
$$

Higher values indicate better predictive performance, with

$$
R^2=1
$$

representing perfect predictions.

??? note "Example"

    ```python
    # Define cross-validation strategy

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Compare models
    print(
        "\nCross-validation results:"
    )

    cv_results = {}

    for name, model in models.items():

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        cv_results[name] = scores

        print(
            f"{name:20s}: "
            f"R² = "
            f"{scores.mean():.3f} "
            f"± "
            f"{scores.std():.3f}"
        )

    # Select the model with the highest mean CV R²
    best_model_name = max(
        cv_results,
        key=lambda name:
            cv_results[name].mean()
    )

    print(
        f"\nBest model: "
        f"{best_model_name}"
    )
    ```

Cross-validation is performed only on the training data. The test set 
remains untouched during model comparison.

### Step 6: Hyperparameter Tuning

Once the best-performing model family has been selected, its hyperparameters 
can be optimized using cross-validation.
The appropriate hyperparameters depend on the selected algorithm. For example, 
a Random Forest can be tuned by varying the number of trees, maximum tree depth, 
minimum number of samples required for splitting, and number of features considered at each split.

To make the workflow general, a separate search space is defined for each candidate model.

??? note "Example"

    ```python
    # Hyperparameter search spaces

    search_spaces = {

        "Ridge": {
            "model__alpha":
                [0.01,0.1,1.0,10.0,100.0]
        },


        "Lasso": {
            "model__alpha":
                [0.001,0.01,0.1,1.0]
        },

        "Random Forest": {

            "model__n_estimators":
                [100,200,500],

            "model__max_depth":
                [10,20,30,None],

            "model__min_samples_split":
                [2,5,10],

            "model__min_samples_leaf":
                [1,2,4],

            "model__max_features":
                ["sqrt","log2",0.5,1.0]
        },

        "Gradient Boosting": {

            "model__n_estimators":
                [100,200,500],

            "model__learning_rate":
                [0.01,0.05,0.1],

            "model__max_depth":
                [2,3,5],

            "model__min_samples_leaf":
                [1,2,4]
        },


        "SVR": {

            "model__C":
                [0.1,1.0,10.0,100.0],

            "model__gamma":
                ["scale","auto",0.01,0.1],

            "model__epsilon":
                [0.01,0.1,0.2]
        }
    }

    # Tune the selected model
    random_search = RandomizedSearchCV(

        estimator=models[
            best_model_name
        ],

        param_distributions=
            search_spaces[
                best_model_name
            ],

        n_iter=20,
        cv=cv,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    random_search.fit(
        X_train,
        y_train
    )

    print("\nBest parameters:")

    print(random_search.best_params_)

    print(
        f"\nBest CV R²: "
        f"{random_search.best_score_:.3f}"
    )

    # Best fitted pipeline
    best_model = (random_search.best_estimator_)
    ```

The parameter names contain the prefix `model__` because the estimator is 
stored inside a scikit-learn `Pipeline`.
The test set is still not used during this stage.

### Step 7: Final Evaluation

After model selection and hyperparameter tuning, the optimized pipeline is 
evaluated on the independent test set.
Three common regression metrics are used: the **Mean Absolute Error (MAE)**, the
 **Root Mean Squared Error (RMSE)**, and the coefficient of determination (R^2).
 
??? note "Example"

    ```python
    # Predictions

    y_pred_train = best_model.predict(
        X_train
    )

    y_pred_test = best_model.predict(
        X_test
    )

    # Training metrics
    train_r2 = r2_score(
        y_train,
        y_pred_train
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            y_pred_train
        )
    )

    train_mae = mean_absolute_error(
        y_train,
        y_pred_train
    )

    # Test metrics
    test_r2 = r2_score(
        y_test,
        y_pred_test
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred_test
        )
    )

    test_mae = mean_absolute_error(
        y_test,
        y_pred_test
    )

    # Report results
    print("\nFinal results:")

    print(
        "Training - "
        f"R²: {train_r2:.3f}, "
        f"RMSE: {train_rmse:.3f}, "
        f"MAE: {train_mae:.3f}"
    )

    print(
        "Test     - "
        f"R²: {test_r2:.3f}, "
        f"RMSE: {test_rmse:.3f}, "
        f"MAE: {test_mae:.3f}"
    )

    r2_gap = (train_r2 - test_r2)

    print(
        f"\nTrain-test R² gap: "
        f"{r2_gap:.3f}"
    )
    ```

A substantially better training score than test score can be evidence of overfitting, 
but no single threshold establishes whether a model generalizes well. The train-test 
difference should be interpreted together with cross-validation performance, dataset 
size, split strategy, and the intended application domain.

For this dataset, MAE and RMSE are expressed in units of logarithmic molar solubility.

### Step 8: Visualization and Analysis

Predicted-versus-observed plots provide a convenient visual representation of model 
performance. Perfect predictions lie on the diagonal line

$$
\hat{y}=y.
$$

Residuals provide additional information about prediction errors and are defined as

$$
e_i=y_i-\hat{y}_i.
$$

Residual patterns may reveal systematic model errors, regions of poor predictive performance, 
or unusually difficult molecules.

??? note "Example"

    ```python
    # Predicted vs. observed values
    fig, axes = plt.subplots(1,2,figsize=(14, 6))

    # Training data
    axes[0].scatter(
        y_train,
        y_pred_train,
        alpha=0.5
    )

    train_min = min(
        y_train.min(),
        y_pred_train.min()
    )

    train_max = max(
        y_train.max(),
        y_pred_train.max()
    )

    axes[0].plot(
        [train_min, train_max],
        [train_min, train_max],
        linestyle="--"
    )

    axes[0].set_xlabel("Experimental Solubility")

    axes[0].set_ylabel("Predicted Solubility")

    axes[0].set_title(
        f"Training Set "
        f"(R² = {train_r2:.3f})"
    )

    # Test data
    axes[1].scatter(y_test,y_pred_test,alpha=0.5)

    test_min = min(
        y_test.min(),
        y_pred_test.min()
    )

    test_max = max(
        y_test.max(),
        y_pred_test.max()
    )

    axes[1].plot(
        [test_min, test_max],
        [test_min, test_max],
        linestyle="--"
    )

    axes[1].set_xlabel("Experimental Solubility")

    axes[1].set_ylabel("Predicted Solubility")

    axes[1].set_title(
        f"Test Set "
        f"(R² = {test_r2:.3f})"
    )

    plt.tight_layout()

    plt.savefig("predicted_solubility.png",dpi=300,bbox_inches="tight")

    plt.close()
    ```

#### Permutation Feature Importance

Because several different model families are considered in this workflow, 
**permutation importance** provides a convenient model-independent method for 
estimating which descriptors are important to the final model.

Permutation importance measures the decrease in predictive performance after 
randomly shuffling one feature while leaving the others unchanged.

??? note "Example"

    ```python
    from sklearn.inspection import (
        permutation_importance
    )

    # Calculate permutation importance
    # on the independent test set
    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            result.importances_mean,

        "std":
            result.importances_std

    }).sort_values(
        "importance",
        ascending=False
    )

    print("\nPermutation feature importance:")
    print(importance_df.head(10))

    # Plot feature importance
    top_features = (
        importance_df.head(10)
        .sort_values("importance")
    )

    plt.figure(figsize=(10, 6))

    plt.barh(top_features["feature"],top_features["importance"])

    plt.xlabel("Decrease in Test R²")

    plt.title("Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig("best_feature_importance.png",dpi=300,bbox_inches="tight")

    plt.close()
    ```


Permutation importance describes how strongly the **trained model** relies on each 
descriptor for this particular evaluation dataset. It should not be interpreted as 
evidence that the descriptor has a causal relationship with solubility.

#### Residual Analysis

??? note "Example"

    ```python
    residuals = (
        y_test.to_numpy()
        - y_pred_test
    )

    # Residuals vs. predicted values
    plt.figure(figsize=(8, 6))

    plt.scatter(y_pred_test,residuals,alpha=0.5)

    plt.axhline(y=0,linestyle="--")

    plt.xlabel("Predicted Solubility")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig("residual_plot.png",dpi=300,bbox_inches="tight")

    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 6))

    plt.hist(residuals,bins=30)

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.title("Residual Distribution")

    plt.tight_layout()

    plt.savefig("residual_distribution.png",dpi=300,bbox_inches="tight")

    plt.close()
    ```

For nonlinear machine learning models, residuals are not required to follow a normal 
distribution. The purpose of these plots is therefore primarily diagnostic: they help 
identify systematic bias, heteroscedasticity, large errors, and unusual regions of 
the prediction space.

### Step 9: Model Persistence and Prediction of New Molecules

Once the final model has been selected, it can be saved for later use.
Because preprocessing and prediction are contained inside the same scikit-learn `Pipeline`, 
the complete workflow can be saved as a single object. This is safer than saving the scaler 
and estimator separately because the correct preprocessing is automatically applied 
whenever predictions are made.

??? note "Example"


    ```python
    import joblib
    # Save the complete pipeline
    joblib.dump(best_model,"solubility_model.pkl")

    print("\nModel saved successfully.")

    # Load the model
    loaded_model = joblib.load(
        "solubility_model.pkl"
    )

    # Predict solubility for a new molecule
    new_smiles = "CCO"

    new_features = (
        calculate_molecular_features(
            new_smiles
        )
    )

    if new_features is None:

        raise ValueError(
            "Invalid SMILES string"
        )

    # Preserve the same feature columns
    # used during model training
    new_X = pd.DataFrame([new_features],columns=X.columns)

    prediction = (
        loaded_model.predict(
            new_X
        )
    )

    print(
        f"\nPredicted log solubility "
        f"for {new_smiles}: "
        f"{prediction[0]:.3f}"
    )
    ```


The prediction corresponds to

$$
\widehat{\log_{10}
\left(S_{\mathrm{mol/L}}\right)
}.
$$

If the predicted logarithmic solubility is denoted by $\hat{y}$, the 
corresponding predicted molar solubility can be recovered using

$$
\hat{S}_{\mathrm{mol/L}}=10^{\hat{y}}.
$$

For example, this can be calculated with

```python
predicted_molar_solubility = (10 ** prediction[0])

print("Predicted solubility:",predicted_molar_solubility,"mol/L")
```

This final step demonstrates how a trained QSAR model can be used to 
generate predictions for previously unseen molecular structures.


## 8. Practical Example: QM9 dataset

QM9 is a standard quantum-chemistry benchmark of ~134,000 small organic molecules with
DFT-computed properties. Here we predict the HOMO energy with a random forest.

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
    #   X = molecular features
    #   y = target values (already normalized by the transformer, see note)
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

Two points are worth keeping in mind when interpreting these results:

- **Normalized targets**: `dc.molnet.load_qm9` returns a `NormalizationTransformer` that
  standardizes `y`, so `dataset.y` is expressed in standard-deviation units rather than in
  Hartree. This does not affect $R^2$ (which is invariant under an affine rescaling of the
  target), but the reported MAE and RMSE are in normalized units, not physical energies.
  To report errors in Hartree, undo the transformation on the predictions and true values
  using the returned `transformers` before computing the metrics.

- **Representation limits**: ECFP is a topological fingerprint that encodes connectivity but
  no 3D geometry or electronic structure. HOMO energies depend on both, so a fingerprint-based
  model is inherently limited for this target. Representations that capture geometry (such as
  the Coulomb matrix, or 3D/graph neural network features) are generally far more suitable for
  electronic properties.

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

- RDKit Cookbook  https://www.rdkit.org/docs/Cookbook.html
- Scikit-learn User Guide  https://scikit-learn.org/stable/user_guide.html
- DeepChem Tutorials  https://deepchem.io/
- Daylight Theory Manual (SMILES reference) https://www.daylight.com/dayhtml/doc/theory/index.pdf 
- Gaussian Processes playground https://infallible-thompson-49de36.netlify.app/