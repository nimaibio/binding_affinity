# ML-VSPred

**An Interpretable Machine Learning Model for Binding Affinity Prediction and Virtual Screening of Plant-Based Bioactive Compounds**

ML-VSPred is a feature-based machine-learning pipeline, wrapped in a small Flask web application, that predicts protein–ligand binding affinity and ranks plant-derived bioactive compounds for virtual screening. Instead of relying on docking or black-box deep learning, it scores each protein–ligand pair from a compact set of interpretable physicochemical and structural descriptors, so that every prediction can be traced back to chemically meaningful features.

<!--
  TODO before publishing:
  - Confirm author spellings, affiliations, and ORCID iDs.
  - Add the paper DOI / journal / year in the Citation section.
  - Fill in the training-dataset description and performance metrics.
  - Choose and add a LICENSE file.
-->

**Original developer:** Nimai Mahanida
**Contributor:** Satyaranjan Biswal

---

## Overview

The tool answers two related questions:

1. **Single-pair prediction** — given one ligand (SDF) and one protein pocket or structure (PDB), what is the predicted binding affinity?
2. **Virtual screening** — given one protein target, rank a library of plant-based bioactive compounds (the bundled **NatProCP** set) by predicted affinity.

A user supplies the protein file through the web interface and either uploads their own ligand or screens against the pre-computed NatProCP library. The app extracts features, assembles them into the exact layout the models expect, runs the chosen regressor, and returns a downloadable table of affinities.

---

## How the model is built

The design philosophy is *interpretability first*: a small, fixed descriptor set feeding tree-based ensembles, rather than learned embeddings. The pipeline has three stages — feature extraction, feature assembly, and prediction.

### 1. Feature extraction (27 descriptors)

Each protein–ligand pair is represented by **27 features** drawn from three groups.

**Ligand descriptors (9) — RDKit**

| Feature | Description |
|---|---|
| Molecular Weight | Exact molecular weight |
| LogP | Lipophilicity (Crippen) |
| Number of H-bond Donors | Hydrogen-bond donor count |
| Number of H-bond Acceptors | Hydrogen-bond acceptor count |
| Topological Polar Surface Area (TPSA) | Polar surface area |
| Number of Rotatable Bonds | Molecular flexibility |
| Molar Refractivity | Crippen molar refractivity |
| Number of Aromatic Rings | Aromatic ring count |
| `winer_index` | Connectivity index, defined here as the sum of √(atom degree) over all atoms |

**Protein pocket descriptors — 3D structure (12) — BioPython / SciPy / FreeSASA**

| Feature | Description |
|---|---|
| residue_count | Number of residues in the pocket |
| hydrophobicity | Mean Kyte–Doolittle hydrophobicity |
| volume | Pocket volume from a convex hull of atom coordinates |
| molecular_weight | Molecular weight of the pocket sequence |
| isoelectric_point | Theoretical pI (ProtParam) |
| solubility | Hydrophobic/polar balance score |
| extinction_coefficient | From Trp / Tyr / Cys content |
| radius_of_gyration | Compactness of the pocket atoms |
| average_b_factor | Mean atomic B-factor (flexibility) |
| hydrogen_bond_count | Atom pairs within a 3.5 Å cutoff |
| sasa | Solvent-accessible surface area (FreeSASA) |
| pocket_depth | Spread of atoms about the pocket centroid |

**Protein descriptors — 2D sequence (6) — BioPython ProtParam**

| Feature | Description |
|---|---|
| Hydrogen Bond Donors | Donor-type residue count |
| Hydrogen Bond Acceptors | Acceptor-type residue count |
| Helices | Helix fraction |
| Sheets | Sheet fraction |
| Turns | Turn fraction |
| Emulsification Estimate | Hydrophobic-residue fraction |

### 2. Feature assembly

The three feature blocks are concatenated in a fixed order — ligand (9), then protein 3D (12), then protein 2D (6) — to match the column order the models were trained on. In screening mode, the single protein feature row is broadcast across every ligand in the library so that all compounds are scored against the same target.

### 3. Prediction models

Three interpretable regressors are trained and shipped as serialized model files:

| Model | File | Library |
|---|---|---|
| Random Forest | `random_forest_model.sav` | scikit-learn |
| Gradient Boosting | `gradient_boosting_model.sav` | scikit-learn |
| XGBoost | `xgb_model.pkl` | xgboost |

A Keras deep neural network (`keras_dnn_model.h5`) is also included in the repository as an experimental alternative.

The models output a positive pK-style affinity (roughly the 4–9 range, where higher values indicate stronger predicted binding). Because the descriptors are explicit and few, per-feature importances are directly interpretable — molecular weight, molar refractivity, LogP, pocket volume and SASA are typically among the strongest contributors.

<!-- TODO: replace with the actual training dataset, splits, and reported metrics (R^2, RMSE, etc.) from the paper. -->

---

## Repository structure

```
binding_affinity/
├── app.py                       # Flask app: routes, feature assembly, prediction
├── features_functions.py        # Ligand + protein feature extraction
├── dpnp_ligand_features.csv     # Pre-computed NatProCP ligand features (screening library)
├── random_forest_model.sav      # Trained Random Forest
├── gradient_boosting_model.sav  # Trained Gradient Boosting
├── xgb_model.pkl                # Trained XGBoost
├── keras_dnn_model.h5           # Experimental DNN
├── templates/                   # HTML pages (input form, output, about, help)
├── static/                      # Assets + example 1a1e ligand/pocket/protein files
├── requirements.txt
└── runtime.txt                  # Python 3.11.11
```

---

## Installation

Requires **Python 3.11**.

```bash
git clone https://github.com/nimaibio/binding_affinity.git
cd binding_affinity

python3 -m venv env
source env/bin/activate          # Windows: env\Scripts\activate

pip install -r requirements.txt
```

Core dependencies: RDKit, BioPython, FreeSASA, scikit-learn, XGBoost, TensorFlow/Keras, Flask. (`calculate_secondary_structure` additionally relies on the external DSSP binary, but the default pipeline does not require it.)

---

## Usage

### Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser.

For production, serve with gunicorn:

```bash
gunicorn app:app
```

### Workflow

1. Go to the **Virtual Screening** page.
2. Choose whether to screen against the **NatProCP** library (*Yes*) or upload your own ligand (*No*).
3. Upload a protein file — an active-site pocket or a whole protein, in **PDB** format.
4. If not using NatProCP, upload a ligand in **SDF** format.
5. Pick a model: Random Forest, Gradient Boosting, or XGBoost.
6. Submit. The result table lists each compound with its predicted affinity, and a CSV is available to download.

### Input formats

- **Ligand:** `.sdf` (one or many molecules).
- **Protein:** `.pdb` (pocket or full structure).

Example files (`1a1e_ligand.sdf`, `1a1e_pocket.pdb`, `1a1e_protein.pdb`) are bundled in `static/` and downloadable from the app.

### Output

A table and CSV with two columns: compound name / PubChem ID, and predicted binding affinity.

---

## Reproducibility note

Predictions are only valid when inference features are computed with the **same code** used to build the training data. If you modify any descriptor in `features_functions.py`, regenerate `dpnp_ligand_features.csv` and retrain the models so that training and inference stay consistent.

---

## Tech stack

Python 3.11 · Flask · RDKit · BioPython · FreeSASA · SciPy · scikit-learn · XGBoost · TensorFlow / Keras · pandas · NumPy

---

## Citation

If you use ML-VSPred in your work, please cite:

> Mahanida, N., Biswal, S. *ML-VSPred: An Interpretable Machine Learning Model for Binding Affinity Prediction and Virtual Screening of Plant-Based Bioactive Compounds.* <!-- TODO: journal, year, volume, DOI -->

---

## License

<!-- TODO: add a license (e.g. MIT) and a LICENSE file. -->

## Acknowledgments

<!-- TODO: add labs, institutions, and funding sources. -->
