import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from Bio.PDB.Polypeptide import is_aa
import csv
from Bio.PDB import PDBParser
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.DSSP import DSSP
from scipy.spatial import ConvexHull
from Bio.PDB import PDBIO
import numpy as np
import os
import tempfile
import pandas as pd


# get lingand features
# Function to calculate molecular features
def calculate_features_ligand(mol):
    """
    Calculate various molecular features for a given molecule.
    """
    features = {}
    features["Molecular Weight"] = Descriptors.MolWt(mol)
    features["LogP"] = Descriptors.MolLogP(mol)
    features["Number of H-bond Donors"] = Descriptors.NumHDonors(mol)
    features["Number of H-bond Acceptors"] = Descriptors.NumHAcceptors(mol)
    features["Topological Polar Surface Area (TPSA)"] = rdMolDescriptors.CalcTPSA(mol)
    features["Number of Rotatable Bonds"] = Descriptors.NumRotatableBonds(mol)
    features["Molar Refractivity"] = Descriptors.MolMR(mol)
    features["Number of Aromatic Rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    return features


# Function to process SDF files in a folder and calculate features for all molecules
def process_sdf_folder(file_path):
    """
    Process all SDF files in a folder, calculate features for each molecule, and save them to a CSV file.
    """
    data = []

    suppl = Chem.SDMolSupplier(file_path, sanitize=False)

    for mol in suppl:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)  # Try sanitizing the molecule
                features = calculate_features_ligand(mol)
                features["Molecule Name"] = mol.GetProp('_Name') if mol.HasProp('_Name') else "Unknown"
                # features["Source File"] = filename
                data.append(features)
            except Exception as e:
                print(e)
    try:
        winer_index = sum((atom.GetDegree() ** 0.5) for atom in mol.GetAtoms())
    except:
        winer_index = 0
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    winer_index = sum((atom.GetDegree() ** 0.5) for atom in mol.GetAtoms())
    df['winer_index'] = winer_index
    return df


# 3d features
# compute protein features
# Hydrophobicity scale (Kyte-Doolittle)
hydrophobicity_scale = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
    'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
    'Y': -1.3, 'V': 4.2
}


# Function to parse PDB files
def parse_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", file_path)
    return structure


# Extract active site residues
def get_active_site_residues(structure):
    residues = []
    for chain in structure[0]:  # Access first model
        for residue in chain:
            residues.append(residue)
    return residues


# Calculate residue count
def calculate_residue_count(active_site_residues):
    return len(active_site_residues)


# Calculate hydrophobicity
def calculate_hydrophobicity(active_site_residues):
    total_hydrophobicity = 0
    for residue in active_site_residues:
        residue_name = residue.get_resname().capitalize()
        one_letter = residue_name[0]  # Approximate 1-letter code from 3-letter code
        total_hydrophobicity += hydrophobicity_scale.get(one_letter, 0)
    return total_hydrophobicity / len(active_site_residues)


# Calculate pocket volume using Convex Hull
def calculate_pocket_volume(active_site_residues):
    coordinates = []
    for residue in active_site_residues:
        for atom in residue:
            coordinates.append(atom.get_coord())
    hull = ConvexHull(np.array(coordinates))
    return hull.volume


# Calculate molecular weight
def calculate_molecular_weight(active_site_residues):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    sequence = []
    for residue in active_site_residues:
        resname = residue.get_resname()
        if resname in three_to_one:
            sequence.append(three_to_one[resname])
    if not sequence:
        raise ValueError("No valid amino acids found in the active site residues.")
    seq_str = ''.join(sequence)
    return molecular_weight(seq_str, seq_type="protein")


# Calculate isoelectric point
def calculate_isoelectric_point(active_site_residues):
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    sequence = []
    for residue in active_site_residues:
        resname = residue.get_resname()
        if resname in three_to_one:
            sequence.append(three_to_one[resname])
    if not sequence:
        raise ValueError("No valid amino acids found in the active site residues.")
    seq_str = ''.join(sequence)
    analysis = ProteinAnalysis(seq_str)
    return analysis.isoelectric_point()


# Calculate secondary structure
def calculate_secondary_structure(pdb_file, model=0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[model]
    dssp = DSSP(model, pdb_file)
    helix = sum(1 for _, res in dssp if res[2] in {'H', 'G', 'I'})  # Helices
    sheet = sum(1 for _, res in dssp if res[2] in {'E', 'B'})  # Sheets
    coil = sum(1 for _, res in dssp if res[2] == 'T')  # Coils
    return {'helix': helix, 'sheet': sheet, 'coil': coil}


# Calculate solubility
def calculate_solubility(active_site_residues):
    hydrophobic_residues = ['A', 'C', 'F', 'I', 'L', 'M', 'P', 'V', 'W', 'Y']
    hydrophobic_count = sum(1 for residue in active_site_residues if residue.get_resname()[0] in hydrophobic_residues)
    total_residues = len(active_site_residues)
    solubility = 1 - (hydrophobic_count / total_residues)  # Simplified solubility score
    return solubility


# Calculate extinction coefficient
def calculate_extinction_coefficient(active_site_residues):
    trp_count = sum(1 for residue in active_site_residues if residue.get_resname() == 'TRP')
    tyr_count = sum(1 for residue in active_site_residues if residue.get_resname() == 'TYR')
    cys_count = sum(1 for residue in active_site_residues if residue.get_resname() == 'CYS')
    extinction_coefficient = (tyr_count * 1490) + (trp_count * 5500) + (cys_count * 125)
    return extinction_coefficient


# Calculate radius of gyration
def calculate_radius_of_gyration(active_site_residues):
    coordinates = []
    for residue in active_site_residues:
        for atom in residue:
            coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates)
    centroid = np.mean(coordinates, axis=0)
    radius_of_gyration = np.sqrt(np.mean(np.sum((coordinates - centroid) ** 2, axis=1)))
    return radius_of_gyration


# Calculate B-factor (average flexibility)
def calculate_b_factor(active_site_residues):
    b_factors = []
    for residue in active_site_residues:
        for atom in residue:
            b_factors.append(atom.get_bfactor())
    return np.mean(b_factors)


# Calculate hydrogen bond network
def calculate_hydrogen_bonds(active_site_residues, distance_cutoff=3.5):
    bonds = []
    for residue in active_site_residues:
        for atom1 in residue:
            for other_residue in active_site_residues:
                if residue != other_residue:
                    for atom2 in other_residue:
                        distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                        if distance <= distance_cutoff:
                            bonds.append((atom1, atom2))
    return len(bonds)


# Calculate solvent-accessible surface area (SASA)
def calculate_sasa(active_site_residues):
    try:
        from freesasa import Structure, calc
    except ImportError:
        raise ImportError("FreeSASA library is required for SASA calculation.")

    # Write residues to a temporary PDB file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_pdb:
        io = PDBIO()
        io.set_structure(active_site_residues[0].get_parent().get_parent())  # Get the structure
        io.save(temp_pdb.name, select=ResidueSelector(active_site_residues))
        temp_pdb_path = temp_pdb.name

    # Use FreeSASA to calculate SASA
    sasa_structure = Structure(temp_pdb_path)
    sasa_result = calc(sasa_structure)

    # Remove temporary file
    os.unlink(temp_pdb_path)

    return sasa_result.totalArea()


class ResidueSelector:
    def __init__(self, residues):
        self.residues = residues

    def accept_model(self, model):
        # Accept all models
        return True

    def accept_chain(self, chain):
        # Accept all chains
        return True

    def accept_residue(self, residue):
        # Accept only the selected residues
        return residue in self.residues

    def accept_atom(self, atom):
        # Accept all atoms in the selected residues
        return True


# Calculate pocket depth
def calculate_pocket_depth(active_site_residues):
    coordinates = []
    for residue in active_site_residues:
        for atom in residue:
            coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates)
    centroid = np.mean(coordinates, axis=0)
    distances = np.linalg.norm(coordinates - centroid, axis=1)
    return np.max(distances) - np.min(distances)


# Process pocket PDB files
def process_pocket_files(file):
    results = []

    if file.endswith('.pdb'):
        structure = parse_pdb(file)
        active_site_residues = get_active_site_residues(structure)

        features = {
            'residue_count': calculate_residue_count(active_site_residues),
            'hydrophobicity': calculate_hydrophobicity(active_site_residues),
            'volume': calculate_pocket_volume(active_site_residues),
            'molecular_weight': calculate_molecular_weight(active_site_residues),
            'isoelectric_point': calculate_isoelectric_point(active_site_residues),
            'solubility': calculate_solubility(active_site_residues),
            'extinction_coefficient': calculate_extinction_coefficient(active_site_residues),
            'radius_of_gyration': calculate_radius_of_gyration(active_site_residues),
            'average_b_factor': calculate_b_factor(active_site_residues),
            'hydrogen_bond_count': calculate_hydrogen_bonds(active_site_residues),
            'sasa': calculate_sasa(active_site_residues),
            'pocket_depth': calculate_pocket_depth(active_site_residues)
        }

        results.append(features)
    return pd.DataFrame(results)


# extract 2-D features
# Mapping for three-letter to one-letter amino acid codes
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}


def extract_sequence_from_pdb(pdb_file):
    """
    Extract the protein sequence from a PDB file.

    Parameters:
        pdb_file (str): Path to the PDB file.

    Returns:
        str: The amino acid sequence of the protein.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):  # Check if the residue is a standard amino acid
                    resname = residue.get_resname()
                    sequence += THREE_TO_ONE.get(resname, "X")  # Use "X" for unknown residues
    return sequence


def calculate_features(protein_sequence):
    """
    Calculate specified features of a protein sequence.

    Parameters:
        protein_sequence (str): The amino acid sequence of the protein.

    Returns:
        dict: A dictionary of calculated features.
    """
    analysis = ProteinAnalysis(protein_sequence)

    # Secondary structure fraction
    secondary_structure_fraction = analysis.secondary_structure_fraction()

    # Hydrogen bond donor and acceptor counts
    h_bond_donor_count = sum(1 for aa in protein_sequence if aa in "STYNQHKR")
    h_bond_acceptor_count = sum(1 for aa in protein_sequence if aa in "DEQNSTY")

    # Emulsification estimate (simplified hydrophobic-polar balance)
    hydrophobic_aa_count = sum(1 for aa in protein_sequence if aa in "AILMFWVY")
    emulsification = hydrophobic_aa_count / len(protein_sequence) if protein_sequence else 0

    return {
        "Hydrogen Bond Donors": h_bond_donor_count,
        "Hydrogen Bond Acceptors": h_bond_acceptor_count,
        "Helices": secondary_structure_fraction[0],
        "Sheets": secondary_structure_fraction[1],
        "Turns": secondary_structure_fraction[2],
        "Emulsification Estimate": emulsification,
    }


def save_features_to_csv(all_features, output_file):
    """
    Save calculated features of all PDB files to a single CSV file.

    Parameters:
        all_features (list): A list of dictionaries containing features for each PDB file.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        header = ["PDB File", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors", "Helices", "Sheets", "Turns",
                  "Emulsification Estimate"]
        writer.writerow(header)

        # Write data for each PDB file
        for pdb_name, features in all_features:
            writer.writerow([
                features["Hydrogen Bond Donors"],
                features["Hydrogen Bond Acceptors"],
                features["Helices"],
                features["Sheets"],
                features["Turns"],
                features["Emulsification Estimate"]
            ])


def get_2d_protein_features(pdb_file):
    all_features = []

    if pdb_file.endswith(".pdb"):
        pdb_name = os.path.splitext(pdb_file)[0]

        print(f"Processing PDB File: {pdb_file}")

        # Extract sequence from PDB
        protein_seq = extract_sequence_from_pdb(pdb_file)
        print(f"Extracted Sequence for {pdb_file}: {protein_seq}")

        # Calculate features
        features = calculate_features(protein_seq)

        # Append features to the list
        all_features.append(features)
    return pd.DataFrame(all_features)


