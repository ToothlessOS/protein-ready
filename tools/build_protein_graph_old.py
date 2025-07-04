import os
import torch
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_disulfide_interactions,
    add_ionic_interactions,
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions
)
from torch_geometric.data import Data
import esm
from functools import partial
from multiprocessing import Pool
import numpy as np

# Configuration for protein graph construction
config = ProteinGraphConfig(
    granularity="centroids",
    node_metadata_functions=[amino_acid_one_hot],
    edge_construction_functions=[
        add_peptide_bonds,
        add_aromatic_interactions,
        add_hydrogen_bond_interactions,
        add_disulfide_interactions,
        add_ionic_interactions,
        add_aromatic_sulphur_interactions,
        add_cation_pi_interactions,
    ]
)

# Load the ESM model globally for efficiency
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def three_to_one(resname):
    """
    Convert three-letter amino acid code to one-letter code.
    """
    three_to_one_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'SEC': 'U', 'PYL': 'O'
    }
    return three_to_one_dict.get(resname, 'X')  # Return 'X' for unknown residues

def get_sequences_from_pdb(pdb_file):
    """
    Extract sequences for all chains from a PDB file.
    Returns a dictionary with chain IDs as keys and sequences as values.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    chain_sequences = {}

    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                if not residue.has_id("CA"):  # Skip non-standard residues
                    continue
                sequence.append(residue.get_resname())
            
            chain_id = chain.id
            chain_sequences[chain_id] = Seq("".join([three_to_one(res) for res in sequence]))

    return chain_sequences

def convert_to_pyg(graph):
    """
    Convert Graphein protein graph to PyTorch Geometric format.
    """
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}

    # Extract node features using ESM embeddings for each chain
    node_features = []
    for chain_id, seq in graph.graph["sequences"].items():
        sequences = [(f"chain_{chain_id}", str(seq))]
        _, _, batch_tokens = batch_converter(sequences)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        per_residue_embeddings = results["representations"][33]
        node_features.append(per_residue_embeddings)

    node_features = torch.cat(node_features, dim=1).squeeze(0)

    # Extract edge indices and attributes
    edge_indices = []
    edge_attrs = []
    interaction_type_mapping = {
        'peptide_bond': 0,
        'hbond': 1,
        'disulfide': 2,
        'ionic': 3,
        'aromatic': 4,
        'aromatic_sulphur': 5,
        'cation_pi': 6,
    }

    for u, v, edge_data in graph.edges(data=True):
        # Ensure that u and v are mapped to integers
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        edge_indices.append([u_idx, v_idx])
        edge_indices.append([v_idx, u_idx])

        # Ensure edge attributes are the correct format, e.g., a single float or int per edge
        # This assumes interaction_type is a single value; adjust if it's more complex
        interaction_type = interaction_type_mapping[list(edge_data.get('kind'))[0]]
        dist = edge_data.get('distance')
        if isinstance(interaction_type, (list, np.ndarray)):
            interaction_type = interaction_type[0]  # Take first element if it's a list
        edge_attrs.append([interaction_type, dist])  # Ensure it's wrapped in a list
        edge_attrs.append([interaction_type, dist])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def process_single_protein(protein, config, save_dir, path):
    """
    Process a single protein to generate a PyTorch Geometric graph.
    """
    try:
        path = f'test/{protein}/{protein}_protein_processed.pdb'
        
        # Extract sequences for all chains
        chain_sequences = get_sequences_from_pdb(path)

        # Construct the graph
        graph = construct_graph(config=config, path=path)

        # Attach sequences to the graph metadata
        graph.graph["sequences"] = chain_sequences

        # Convert to PyTorch Geometric Data format
        pyg_graph = convert_to_pyg(graph)

        # Save the graph
        save_path = os.path.join(save_dir, f"pyg_graph_{protein}.pt")
        torch.save(pyg_graph, save_path)
        print(f"Successfully processed {protein}")

    except Exception as e:
        print(f"Failed to process {protein}: {e}")

def chunk_list(data, chunk_size):
    """
    Divide a list into chunks of a specified size.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def parallel_process_proteins(proteins, config, save_dir, num_workers=4, chunk_size=100, path=None):
    """
    Process multiple proteins in parallel.
    """
    os.makedirs(save_dir, exist_ok=True)
    protein_chunks = list(chunk_list(proteins, chunk_size))

    with Pool(processes=num_workers) as pool:
        pool.map(partial(process_proteins_chunk, config=config, save_dir=save_dir, path = path), protein_chunks)

def process_proteins_chunk(protein_chunk, config, save_dir, path):
    """
    Process a chunk of proteins.
    """
    for protein in protein_chunk:
        process_single_protein(protein, config, save_dir, path)

path = "../dataset/diff_MOAD"
save_dir = "../protein_g"
os.makedirs(save_dir, exist_ok=True)
with open('../chosen_train', 'r') as f:
    proteins = [line.strip() for line in f]
# proteins = os.listdir(path)
parallel_process_proteins(proteins, config, save_dir, num_workers=4, chunk_size=100, path=path)

