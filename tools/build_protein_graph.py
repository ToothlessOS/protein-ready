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
    add_cation_pi_interactions,
    # Additional
    add_distance_threshold, # Used in GearNet paper
    add_sequence_distance_edges, # Used in GearNet paper
    add_k_nn_edges, # Used in GearNet paper
    node_coords # Req'd for EGNN
)
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from functools import partial
from multiprocessing import Pool, set_start_method
import numpy as np

# Configuration for protein graph construction
config = ProteinGraphConfig(
    granularity="centroids", # This is different from GearNet paper which uses "atoms"; Allows sidechain details.
    node_metadata_functions=[amino_acid_one_hot],
    edge_construction_functions=[
        add_peptide_bonds,
        add_aromatic_interactions,
        add_hydrogen_bond_interactions,
        add_disulfide_interactions,
        add_ionic_interactions,
        add_aromatic_sulphur_interactions,
        add_cation_pi_interactions,
        # Additional (GearNet)
        partial(add_distance_threshold, threshold=10, long_interaction_threshold=2),  # Distance < 10 A
        partial(add_sequence_distance_edges, d=1),
        partial(add_sequence_distance_edges, d=2),
        partial(add_sequence_distance_edges, d=3),  # Sequence distance < 3
        partial(add_k_nn_edges, k=10),  # KNN with k=10
    ]
)

# Global variable to hold the client (will be initialized in each worker)
client = None

def initialize_esm_client():
    """Initialize ESM-C client in each worker process."""
    global client
    if client is None:
        from esm.models.esmc import ESMC
        client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
    return client

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

def convert_to_pytorch(graph):
    """
    Convert Graphein protein graph to pytorch tensor objects.
    """
    # Initialize client if not already done
    esm_client = initialize_esm_client()
    
    # Generate ESM embeddings for each chain
    chain_embeddings = {}
    for chain_id, seq in graph.graph["sequences"].items():
        protein = ESMProtein(sequence=str(seq))
        protein_tensor = esm_client.encode(protein)
        logits_output = esm_client.logits(
           protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embeddings_trimmed = logits_output.embeddings[:, 1:-1].squeeze(0)  # Remove batch and special tokens
        chain_embeddings[chain_id] = embeddings_trimmed

    # Add debugging for chain embeddings
    print(f"Available chains in embeddings: {list(chain_embeddings.keys())}")
    for chain_id, embedding in chain_embeddings.items():
        print(f"Chain {chain_id}: {embedding.shape[0]} embeddings")
    
    # Create sequential mapping for each chain by sorting nodes by residue number
    chain_node_mapping = {}
    for node in graph.nodes():
        node_data = graph.nodes[node]
        chain_id = node_data.get('chain_id')
        if chain_id not in chain_node_mapping:
            chain_node_mapping[chain_id] = []
        chain_node_mapping[chain_id].append(node)
    
    # Sort nodes by residue number for each chain to create sequential mapping
    for chain_id in chain_node_mapping:
        chain_node_mapping[chain_id].sort(key=lambda x: graph.nodes[x].get('residue_number', 0))
    
    # First pass: collect valid nodes (those with coordinates AND embeddings)
    valid_nodes = []
    node_features = []
    node_coordinates = []
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        chain_id = node_data.get('chain_id')
        
        # Get coordinates
        coords = node_coords(graph, node)
        
        if coords is not None and len(coords) == 3:
            # Use sequential index instead of PDB residue number
            try:
                sequential_idx = chain_node_mapping[chain_id].index(node)
                
                if chain_id in chain_embeddings and 0 <= sequential_idx < len(chain_embeddings[chain_id]):
                    # Only include nodes that have both valid coordinates and embeddings
                    valid_nodes.append(node)
                    node_coordinates.append(coords)
                    node_features.append(chain_embeddings[chain_id][sequential_idx])
                else:
                    print(f"Debug: Sequential index {sequential_idx} out of bounds for chain {chain_id} (size: {len(chain_embeddings[chain_id]) if chain_id in chain_embeddings else 0})")
            except ValueError:
                print(f"Debug: Node {node} not found in chain mapping for chain {chain_id}")
        
    # Create mapping only for valid nodes
    node_mapping = {node: idx for idx, node in enumerate(valid_nodes)}
    
    if len(node_features) == 0:
        print("Warning: No valid nodes found with both coordinates and embeddings")
        return None
        
    node_features = torch.stack(node_features)
    node_coords_tensor = torch.tensor(node_coordinates, dtype=torch.float32)
    
    print(f"Valid nodes: {len(valid_nodes)}")
    print(f"Final aligned shapes - Features: {node_features.shape}, Coords: {node_coords_tensor.shape}")

    # Extract edge indices and attributes - only for edges between valid nodes
    edge_indices = []
    edge_attrs = []
    
    # Separate physio-chemical properties from distance-based metrics
    physio_chemical_mapping = {
        'peptide_bond': [1, 0, 0, 0, 0, 0, 0],
        'hbond': [0, 1, 0, 0, 0, 0, 0],
        'disulfide': [0, 0, 1, 0, 0, 0, 0],
        'ionic': [0, 0, 0, 1, 0, 0, 0],
        'aromatic': [0, 0, 0, 0, 1, 0, 0],
        'aromatic_sulfur': [0, 0, 0, 0, 0, 1, 0],
        'cation_pi': [0, 0, 0, 0, 0, 0, 1],
    }
    
    distance_metric_mapping = {
        'sequence_edge': [1, 0, 0],
        'knn': [0, 1, 0], 
        'distance_threshold': [0, 0, 1],
    }

    for u, v, edge_data in graph.edges(data=True):
        # Only include edges between valid nodes
        if u in node_mapping and v in node_mapping:
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            edge_indices.append([u_idx, v_idx])
            edge_indices.append([v_idx, u_idx])

            # Handle edge kind - it could be a set or list
            edge_kind = edge_data.get('kind', set())
            if isinstance(edge_kind, set):
                edge_kind = list(edge_kind)
            
            # Initialize edge feature vector [physio_chemical_1-7, distance_metric_1-3, distance]
            physio_chemical_vector = [0, 0, 0, 0, 0, 0, 0]  # Default: no physio-chemical interaction
            distance_metric_vector = [0, 0, 0]  # Default: no distance-based metric
            
            # Categorize edge types - accumulate multiple metrics
            for kind in edge_kind:
                if kind in physio_chemical_mapping:
                    # Add the physio-chemical values instead of replacing
                    for i, val in enumerate(physio_chemical_mapping[kind]):
                        physio_chemical_vector[i] += val
                elif kind in distance_metric_mapping:
                    # Add the distance metric values instead of replacing
                    for i, val in enumerate(distance_metric_mapping[kind]):
                        distance_metric_vector[i] += val
            
            dist = edge_data.get('distance', 0.0)
            
            # Create 11D edge feature: [physio_chemical_1-7, distance_metric_1-3, distance]
            edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
            edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    print(f"Final shapes - Features: {node_features.shape}, Coords: {node_coords_tensor.shape}")
    print(f"Final shapes - Edge Index: {edge_index.shape}, Edge Attr: {edge_attr.shape}")

    # Validation checks
    assert node_features.shape[0] == node_coords_tensor.shape[0], "Node features and coordinates must have same length"
    assert edge_index.max() < len(valid_nodes), "Edge indices must be within valid node range"
    assert edge_attr.shape[0] == edge_index.shape[1], "Edge attributes and edge indices must have same length"

    return {
        'node_features': node_features,
        'edge_index': edge_index, 
        'edge_attr': edge_attr,
        'node_pos': node_coords_tensor,
        'num_nodes': len(node_features)
    }

def process_single_protein(protein_filename, config, save_dir, base_path):
    """
    Process a single protein to generate a native PyTorch graph.
    """
    try:
        # Construct full path to the protein file
        protein_file_path = os.path.join(base_path, protein_filename)
        
        # Check if file exists
        if not os.path.exists(protein_file_path):
            print(f"File not found: {protein_file_path}")
            return
            
        # Extract protein name from filename (remove .pdb extension)
        protein_name = os.path.splitext(protein_filename)[0]
            
        # Extract sequences for all chains
        chain_sequences = get_sequences_from_pdb(protein_file_path)

        # Construct the graph
        graph = construct_graph(config=config, path=protein_file_path)

        # Attach sequences to the graph metadata
        graph.graph["sequences"] = chain_sequences

        # Convert to native PyTorch format
        pytorch_graph = convert_to_pytorch(graph)
        
        # Check if conversion was successful
        if pytorch_graph is None:
            print(f"Failed to convert {protein_name}: No valid nodes found")
            return

        # Save the graph
        save_path = os.path.join(save_dir, f"pytorch_graph_{protein_name}.pt")
        torch.save(pytorch_graph, save_path)
        print(f"Successfully processed {protein_name}")

    except Exception as e:
        print(f"Failed to process {protein_filename}: {e}")

def chunk_list(data, chunk_size):
    """
    Divide a list into chunks of a specified size.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def parallel_process_proteins(proteins, config, save_dir, num_workers=4, chunk_size=25, base_path=None):
    """
    Process multiple proteins in parallel.
    """
    os.makedirs(save_dir, exist_ok=True)
    protein_chunks = list(chunk_list(proteins, chunk_size))

    with Pool(processes=num_workers) as pool:
        pool.map(partial(process_proteins_chunk, config=config, save_dir=save_dir, base_path=base_path), protein_chunks)

def process_proteins_chunk(protein_chunk, config, save_dir, base_path):
    """
    Process a chunk of proteins.
    """
    # Initialize ESM client in this worker process
    initialize_esm_client()
    
    for protein_filename in protein_chunk:
        process_single_protein(protein_filename, config, save_dir, base_path)

if __name__ == "__main__":
    # Set spawn method for CUDA compatibility
    set_start_method('spawn', force=True)
    
    base_path = "../dataset/rcsb/human"
    save_dir = "../dataset/protein_g"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all PDB files from the base path
    proteins = [f for f in os.listdir(base_path) 
                if f.endswith('.pdb') and os.path.isfile(os.path.join(base_path, f))]
    
    print(f"Found {len(proteins)} PDB files to process")
    parallel_process_proteins(proteins, config, save_dir, num_workers=2, chunk_size=20, base_path=base_path)
