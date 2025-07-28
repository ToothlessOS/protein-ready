"""
PDB Processing Module for On-the-fly Graph Generation

This module provides functionality to process PDB files into graph representations
with ESM-C embeddings, designed for integration into PyTorch datasets.
"""

import os
import torch
import threading
from typing import Dict, Optional, Any
from pathlib import Path
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
    add_distance_threshold,
    add_sequence_distance_edges,
    add_k_nn_edges,
    node_coords
)
from functools import partial


class ESMCSingleton:
    """Thread-safe singleton for ESM-C model management with multiprocessing support."""
    
    _instances = {}  # Process-specific instances
    _locks = {}      # Process-specific locks
    _model = None
    _device = None
    _worker_id = None
    
    def __new__(cls):
        # Get process-specific identifier
        import os
        process_id = os.getpid()
        
        # Create process-specific instance and lock
        if process_id not in cls._instances:
            if process_id not in cls._locks:
                cls._locks[process_id] = threading.Lock()
            
            with cls._locks[process_id]:
                if process_id not in cls._instances:
                    cls._instances[process_id] = super(ESMCSingleton, cls).__new__(cls)
                    # Initialize instance variables
                    instance = cls._instances[process_id]
                    instance._model = None
                    instance._device = None
                    instance._worker_id = None
        
        return cls._instances[process_id]
    
    def __getstate__(self):
        """Custom pickling - exclude locks and model from state."""
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state['_model'] = None  # Don't pickle the model
        state['_device'] = None  # Will be re-determined in worker
        state['_worker_id'] = None  # Will be re-determined in worker
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state and create new lock."""
        self.__dict__.update(state)
        # Model will be loaded lazily when get_model() is called
    
    def get_model(self):
        """Get ESM-C model, loading it if necessary with worker-aware initialization."""
        # Get process-specific lock
        import os
        process_id = os.getpid()
        
        # Ensure process has a lock
        if process_id not in self._locks:
            self._locks[process_id] = threading.Lock()
        
        # Check if we're in a different worker process
        current_worker_id = self._get_worker_id()
        
        if self._model is None or self._worker_id != current_worker_id:
            with self._locks[process_id]:
                if self._model is None or self._worker_id != current_worker_id:
                    self._worker_id = current_worker_id
                    self._load_model()
        return self._model
    
    def _get_worker_id(self):
        """Get current worker ID for multiprocessing awareness."""
        try:
            import torch.utils.data as data
            worker_info = data.get_worker_info()
            return worker_info.id if worker_info is not None else 'main'
        except:
            return 'main'
    
    def _load_model(self):
        """Load ESM-C model with optimal device selection and multiprocessing support."""
        try:
            from esm.models.esmc import ESMC
            import torch
            import os
            
            # Set CUDA device for this worker if available
            worker_id = self._worker_id
            
            # Check if CPU-only mode is forced via environment variable
            force_cpu = os.getenv('ESM_USE_CPU', 'false').lower() == 'true'
            
            # Determine best device based on worker and availability
            if torch.cuda.is_available() and not force_cpu:
                # For multiprocessing, we might want to distribute workers across GPUs
                if worker_id != 'main' and isinstance(worker_id, int):
                    # Distribute workers across available GPUs
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 1:
                        gpu_id = worker_id % num_gpus
                        torch.cuda.set_device(gpu_id)
                        self._device = f"cuda:{gpu_id}"
                        print(f"ESM-C: Worker {worker_id} using GPU {gpu_id}")
                    else:
                        self._device = "cuda"
                        print(f"ESM-C: Worker {worker_id} using GPU (cuda)")
                else:
                    self._device = "cuda"
                    print(f"ESM-C: Main process using GPU (cuda)")
            else:
                self._device = "cpu"
                reason = "forced by ESM_USE_CPU" if force_cpu else "CUDA not available"
                print(f"ESM-C: Worker {worker_id} using CPU ({reason})")
            
            # Load model
            self._model = ESMC.from_pretrained("esmc_300m").to(self._device)
            self._model.eval()  # Set to evaluation mode
            
            print(f"ESM-C: Model loaded successfully on {self._device} (worker: {worker_id})")
            
        except Exception as e:
            print(f"ESM-C: Failed to load model: {e}")
            
            # Fallback to CPU if GPU initialization fails
            try:
                print("ESM-C: Falling back to CPU...")
                self._device = "cpu"
                self._model = ESMC.from_pretrained("esmc_300m").to(self._device)
                self._model.eval()
                print(f"ESM-C: Model loaded successfully on CPU (worker: {self._worker_id})")
            except Exception as e2:
                print(f"ESM-C: CPU fallback also failed: {e2}")
                self._model = None
                self._device = None
                raise
    
    def get_device(self):
        """Get the device the model is on."""
        return self._device
    
    def is_available(self):
        """Check if model is available."""
        return self._model is not None
    
    def reset(self):
        """Reset the singleton (useful for testing or worker cleanup)."""
        import os
        process_id = os.getpid()
        
        # Ensure process has a lock
        if process_id not in self._locks:
            self._locks[process_id] = threading.Lock()
            
        with self._locks[process_id]:
            if self._model is not None:
                try:
                    del self._model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            self._model = None
            self._device = None
            self._worker_id = None


class PDBProcessor:
    """Process PDB files into graph representations with ESM-C embeddings."""
    
    def __init__(self):
        # Configure protein graph construction
        self.config = ProteinGraphConfig(
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
                # Additional (GearNet)
                partial(add_distance_threshold, threshold=10, long_interaction_threshold=2),
                partial(add_sequence_distance_edges, d=1),
                partial(add_sequence_distance_edges, d=2),
                partial(add_sequence_distance_edges, d=3),
                partial(add_k_nn_edges, k=10),
            ]
        )
        
        # ESM-C singleton
        self.esm_singleton = ESMCSingleton()
    
    def __getstate__(self):
        """Custom pickling - ensure ESM singleton is properly handled."""
        state = self.__dict__.copy()
        # ESMCSingleton will handle its own pickling
        return state
    
    def __setstate__(self, state):
        """Custom unpickling - restore state."""
        self.__dict__.update(state)
        # ESM singleton will be recreated as needed
    
    def three_to_one(self, resname: str) -> str:
        """Convert three-letter amino acid code to one-letter code."""
        three_to_one_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            'SEC': 'U', 'PYL': 'O'
        }
        return three_to_one_dict.get(resname, 'X')
    
    def get_sequences_from_pdb(self, pdb_file: str) -> Dict[str, Seq]:
        """Extract sequences for all chains from a PDB file."""
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
                chain_sequences[chain_id] = Seq("".join([self.three_to_one(res) for res in sequence]))
        
        return chain_sequences
    
    def generate_esm_embeddings(self, chain_sequences: Dict[str, Seq]) -> Dict[str, torch.Tensor]:
        """Generate ESM-C embeddings for protein sequences."""
        if not self.esm_singleton.is_available():
            esm_client = self.esm_singleton.get_model()
        else:
            esm_client = self.esm_singleton.get_model()
        
        chain_embeddings = {}
        
        try:
            from esm.sdk.api import ESMProtein, LogitsConfig
            
            for chain_id, seq in chain_sequences.items():
                protein = ESMProtein(sequence=str(seq))
                protein_tensor = esm_client.encode(protein)
                logits_output = esm_client.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings_trimmed = logits_output.embeddings[:, 1:-1].squeeze(0)  # Remove batch and special tokens
                chain_embeddings[chain_id] = embeddings_trimmed.cpu()  # Move to CPU for storage
                
        except Exception as e:
            print(f"Error generating ESM-C embeddings: {e}")
            raise
        
        return chain_embeddings
    
    def convert_to_pytorch(self, graph, chain_sequences: Dict[str, Seq]) -> Optional[Dict[str, torch.Tensor]]:
        """Convert Graphein protein graph to pytorch tensor objects with robust error handling."""
        try:
            # Generate ESM embeddings for each chain
            chain_embeddings = self.generate_esm_embeddings(chain_sequences)
            
            # Attach sequences to the graph metadata
            graph.graph["sequences"] = chain_sequences
            
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
                    # Check for NaN or infinite coordinates
                    import numpy as np
                    if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                        print(f"PDB Processor: Skipping node {node} with invalid coordinates: {coords}")
                        continue
                    
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
            
            # Convert to tensors and validate
            node_features = torch.stack(node_features)
            node_coords_tensor = torch.tensor(node_coordinates, dtype=torch.float32)
            
            # Final validation of tensors
            if torch.any(torch.isnan(node_features)) or torch.any(torch.isinf(node_features)):
                print("PDB Processor: NaN/Inf detected in node features")
                return None
            
            if torch.any(torch.isnan(node_coords_tensor)) or torch.any(torch.isinf(node_coords_tensor)):
                print("PDB Processor: NaN/Inf detected in node coordinates")
                return None
                
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
                    
                    # Validate distance value
                    if np.isnan(dist) or np.isinf(dist):
                        dist = 0.0  # Use default distance for invalid values
                    
                    # Create 11D edge feature: [physio_chemical_1-7, distance_metric_1-3, distance]
                    edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
                    edge_attrs.append(physio_chemical_vector + distance_metric_vector + [dist])
            
            if len(edge_indices) == 0:
                print("Warning: No valid edges found")
                # Create minimal connectivity for isolated nodes
                if len(valid_nodes) > 1:
                    # Create a simple chain connectivity
                    for i in range(len(valid_nodes) - 1):
                        edge_indices.append([i, i + 1])
                        edge_indices.append([i + 1, i])
                        # Simple edge features (all zeros except distance)
                        edge_attrs.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1.0])  # Sequential edge
                        edge_attrs.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1.0])
                else:
                    print("PDB Processor: Single node graph - cannot create edges")
                    return None
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            
            # Final validation of edge tensors
            if torch.any(torch.isnan(edge_attr)) or torch.any(torch.isinf(edge_attr)):
                print("PDB Processor: NaN/Inf detected in edge attributes")
                return None

            print(f"Final shapes - Features: {node_features.shape}, Coords: {node_coords_tensor.shape}")
            print(f"Final shapes - Edge Index: {edge_index.shape}, Edge Attr: {edge_attr.shape}")

            # Validation checks
            assert node_features.shape[0] == node_coords_tensor.shape[0], "Node features and coordinates must have same length"
            assert edge_index.max() < len(valid_nodes), "Edge indices must be within valid node range"
            assert edge_attr.shape[0] == edge_index.shape[1], "Edge attributes and edge indices must have same length"

            result = {
                'node_features': node_features,
                'edge_index': edge_index, 
                'edge_attr': edge_attr,
                'node_pos': node_coords_tensor,
                'num_nodes': len(node_features)
            }
            
            # Final comprehensive validation
            if not self._validate_final_graph(result):
                print("PDB Processor: Final graph validation failed")
                return None
            
            return result
            
        except Exception as e:
            print(f"Error converting graph to PyTorch format: {e}")
            return None
    
    def _validate_final_graph(self, graph_data: Dict[str, torch.Tensor]) -> bool:
        """Perform final validation of graph data."""
        try:
            import torch
            import numpy as np
            
            # Check required keys
            required_keys = ['node_features', 'edge_index', 'node_pos', 'edge_attr']
            if not all(key in graph_data for key in required_keys):
                return False
            
            # Check for NaN/Inf in all tensors
            for key, tensor in graph_data.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                        print(f"PDB Processor: NaN/Inf found in {key}")
                        return False
            
            # Check tensor shapes consistency
            num_nodes = graph_data['node_features'].shape[0]
            if graph_data['node_pos'].shape[0] != num_nodes:
                return False
            
            num_edges = graph_data['edge_index'].shape[1]
            if graph_data['edge_attr'].shape[0] != num_edges:
                return False
            
            # Check edge indices are within valid range
            if graph_data['edge_index'].max() >= num_nodes or graph_data['edge_index'].min() < 0:
                return False
            
            return True
            
        except Exception as e:
            print(f"PDB Processor: Error in final validation: {e}")
            return False
    
    def process_pdb_file(self, pdb_file_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process a single PDB file into a PyTorch graph representation.
        
        Args:
            pdb_file_path: Path to the PDB file
            
        Returns:
            Graph data dictionary or None if processing failed
        """
        try:
            if not os.path.exists(pdb_file_path):
                print(f"PDB file not found: {pdb_file_path}")
                return None
            
            print(f"Processing PDB file: {pdb_file_path}")
            
            # Extract sequences for all chains
            chain_sequences = self.get_sequences_from_pdb(pdb_file_path)
            
            if not chain_sequences:
                print(f"No valid sequences found in {pdb_file_path}")
                return None
            
            # Construct the graph
            graph = construct_graph(config=self.config, path=pdb_file_path)
            
            # Convert to native PyTorch format
            pytorch_graph = self.convert_to_pytorch(graph, chain_sequences)
            
            if pytorch_graph is None:
                print(f"Failed to convert {pdb_file_path}: No valid nodes found")
                return None
            
            print(f"Successfully processed {pdb_file_path}")
            return pytorch_graph
            
        except Exception as e:
            print(f"Failed to process {pdb_file_path}: {e}")
            return None
