"""
Test script for the new hybrid protein dataset implementation.
Run this to validate that the on-the-fly PDB processing works correctly.
"""

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.protein_dataset import ProteinDataset
from data.cache_manager import CacheManager
import torch

def test_dataset_modes():
    """Test different dataset initialization modes."""
    
    print("="*80)
    print("Testing Hybrid Protein Dataset Implementation")
    print("="*80)
    
    # Test 1: Check if dataset can auto-detect data source
    print("\n1. Testing auto-detection with default paths...")
    try:
        dataset = ProteinDataset(
            train=True,
            data_path="dataset/protein_g/",  # Preprocessed files
            pdb_path="dataset/rcsb/human/",  # PDB files
            cache_path="dataset/protein_cache_test/",
            enable_pdb_processing=True
        )
        print(f"   ✓ Dataset initialized successfully")
        print(f"   ✓ Data source: {dataset.data_source_type}")
        print(f"   ✓ Using PDB files: {dataset.use_pdb_files}")
        print(f"   ✓ Found {len(dataset)} samples")
        
        # Get cache info
        cache_info = dataset.get_cache_info()
        print(f"   ✓ Cache info: {cache_info}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Force PDB processing with error handling
    print("\n2. Testing forced PDB processing with error handling...")
    try:
        dataset_pdb = ProteinDataset(
            train=True,
            pdb_path="dataset/rcsb/human/",
            cache_path="dataset/protein_cache_test/",
            force_pdb_processing=True,
            enable_cache=True,
            skip_invalid_files=True,
            validate_on_init=False  # Set to True to pre-validate files (slower but safer)
        )
        print(f"   ✓ PDB-forced dataset initialized successfully")
        print(f"   ✓ Data source: {dataset_pdb.data_source_type}")
        print(f"   ✓ Found {len(dataset_pdb)} PDB files")
        
        if len(dataset_pdb) > 0:
            print(f"   ✓ Testing sample loading...")
            sample = dataset_pdb[0]
            
            if sample['load_error']:
                print(f"   ⚠ Sample loading failed: {sample}")
            else:
                print(f"   ✓ Sample loaded successfully")
                print(f"   ✓ Protein ID: {sample['protein_id']}")
                print(f"   ✓ Graph shape: {sample['protein_graph']['node_features'].shape}")
                print(f"   ✓ Views created: {sample['view1'] is not None}, {sample['view2'] is not None}")
                print(f"   ✓ Attempts needed: {sample.get('attempts_needed', 'N/A')}")
                
                # Check processing stats
                stats = dataset_pdb.get_processing_stats()
                print(f"   ✓ Processing stats: {stats}")
                
                # Test error handling by trying to load a few more samples
                print(f"   ✓ Testing multiple samples for error handling...")
                valid_samples = 0
                failed_samples = 0
                
                for i in range(min(5, len(dataset_pdb))):
                    try:
                        sample = dataset_pdb[i]
                        if sample['load_error']:
                            failed_samples += 1
                        else:
                            valid_samples += 1
                    except Exception as e:
                        print(f"      ⚠ Exception loading sample {i}: {e}")
                        failed_samples += 1
                
                print(f"   ✓ Sample loading results: {valid_samples} valid, {failed_samples} failed")
                
                # Show failed files if any
                failed_files = dataset_pdb.get_failed_files()
                if failed_files:
                    print(f"   ⚠ Failed files: {[Path(f).name for f in failed_files[:5]]}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Cache manager functionality
    print("\n3. Testing cache manager...")
    try:
        cache_manager = CacheManager("dataset/protein_cache_test/")
        cache_stats = cache_manager.get_cache_stats()
        print(f"   ✓ Cache manager initialized")
        print(f"   ✓ Cache stats: {cache_stats}")
        
        # Test cache validation
        validation_results = cache_manager.validate_cache()
        print(f"   ✓ Cache validation: {validation_results}")
        
    except Exception as e:
        print(f"   ✗ Cache manager error: {e}")
    
    # Test 4: Test specific problematic file (3ESP.pdb)
    print("\n4. Testing specific problematic file handling...")
    try:
        from data.pdb_processor import PDBProcessor
        
        problematic_file = "dataset/rcsb/human/3ESP.pdb"
        if os.path.exists(problematic_file):
            print(f"   ✓ Found problematic file: {problematic_file}")
            
            processor = PDBProcessor()
            result = processor.process_pdb_file(problematic_file)
            
            if result is None:
                print(f"   ✓ Correctly handled problematic file (returned None)")
            else:
                print(f"   ⚠ Unexpectedly succeeded processing problematic file")
                
            # Test with dataset
            dataset_test = ProteinDataset(
                train=True,
                pdb_path="dataset/rcsb/human/",
                cache_path="dataset/protein_cache_test/",
                force_pdb_processing=True,
                skip_invalid_files=True
            )
            
            # Find the index of the problematic file
            problem_idx = None
            for i, file_path in enumerate(dataset_test.file_paths):
                if "3ESP.pdb" in file_path:
                    problem_idx = i
                    break
            
            if problem_idx is not None:
                print(f"   ✓ Found 3ESP.pdb at index {problem_idx}")
                sample = dataset_test[problem_idx]
                
                if sample['load_error']:
                    print(f"   ✓ Dataset correctly handled problematic file (load_error=True)")
                else:
                    print(f"   ⚠ Dataset didn't mark problematic file as error")
            else:
                print(f"   ⚠ Could not find 3ESP.pdb in dataset file list")
        else:
            print(f"   ⚠ Problematic file not found: {problematic_file}")
            
    except Exception as e:
        print(f"   ✗ Error testing problematic file: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

if __name__ == "__main__":
    test_dataset_modes()
