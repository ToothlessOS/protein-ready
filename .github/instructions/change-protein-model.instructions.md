---
applyTo: '**'
---

# Task description

Please change the current data pipeline of the protein model.

The current pipeline requires the pdb files to be preprocessed into graphs with embeddings from ESM-C prior to training (this is currently handled by `tools/build_protein_graph.py`). For the new pipeline, we want to directly use pdb files as input, meaning that the graphs and graph embeddings will be generated on-the-fly during training. Moreover, the graphs created during training should be cached for future use, so that the same pdb file does not need to be processed multiple times.

# Project structure

This is a PyTorch lightning project. With the model related scripts located in `model/` and the data related scripts in `data/`. `model/model_interface.py` and `data/data_interface.py` are the main interfaces for the model and data respectively. The lightning app is defined in `main.py`. Note that there exists a ligand model in this project, which is not relevant to this task.