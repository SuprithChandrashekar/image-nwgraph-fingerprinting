"""
AI Export Module

Handles exporting graph datasets for GNN frameworks.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

def export_gnn_dataset(
    output_dir: str,
    X_nodes: np.ndarray,
    edge_index_pairs: np.ndarray,
    X_edges: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    node_ids: Optional[List[int]] = None,
    meta: Optional[Dict[str, Any]] = None
):
    """
    Export dataset in numpy format suitable for PyG/DGL.
    
    Parameters
    ----------
    output_dir : str
        Directory to save files.
    X_nodes : np.ndarray
        Node features.
    edge_index_pairs : np.ndarray
        Edge indices (N, 2).
    X_edges : np.ndarray, optional
        Edge features.
    y : np.ndarray, optional
        Labels.
    node_ids : List[int], optional
        Original node IDs corresponding to rows.
    meta : Dict, optional
        Metadata (feature names, etc).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save Node Features
    np.save(out_path / 'X_nodes.npy', X_nodes)
    
    # Save Edge Index (convert to 2xE for PyG convention)
    # edge_index_pairs is (E, 2), we want (2, E)
    np.save(out_path / 'edge_index.npy', edge_index_pairs.T)
    
    # Save Edge Features
    if X_edges is not None:
        np.save(out_path / 'X_edges.npy', X_edges)
        
    # Save Labels
    if y is not None:
        np.save(out_path / 'y.npy', y)
        
    # Save Node IDs
    if node_ids is not None:
        pd.DataFrame({'node_id': node_ids}).to_csv(out_path / 'node_ids.csv', index=False)
        
    # Save Metadata
    if meta:
        with open(out_path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
            
    print(f"Exported GNN dataset to {out_path}")
