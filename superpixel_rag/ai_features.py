"""
AI Features Module

Handles construction of feature matrices from graph data for AI/ML tasks.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

def align_graph_tables(
    G: nx.Graph, 
    node_df: pd.DataFrame, 
    edge_df: pd.DataFrame, 
    weight_attr: str = 'weight'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures tables only contain nodes/edges present in the chosen graph.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to align with.
    node_df : pd.DataFrame
        DataFrame containing node attributes.
    edge_df : pd.DataFrame
        DataFrame containing edge attributes.
    weight_attr : str
        The edge weight attribute name.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered node_df and edge_df.
    """
    # Filter nodes
    valid_nodes = set(G.nodes())
    # Handle both 'label' and 'node' column names
    node_id_col = 'label' if 'label' in node_df.columns else 'node'
    node_df_aligned = node_df[node_df[node_id_col].isin(valid_nodes)].copy()
    node_df_aligned = node_df_aligned.sort_values(node_id_col).reset_index(drop=True)
    
    # Filter edges
    # Create a set of valid edges (undirected)
    valid_edges = set()
    for u, v in G.edges():
        valid_edges.add(tuple(sorted((u, v))))
        
    # Check which rows in edge_df correspond to edges in G
    # Assuming edge_df has 'source' and 'target' columns
    mask = []
    for _, row in edge_df.iterrows():
        u, v = int(row['source']), int(row['target'])
        edge_key = tuple(sorted((u, v)))
        mask.append(edge_key in valid_edges)
        
    edge_df_aligned = edge_df[mask].copy().reset_index(drop=True)
    
    return node_df_aligned, edge_df_aligned


def build_feature_matrices(
    node_df: pd.DataFrame, 
    edge_df: pd.DataFrame, 
    feature_set: str = "full"
) -> Tuple[np.ndarray, Dict[int, int], List[str], np.ndarray, np.ndarray, List[str]]:
    """
    Build numerical feature matrices for nodes and edges.
    
    Parameters
    ----------
    node_df : pd.DataFrame
        Aligned node dataframe.
    edge_df : pd.DataFrame
        Aligned edge dataframe.
    feature_set : str
        "basic" or "full".
        
    Returns
    -------
    X_nodes : np.ndarray
        Node feature matrix (n_nodes, d_nodes).
    node_index : Dict[int, int]
        Mapping from node_id to row index in X_nodes.
    node_feature_cols : List[str]
        Names of node features.
    X_edges : np.ndarray
        Edge feature matrix (n_edges, d_edges).
    edge_index_pairs : np.ndarray
        Edge connectivity (n_edges, 2) with row indices into X_nodes.
    edge_feature_cols : List[str]
        Names of edge features.
    """
    # --- Node Features ---
    
    # Basic features
    basic_node_cols = [
        'area', 'perimeter', 'mean_intensity', 
        'centroid-0', 'centroid-1'
    ]
    
    # Handle mean_color if it's a list/string or separate columns
    # The previous phases might have saved it as a string representation of a list in CSV
    # or as separate columns if processed. Let's check for r, g, b columns first.
    color_cols = ['r', 'g', 'b']
    if all(c in node_df.columns for c in color_cols):
        basic_node_cols.extend(color_cols)
    elif 'mean_color' in node_df.columns:
        # If it's a single column, we might need to parse it, but for now let's skip complex parsing
        # and assume the enrichment phase expanded it or we rely on what's available.
        # If the user followed the previous phases, 'mean_color' might be a list.
        # We'll try to extract if possible, otherwise skip.
        pass

    # Full features
    full_node_cols = basic_node_cols + [
        'eccentricity', 'solidity', 'extent', 
        'major_axis_length', 'minor_axis_length'
    ]
    
    # Add Phase 3 fields if present
    phase3_cols = ['centrality_degree', 'centrality_betweenness', 'centrality_closeness', 'community']
    
    target_cols = full_node_cols if feature_set == "full" else basic_node_cols
    
    # Filter for columns that actually exist
    available_node_cols = [c for c in target_cols if c in node_df.columns]
    
    if feature_set == "full":
        for c in phase3_cols:
            if c in node_df.columns:
                available_node_cols.append(c)
                
    # Extract node features
    # Ensure numeric types
    X_nodes_df = node_df[available_node_cols].select_dtypes(include=[np.number])
    X_nodes = X_nodes_df.to_numpy(dtype=np.float32)
    node_feature_cols = X_nodes_df.columns.tolist()
    
    # Create node index mapping
    node_id_col = 'label' if 'label' in node_df.columns else 'node'
    node_index = {label: idx for idx, label in enumerate(node_df[node_id_col])}
    
    # --- Edge Features ---
    
    # Basic edge features
    basic_edge_cols = ['weight'] # Usually weight is the primary one
    
    # Full edge features
    full_edge_cols = basic_edge_cols + ['weight2', 'boundary_contrast']
    
    target_edge_cols = full_edge_cols if feature_set == "full" else basic_edge_cols
    available_edge_cols = [c for c in target_edge_cols if c in edge_df.columns]
    
    # Extract edge features
    X_edges_df = edge_df[available_edge_cols].select_dtypes(include=[np.number])
    X_edges = X_edges_df.to_numpy(dtype=np.float32)
    edge_feature_cols = X_edges_df.columns.tolist()
    
    # Build edge_index_pairs (u_idx, v_idx)
    edge_index_list = []
    for _, row in edge_df.iterrows():
        u, v = int(row['source']), int(row['target'])
        if u in node_index and v in node_index:
            edge_index_list.append([node_index[u], node_index[v]])
        else:
            # This shouldn't happen if aligned, but good to be safe
            pass
            
    edge_index_pairs = np.array(edge_index_list, dtype=np.int64)
    
    return X_nodes, node_index, node_feature_cols, X_edges, edge_index_pairs, edge_feature_cols


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
        
    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    scaler_params : Dict[str, Any]
        Mean and scale parameters.
    """
    if X.shape[0] == 0:
        return X, {'mean': [], 'scale': []}
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
