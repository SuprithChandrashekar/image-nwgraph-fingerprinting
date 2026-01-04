"""
Utility functions for Superpixel RAG Analysis.

Contains shared helper functions for data conversion, graph export, and sanitization.
"""

import json
import hashlib
import pickle
from typing import Tuple, Dict, Any, List, Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


def get_cache_key(identifier: str, params: Dict[str, Any]) -> str:
    """Generate a hash key for caching based on identifier and parameters."""
    # Sort params to ensure consistent hash
    param_str = json.dumps(params, sort_keys=True, default=str)
    content = f"{identifier}_{param_str}"
    return hashlib.md5(content.encode()).hexdigest()


def load_from_cache(cache_dir: str, key: str, artifact_name: str) -> Optional[Any]:
    """Load an artifact from cache."""
    cache_path = Path(cache_dir) / key / artifact_name
    if not cache_path.exists():
        return None
        
    if artifact_name.endswith('.npy'):
        return np.load(cache_path)
    elif artifact_name.endswith('.parquet'):
        return pd.read_parquet(cache_path)
    elif artifact_name.endswith('.pkl'):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    elif artifact_name.endswith('.graphml'):
        return nx.read_graphml(cache_path)
    return None


def save_to_cache(cache_dir: str, key: str, artifact_name: str, data: Any) -> None:
    """Save an artifact to cache."""
    cache_path = Path(cache_dir) / key
    cache_path.mkdir(parents=True, exist_ok=True)
    file_path = cache_path / artifact_name
    
    if artifact_name.endswith('.npy'):
        np.save(file_path, data)
    elif artifact_name.endswith('.parquet'):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(file_path)
    elif artifact_name.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif artifact_name.endswith('.graphml'):
        # Ensure graph is sanitized before saving
        # Make a copy to avoid modifying the original
        graph_copy = data.copy()
        sanitize_attrs_for_graphml(graph_copy)
        nx.write_graphml(graph_copy, file_path)


def to_python_type(val: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, np.ndarray):
        return [to_python_type(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [to_python_type(x) for x in val]
    return val


def sanitize_attrs_for_graphml(G: nx.Graph) -> None:
    """
    Convert list/array attributes to strings for GraphML compatibility.
    
    Modifies the graph in place.
    
    Parameters
    ----------
    G : nx.Graph
        The graph to sanitize.
    """
    def convert_list(lst):
        return [
            int(x) if isinstance(x, np.integer) else 
            float(x) if isinstance(x, np.floating) else x 
            for x in lst
        ]
    
    # Sanitize node attributes
    for n, attrs in G.nodes(data=True):
        for k, v in list(attrs.items()):
            if isinstance(v, (list, tuple, np.ndarray)):
                attrs[k] = json.dumps(convert_list(list(v)))
            elif isinstance(v, (np.integer, np.floating)):
                attrs[k] = to_python_type(v)
    
    # Sanitize edge attributes
    for u, v, attrs in G.edges(data=True):
        for k, v_ in list(attrs.items()):
            if isinstance(v_, (list, tuple, np.ndarray)):
                attrs[k] = json.dumps(convert_list(list(v_)))
            elif isinstance(v_, (np.integer, np.floating)):
                attrs[k] = to_python_type(v_)


def export_graph(
    rag: nx.Graph,
    nodes_csv: str = 'rag_nodes.csv',
    edges_csv: str = 'rag_edges.csv',
    graphml_file: str = 'rag.graphml',
    output_dir: str = '.'
) -> Tuple[str, str, str]:
    """
    Export RAG nodes and edges to CSV and GraphML formats.
    
    Parameters
    ----------
    rag : nx.Graph
        The Region Adjacency Graph to export.
    nodes_csv : str
        Filename for nodes CSV.
    edges_csv : str
        Filename for edges CSV.
    graphml_file : str
        Filename for GraphML export.
    output_dir : str
        Output directory path.
    
    Returns
    -------
    Tuple[str, str, str]
        Paths to the exported nodes CSV, edges CSV, and GraphML files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    nodes_path = output_path / nodes_csv
    edges_path = output_path / edges_csv
    graphml_path = output_path / graphml_file
    
    # Export nodes
    nodes_data = []
    for n, attrs in rag.nodes(data=True):
        row = {'node': n}
        for k, v in attrs.items():
            if isinstance(v, (list, tuple, dict, np.ndarray)):
                row[k] = json.dumps(to_python_type(v))
            elif isinstance(v, (np.integer, np.floating)):
                row[k] = to_python_type(v)
            else:
                row[k] = v
        nodes_data.append(row)
    
    pd.DataFrame(nodes_data).to_csv(nodes_path, index=False)
    print(f"Nodes exported to {nodes_path}")
    
    # Export edges
    edges_data = []
    for u, v, d in rag.edges(data=True):
        row = {'source': u, 'target': v}
        for k, vv in d.items():
            if isinstance(vv, (list, tuple, dict, np.ndarray)):
                row[k] = json.dumps(to_python_type(vv))
            elif isinstance(vv, (np.integer, np.floating)):
                row[k] = to_python_type(vv)
            else:
                row[k] = vv
        edges_data.append(row)
    
    pd.DataFrame(edges_data).to_csv(edges_path, index=False)
    print(f"Edges exported to {edges_path}")
    
    # Export GraphML
    rag_copy = rag.copy()
    sanitize_attrs_for_graphml(rag_copy)
    nx.write_graphml(rag_copy, str(graphml_path))
    print(f"Graph exported to {graphml_path}")
    
    # Print summary
    print(f"Graph stats: nodes={rag.number_of_nodes()}, edges={rag.number_of_edges()}")
    
    return str(nodes_path), str(edges_path), str(graphml_path)


def compute_centroids(labels: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Compute centroids for each labeled region.
    
    Parameters
    ----------
    labels : np.ndarray
        Label map from segmentation.
    
    Returns
    -------
    Dict[int, Tuple[float, float]]
        Dictionary mapping label to (x, y) centroid coordinates.
    """
    centroids = {}
    for region_label in np.unique(labels):
        mask = labels == region_label
        coords = np.column_stack(np.nonzero(mask))
        centroid = coords.mean(axis=0)[::-1]  # (x, y)
        centroids[region_label] = tuple(centroid)
    return centroids


def rag_sanity_checks(rag: nx.Graph) -> Dict[str, Any]:
    """
    Perform sanity checks on a RAG and return statistics.
    
    Parameters
    ----------
    rag : nx.Graph
        The Region Adjacency Graph to check.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing graph statistics.
    """
    degrees = [d for n, d in rag.degree()]
    
    stats = {
        'num_nodes': rag.number_of_nodes(),
        'num_edges': rag.number_of_edges(),
        'degree_min': int(np.min(degrees)) if degrees else 0,
        'degree_max': int(np.max(degrees)) if degrees else 0,
        'degree_mean': float(np.mean(degrees)) if degrees else 0.0,
        'is_connected': nx.is_connected(rag) if rag.number_of_nodes() > 0 else False,
    }
    
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Degree stats: min={stats['degree_min']}, max={stats['degree_max']}, mean={stats['degree_mean']:.2f}")
    print(f"Graph connected: {stats['is_connected']}")
    
    return stats


def select_graph(
    graphs: Dict[str, nx.Graph],
    graph_name: str,
    weight_attr: str = 'weight2',
    remove_isolates: bool = True
) -> Tuple[nx.Graph, nx.Graph, str]:
    """
    Select and sanitize a graph for analysis.
    
    Parameters
    ----------
    graphs : Dict[str, nx.Graph]
        Dictionary of available graphs.
    graph_name : str
        Name of the graph to select.
    weight_attr : str
        Edge weight attribute to use.
    remove_isolates : bool
        Whether to remove isolated nodes.
    
    Returns
    -------
    Tuple[nx.Graph, nx.Graph, str]
        Original graph, working graph (largest connected component), and weight attribute.
    """
    G = graphs.get(graph_name)
    
    if G is None:
        # Fallback chain
        fallbacks = ["g_pruned", "g2", "rag"]
        for fb in fallbacks:
            G = graphs.get(fb)
            if G is not None:
                print(f"'{graph_name}' not found, using '{fb}' instead.")
                break
        if G is None:
            raise ValueError(f"No graph found. Tried: {graph_name}, {fallbacks}")
    
    # Ensure undirected simple graph
    G = nx.Graph(G)
    
    # Remove isolated nodes if requested
    if remove_isolates:
        isolates = list(nx.isolates(G))
        if isolates:
            G.remove_nodes_from(isolates)
            print(f"Removed {len(isolates)} isolated nodes")
    
    # Check weight attribute and fallback if needed
    if G.number_of_edges() > 0:
        sample_edge = next(iter(G.edges(data=True)))
        if weight_attr not in sample_edge[2]:
            for alt in ["weight", "distance"]:
                if alt in sample_edge[2]:
                    print(f"Weight attribute '{weight_attr}' not found, using '{alt}'")
                    weight_attr = alt
                    break
            else:
                print(f"No edge weights found; setting all weights to 1.0")
                nx.set_edge_attributes(G, 1.0, name=weight_attr)
    
    # Add inverse weight for algorithms that need it
    eps = 1e-8
    for u, v, d in G.edges(data=True):
        w = d.get(weight_attr, 1.0)
        d["inv_weight"] = 1.0 / (eps + abs(w))
    
    # Statistics
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    if n_nodes == 0:
        raise ValueError("Graph has no nodes after sanitization!")
    
    n_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"  Connected components: {n_components}")
    print(f"  Largest component size: {len(largest_cc)}")
    
    # Keep only largest connected component for analysis
    G_work = G.subgraph(largest_cc).copy()
    
    return G, G_work, weight_attr
