"""
Spectral Clustering using the Fiedler Vector

The Fiedler vector is the eigenvector corresponding to the second smallest 
eigenvalue (algebraic connectivity) of the graph Laplacian matrix.
It provides optimal graph bipartitioning and can be used for spectral clustering.

Key concepts:
- Laplacian matrix L = D - A (degree matrix minus adjacency matrix)
- Second smallest eigenvalue λ₂ (Fiedler value / algebraic connectivity)
- Corresponding eigenvector v₂ (Fiedler vector)
- Sign of Fiedler vector components determines partition membership
"""

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Dict, List, Tuple, Optional
import warnings


def compute_laplacian(G: nx.Graph, normalized: bool = False) -> np.ndarray:
    """
    Compute the Laplacian matrix of a graph.
    
    Args:
        G: NetworkX graph
        normalized: If True, compute normalized Laplacian (L_sym = D^(-1/2) L D^(-1/2))
    
    Returns:
        Laplacian matrix as numpy array
    """
    if normalized:
        return np.array(nx.normalized_laplacian_matrix(G).toarray())
    else:
        return np.array(nx.laplacian_matrix(G).toarray())


def compute_fiedler_vector(G: nx.Graph, normalized: bool = False) -> Tuple[float, np.ndarray, List]:
    """
    Compute the Fiedler vector (second smallest eigenvector of Laplacian).
    
    Args:
        G: NetworkX graph (must be connected)
        normalized: Whether to use normalized Laplacian
    
    Returns:
        Tuple of (fiedler_value, fiedler_vector, node_list)
    """
    if not nx.is_connected(G):
        # Get largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        warnings.warn(f"Graph not connected. Using largest component ({len(G)} nodes).")
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n < 2:
        raise ValueError("Graph must have at least 2 nodes")
    
    # Compute Laplacian
    L = compute_laplacian(G, normalized=normalized)
    
    # For small graphs, use dense eigenvalue decomposition
    if n < 100:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue (should already be sorted, but ensure)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        fiedler_value = eigenvalues[1]
        fiedler_vector = eigenvectors[:, 1]
    else:
        # For larger graphs, use sparse solver for efficiency
        L_sparse = sparse.csr_matrix(L)
        # Get smallest 2 eigenvalues/vectors (first is always 0 for connected graph)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=2, which='SM')
        idx = np.argsort(eigenvalues)
        
        fiedler_value = eigenvalues[idx[1]]
        fiedler_vector = eigenvectors[:, idx[1]]
    
    return fiedler_value, fiedler_vector, nodes


def spectral_bipartition(G: nx.Graph, normalized: bool = False) -> Dict[str, int]:
    """
    Perform spectral bipartitioning using the Fiedler vector.
    Nodes with positive Fiedler values go to partition 1, negative to partition 0.
    
    Args:
        G: NetworkX graph
        normalized: Whether to use normalized Laplacian
    
    Returns:
        Dictionary mapping node -> partition (0 or 1)
    """
    fiedler_value, fiedler_vector, nodes = compute_fiedler_vector(G, normalized)
    
    partition = {}
    for i, node in enumerate(nodes):
        partition[node] = 1 if fiedler_vector[i] >= 0 else 0
    
    return partition


def spectral_clustering(G: nx.Graph, n_clusters: int = 2, normalized: bool = True) -> Dict[str, int]:
    """
    Perform spectral clustering using multiple eigenvectors.
    
    For k clusters, uses the first k eigenvectors (excluding the trivial one)
    and applies k-means clustering in the embedded space.
    
    Args:
        G: NetworkX graph
        n_clusters: Number of clusters to create
        normalized: Whether to use normalized Laplacian
    
    Returns:
        Dictionary mapping node -> cluster_id
    """
    from sklearn.cluster import KMeans
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        warnings.warn(f"Graph not connected. Using largest component ({len(G)} nodes).")
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n_clusters > n:
        n_clusters = n
    
    # Compute Laplacian
    L = compute_laplacian(G, normalized=normalized)
    
    # Compute first k+1 eigenvectors
    if n < 100:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        # Skip first eigenvector (trivial), take next k
        embedding = eigenvectors[:, idx[1:n_clusters+1]]
    else:
        L_sparse = sparse.csr_matrix(L)
        k = min(n_clusters + 1, n - 1)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which='SM')
        idx = np.argsort(eigenvalues)
        embedding = eigenvectors[:, idx[1:]]
    
    # Normalize rows (for normalized cuts)
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # Avoid division by zero
    embedding = embedding / row_norms
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding)
    
    cluster_assignment = {}
    for i, node in enumerate(nodes):
        cluster_assignment[node] = int(labels[i])
    
    return cluster_assignment


def get_fiedler_stats(G: nx.Graph) -> Dict:
    """
    Get comprehensive Fiedler vector statistics for a graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with Fiedler statistics
    """
    try:
        fiedler_value, fiedler_vector, nodes = compute_fiedler_vector(G)
        
        # Bipartition based on sign
        positive_count = np.sum(fiedler_vector >= 0)
        negative_count = np.sum(fiedler_vector < 0)
        
        # Compute cut edges (edges between partitions)
        partition = {nodes[i]: (1 if fiedler_vector[i] >= 0 else 0) for i in range(len(nodes))}
        cut_edges = sum(1 for u, v in G.edges() if partition.get(u, 0) != partition.get(v, 0))
        
        stats = {
            'fiedler_value': float(fiedler_value),
            'algebraic_connectivity': float(fiedler_value),
            'fiedler_vector_mean': float(np.mean(fiedler_vector)),
            'fiedler_vector_std': float(np.std(fiedler_vector)),
            'fiedler_vector_min': float(np.min(fiedler_vector)),
            'fiedler_vector_max': float(np.max(fiedler_vector)),
            'partition_0_size': int(negative_count),
            'partition_1_size': int(positive_count),
            'partition_balance': float(min(positive_count, negative_count) / max(positive_count, negative_count)),
            'cut_edges': int(cut_edges),
            'cut_ratio': float(cut_edges / G.number_of_edges()) if G.number_of_edges() > 0 else 0,
            'nodes': nodes,
            'fiedler_vector': fiedler_vector.tolist()
        }
        
        return stats
        
    except Exception as e:
        return {
            'error': str(e),
            'fiedler_value': None,
            'algebraic_connectivity': None
        }


def compute_spectral_embedding(G: nx.Graph, n_components: int = 3) -> Tuple[np.ndarray, List]:
    """
    Compute spectral embedding of graph nodes.
    
    Args:
        G: NetworkX graph
        n_components: Number of spectral components to compute
    
    Returns:
        Tuple of (embedding_matrix, node_list)
    """
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    L = nx.normalized_laplacian_matrix(G).toarray()
    
    k = min(n_components + 1, n)
    
    if n < 100:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        embedding = eigenvectors[:, idx[1:k]]
    else:
        L_sparse = sparse.csr_matrix(L)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=k, which='SM')
        idx = np.argsort(eigenvalues)
        embedding = eigenvectors[:, idx[1:]]
    
    return embedding, nodes


if __name__ == "__main__":
    # Example usage
    G = nx.karate_club_graph()
    
    print("Fiedler Vector Analysis")
    print("=" * 50)
    
    stats = get_fiedler_stats(G)
    print(f"Algebraic Connectivity (λ₂): {stats['fiedler_value']:.4f}")
    print(f"Partition sizes: {stats['partition_0_size']} vs {stats['partition_1_size']}")
    print(f"Partition balance: {stats['partition_balance']:.2%}")
    print(f"Cut edges: {stats['cut_edges']}")
    print(f"Cut ratio: {stats['cut_ratio']:.2%}")
    
    print("\nSpectral Clustering (k=4)")
    print("=" * 50)
    clusters = spectral_clustering(G, n_clusters=4)
    for i in range(4):
        count = sum(1 for v in clusters.values() if v == i)
        print(f"Cluster {i}: {count} nodes")
