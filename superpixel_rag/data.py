"""
Data Structures for Superpixel RAG Analysis.

Defines the shared data objects passed between phases.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import networkx as nx

@dataclass
class GraphBundle:
    """
    Unified data object containing all analysis artifacts for a single image.
    Passed from Phase 3 to Phase 4 and used for final export.
    """
    # Metadata
    image_id: str
    title: str
    
    # Raw Data
    image: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    
    # Graph Data (The selected graph for analysis)
    G: nx.Graph
    graph_name: str  # 'rag', 'g2', 'g_pruned'
    
    # Attributes
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    
    # Analysis Results (Phase 3)
    centrality_df: Optional[pd.DataFrame] = None
    community_df: Optional[pd.DataFrame] = None
    path_nodes: Optional[List[int]] = None
    path_edges: Optional[List[Tuple[int, int]]] = None
    mst: Optional[nx.Graph] = None
    bottleneck_edges_df: Optional[pd.DataFrame] = None
    cut_assignment_df: Optional[pd.DataFrame] = None
    
    # AI Results (Phase 4)
    phase4_results: Optional[Any] = None  # Phase4Result object
    
    # Extra
    meta: Dict[str, Any] = field(default_factory=dict)
