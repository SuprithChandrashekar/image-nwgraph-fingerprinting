"""
AI Reasoning Module

Generates deterministic reasoning reports and provides hooks for LLM integration.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

def build_graph_digest(
    G: nx.Graph, 
    node_df: pd.DataFrame, 
    edge_df: pd.DataFrame, 
    analysis_result: Any = None,
    max_nodes: int = 250, 
    max_edges: int = 800
) -> Dict[str, Any]:
    """
    Build a compact, JSON-serializable digest of the graph state.
    """
    
    # Graph Stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    components = list(nx.connected_components(G))
    n_components = len(components)
    avg_degree = np.mean([d for _, d in G.degree()]) if n_nodes > 0 else 0
    
    # Top Nodes
    top_nodes = {}
    
    # By Degree
    degree_dict = dict(G.degree())
    top_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    top_nodes['by_degree'] = [{'id': n, 'val': d} for n, d in top_degree]
    
    # By Betweenness (if available from Phase 3)
    if analysis_result and hasattr(analysis_result, 'centrality'):
        # Assuming analysis_result.centrality is a DataFrame or dict
        # If it's a DataFrame
        if isinstance(analysis_result.centrality, pd.DataFrame):
            cent_df = analysis_result.centrality.sort_values('betweenness', ascending=False).head(5)
            top_nodes['by_betweenness'] = cent_df[['betweenness']].to_dict(orient='index')
    
    # By Area (if in node_df)
    if 'area' in node_df.columns:
        area_df = node_df.sort_values('area', ascending=False).head(5)
        # Handle both 'label' and 'node' column names
        node_id_col = 'label' if 'label' in area_df.columns else 'node'
        top_nodes['by_area'] = [{'id': r[node_id_col], 'val': r['area']} for _, r in area_df.iterrows()]

    # Community Summary
    community_summary = {}
    if analysis_result and hasattr(analysis_result, 'communities'):
        # Assuming communities is a dict {node: comm_id}
        comms = analysis_result.communities
        if comms:
            comm_counts = pd.Series(comms).value_counts()
            community_summary = {
                'num_communities': len(comm_counts),
                'top_sizes': comm_counts.head(5).to_dict()
            }
            
    # Edge Weight Stats
    weight_stats = {}
    # Try to find a weight column
    weight_col = None
    for col in ['weight2', 'weight']:
        if col in edge_df.columns:
            weight_col = col
            break
            
    if weight_col:
        weights = edge_df[weight_col]
        weight_stats = {
            'min': float(weights.min()),
            'median': float(weights.median()),
            'max': float(weights.max()),
            'mean': float(weights.mean())
        }

    # Sample Subgraph (Top central nodes + 1-hop)
    # We'll take the top node by degree
    subgraph_nodes = []
    subgraph_edges = []
    
    if n_nodes > 0:
        top_node = top_degree[0][0]
        neighbors = list(G.neighbors(top_node))
        # Limit neighbors
        sample_nodes = [top_node] + neighbors[:10]
        
        subgraph_nodes = sample_nodes
        
        # Get edges between these nodes
        sub_G = G.subgraph(sample_nodes)
        for u, v, d in sub_G.edges(data=True):
            w = d.get('weight', 0)
            subgraph_edges.append({'u': u, 'v': v, 'w': w})

    return {
        'stats': {
            'nodes': n_nodes,
            'edges': n_edges,
            'components': n_components,
            'avg_degree': float(avg_degree)
        },
        'top_nodes': top_nodes,
        'community_summary': community_summary,
        'weight_stats': weight_stats,
        'sample_subgraph': {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges
        }
    }


def generate_executive_summary(digest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured executive summary from the digest.
    """
    stats = digest['stats']
    comms = digest['community_summary']
    
    n_nodes = stats['nodes']
    n_comms = comms.get('num_communities', 'N/A')
    
    one_paragraph = (
        f"The image analysis graph contains {n_nodes} superpixel nodes and {stats['edges']} edges. "
        f"The structure exhibits {stats['components']} connected components with an average degree of {stats['avg_degree']:.2f}. "
        f"Community detection identified {n_comms} distinct regions, suggesting significant structural segmentation."
    )
    
    bullets_value = [
        f"Granular segmentation with {n_nodes} regions.",
        f"Identified {n_comms} major structural communities.",
        "Graph connectivity analysis complete."
    ]
    
    bullets_risks = [
        "Potential over-segmentation if node count is very high.",
        "Disconnected components may indicate isolation of regions."
    ]
    
    bullets_next_actions = [
        "Review largest communities for semantic meaning.",
        "Investigate bottleneck edges between communities."
    ]
    
    return {
        'one_paragraph': one_paragraph,
        'bullets_value': bullets_value,
        'bullets_risks': bullets_risks,
        'bullets_next_actions': bullets_next_actions
    }


def generate_technical_summary(digest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured technical summary from the digest.
    """
    return {
        'assumptions': [
            "Nodes represent SLIC superpixels.",
            "Edges represent spatial adjacency.",
            "Weights reflect color/texture similarity."
        ],
        'methods': [
            "SLIC Segmentation",
            "Region Adjacency Graph (RAG) construction",
            "Community Detection (Modularity/Louvain)",
            "Centrality Analysis"
        ],
        'key_findings': [
            f"Graph Density: {digest['stats']['avg_degree'] / (digest['stats']['nodes'] - 1) if digest['stats']['nodes'] > 1 else 0:.4f}",
            f"Weight Distribution: {digest['weight_stats']}"
        ],
        'limitations': [
            "Based on low-level features only.",
            "No semantic understanding without external labels."
        ],
        'repro_steps': [
            "Run Phase 1 segmentation.",
            "Run Phase 2 enrichment.",
            "Run Phase 3 analysis.",
            "Generate Phase 4 report."
        ]
    }


# --- LLM Interface ---

class LLMClientBase:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class NoOpLLMClient(LLMClientBase):
    def generate(self, prompt: str) -> str:
        return "LLM generation disabled or not configured."

def build_llm_prompt(digest: Dict[str, Any]) -> str:
    """
    Produce a compact prompt from the digest.
    """
    import json
    return f"""
    Analyze the following graph data representing an image segmentation:
    {json.dumps(digest, indent=2)}
    
    Provide a reasoning report on the structural properties of the image based on this graph.
    Focus on:
    1. What the communities might represent.
    2. The significance of the top central nodes.
    3. Any anomalies in the structure.
    """

def llm_reasoning(digest: Dict[str, Any], client: LLMClientBase) -> Dict[str, Any]:
    """
    Execute LLM reasoning.
    """
    prompt = build_llm_prompt(digest)
    response = client.generate(prompt)
    
    return {
        'llm_summary': response,
        'prompt_used': prompt
    }
