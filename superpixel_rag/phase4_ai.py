"""
Phase 4: AI Layer

Orchestrates predictive learning, graph reasoning, and explainability.
"""

import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .config import Phase4Config
from .data import GraphBundle
from .ai_features import build_feature_matrices, align_graph_tables, standardize_features
from .ai_reasoning import (
    build_graph_digest, 
    generate_executive_summary, 
    generate_technical_summary,
    llm_reasoning,
    NoOpLLMClient
)
from .ai_gnn import train_gnn
from .ai_export import export_gnn_dataset

@dataclass
class Phase4Result:
    """Results from Phase 4."""
    graph_used_name: str
    n_nodes: int
    n_edges: int
    node_features_df: pd.DataFrame
    edge_features_df: pd.DataFrame
    node_embeddings_df: Optional[pd.DataFrame] = None
    node_predictions_df: Optional[pd.DataFrame] = None
    anomaly_df: Optional[pd.DataFrame] = None
    reasoning_report: Dict[str, Any] = field(default_factory=dict)
    # export_paths removed, handled by ExportManager

class Phase4AI:
    """
    Orchestrator for Phase 4 AI Layer.
    """
    
    def __init__(self, config: Phase4Config):
        self.config = config
        
    def process(self, bundle: GraphBundle) -> GraphBundle:
        """
        Process a GraphBundle through Phase 4.
        
        Parameters
        ----------
        bundle : GraphBundle
            Input bundle from Phase 3.
            
        Returns
        -------
        GraphBundle
            Bundle with Phase 4 results attached.
        """
        # 1. Prepare Input
        # Use the graph selected in Phase 3
        G = bundle.G
        graph_name = bundle.graph_name
        
        # 2. Align Tables
        # Phase 3 might have filtered nodes/edges, but bundle.node_df usually has all.
        # We align to G.
        node_df, edge_df = align_graph_tables(
            G, 
            bundle.node_df, 
            bundle.edge_df, 
            self.config.weight_attr
        )
        
        # 3. Build Feature Matrices
        X_nodes, node_index, node_cols, X_edges, edge_index_pairs, edge_cols = build_feature_matrices(
            node_df, 
            edge_df, 
            feature_set=self.config.feature_set
        )
        
        # Standardize
        X_nodes_scaled, node_scalers = standardize_features(X_nodes)
        
        # 4. Build Reasoning Digest
        # We can pass bundle.centrality_df etc. via a mock analysis_result or just pass dicts
        # build_graph_digest expects analysis_result object with attributes.
        # Let's create a simple object wrapper or modify build_graph_digest.
        # For now, let's create a simple wrapper.
        class MockAnalysisResult:
            def __init__(self, bundle):
                self.centrality = bundle.centrality_df
                # Build node->community mapping from centrality_df which has both 'node' and 'community_id'
                if bundle.centrality_df is not None and 'community_id' in bundle.centrality_df.columns:
                    self.communities = dict(zip(bundle.centrality_df['node'], bundle.centrality_df['community_id']))
                else:
                    self.communities = {}
        
        analysis_wrapper = MockAnalysisResult(bundle)
        
        digest = build_graph_digest(
            G, node_df, edge_df, analysis_wrapper,
            max_nodes=self.config.llm_max_nodes,
            max_edges=self.config.llm_max_edges
        )
        
        exec_summary = generate_executive_summary(digest)
        tech_summary = generate_technical_summary(digest)
        
        reasoning_report = {
            'digest': digest,
            'executive_summary': exec_summary,
            'technical_summary': tech_summary
        }
        
        # LLM Hook
        if self.config.llm_enabled:
            # Placeholder for actual client
            client = NoOpLLMClient()
            llm_res = llm_reasoning(digest, client)
            reasoning_report['llm_analysis'] = llm_res
            
        # 5. Anomaly Scoring (Baseline)
        anomaly_df = self._compute_anomalies(G, node_df, edge_df, analysis_wrapper)
        
        # 6. GNN Training (Optional)
        embeddings_df = None
        preds_df = None
        
        if self.config.train_enabled:
            # Prepare labels
            y = None
            if self.config.task == 'community_prediction' and bundle.community_df is not None:
                # Map community IDs to 0..K
                comm_map = dict(zip(bundle.community_df['node'], bundle.community_df['community_id']))
                if comm_map:
                    # Ensure alignment with node_index
                    y_list = []
                    for i in range(len(node_index)):
                        # Find node_id for this index
                        node_id = [k for k, v in node_index.items() if v == i][0]
                        if node_id in comm_map:
                            y_list.append(comm_map[node_id])
                        else:
                            y_list.append(-1) # Ignore
                    
                    # Remap labels to 0..C
                    unique_labels = sorted(list(set(y_list) - {-1}))
                    label_map = {l: i for i, l in enumerate(unique_labels)}
                    y = np.array([label_map.get(l, -1) for l in y_list])
            
            if y is not None:
                gnn_res = train_gnn(
                    X_nodes_scaled, 
                    edge_index_pairs, 
                    y, 
                    random_state=self.config.random_state
                )
                
                if 'embeddings' in gnn_res:
                    embeddings_df = pd.DataFrame(
                        gnn_res['embeddings'], 
                        index=[k for k, v in sorted(node_index.items(), key=lambda x: x[1])]
                    )
                
                if 'predictions' in gnn_res:
                    preds_df = pd.DataFrame({
                        'predicted': gnn_res['predictions'],
                        'probability': np.max(gnn_res['probabilities'], axis=1)
                    }, index=[k for k, v in sorted(node_index.items(), key=lambda x: x[1])])

        # 7. Export GNN Dataset (Phase 4 specific export, kept here or moved?)
        # We rely on ExportManager to handle exports.
        # We ensure Phase4Result contains enough info.
        
        # Create DataFrames with proper indices
        # node_index maps label -> row_idx. We want row_idx -> label to sort.
        sorted_nodes = sorted(node_index.items(), key=lambda x: x[1])
        node_labels = [k for k, v in sorted_nodes]
        
        node_feat_df = pd.DataFrame(X_nodes, columns=node_cols, index=node_labels)
        
        # For edges, it's harder to index by (u, v) in pandas cleanly for GNN export,
        # but we can store the edge_index_pairs.
        # edge_index_pairs is (2, E).
        
        result = Phase4Result(
            graph_used_name=graph_name,
            n_nodes=len(node_df),
            n_edges=len(edge_df),
            node_features_df=node_feat_df,
            edge_features_df=pd.DataFrame(X_edges, columns=edge_cols), # Edge index not preserved here easily
            node_embeddings_df=embeddings_df,
            node_predictions_df=preds_df,
            anomaly_df=anomaly_df,
            reasoning_report=reasoning_report
        )
        
        # We can attach edge_index to result if we modify Phase4Result definition
        # Or just rely on re-computation.
        # For now, this is sufficient for the bundle.
        
        bundle.phase4_results = result
        return bundle

    def _compute_anomalies(
        self, 
        G: nx.Graph, 
        node_df: pd.DataFrame, 
        edge_df: pd.DataFrame,
        analysis_result: Any
    ) -> pd.DataFrame:
        """
        Compute simple anomaly scores.
        """
        # Node Anomalies
        # Z-score of area
        areas = node_df['area'].values
        z_area = np.abs((areas - np.mean(areas)) / (np.std(areas) + 1e-6))
        
        # Handle both 'label' and 'node' column names
        node_id_col = 'label' if 'label' in node_df.columns else 'node'
        
        # Z-score of betweenness (if available)
        z_betw = np.zeros_like(z_area)
        if analysis_result and hasattr(analysis_result, 'centrality'):
            # Need to align centrality with node_df order
            cent_df = analysis_result.centrality
            # Map values
            betw_vals = []
            for label in node_df[node_id_col]:
                if label in cent_df.index:
                    betw_vals.append(cent_df.loc[label, 'betweenness'])
                else:
                    betw_vals.append(0)
            betw_arr = np.array(betw_vals)
            z_betw = np.abs((betw_arr - np.mean(betw_arr)) / (np.std(betw_arr) + 1e-6))
            
        # Combined score
        anomaly_score = z_area + z_betw
        
        anomaly_df = pd.DataFrame({
            'label': node_df[node_id_col],
            'anomaly_score': anomaly_score,
            'z_area': z_area,
            'z_betweenness': z_betw
        }).sort_values('anomaly_score', ascending=False)
        
        return anomaly_df
