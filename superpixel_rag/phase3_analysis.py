"""
Phase 3: Classical Graph Analysis & Structure Extraction.

This module handles centrality computation, community detection, shortest paths,
minimum spanning trees, and min-cut partitioning.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from networkx.algorithms.community import greedy_modularity_communities

from .config import Phase3Config
from .phase2_enrichment import EnrichmentResult
from .data import GraphBundle
from .utils import select_graph, sanitize_attrs_for_graphml

# Try to import Louvain community detection
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


class Phase3Analysis:
    """
    Phase 3: Classical Graph Analysis & Structure Extraction.
    
    This class handles centrality analysis, community detection, shortest path
    computation, MST construction, and min-cut partitioning.
    """
    
    def __init__(self, config: Optional[Phase3Config] = None):
        """
        Initialize Phase 3 with configuration.
        
        Parameters
        ----------
        config : Phase3Config, optional
            Configuration for analysis. Uses defaults if not provided.
        """
        self.config = config or Phase3Config()
    
    def compute_centrality(
        self,
        G: nx.Graph,
        weight_attr: str,
        node_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute centrality measures for all nodes.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute.
        node_features : pd.DataFrame, optional
            Node features to merge with centrality.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with centrality measures for each node.
        """
        print("Computing centrality measures...")
        
        # Compute centralities
        betweenness = nx.betweenness_centrality(G, weight=weight_attr)
        closeness = nx.closeness_centrality(G, distance=weight_attr)
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight=weight_attr)
        except Exception as e:
            print(f"Eigenvector centrality failed: {e}; using degree centrality instead.")
            eigenvector = nx.degree_centrality(G)
        
        degree = dict(G.degree())
        weighted_degree = {
            n: sum(G[n][nbr].get('inv_weight', 1.0) for nbr in G[n])
            for n in G.nodes
        }
        
        # Build DataFrame
        centrality_df = pd.DataFrame({
            'node': list(G.nodes),
            'betweenness': [betweenness[n] for n in G.nodes],
            'closeness': [closeness[n] for n in G.nodes],
            'eigenvector': [eigenvector[n] for n in G.nodes],
            'degree': [degree[n] for n in G.nodes],
            'weighted_degree': [weighted_degree[n] for n in G.nodes],
        })
        
        # Merge with node features if available
        if node_features is not None:
            node_features_copy = node_features.copy()
            node_features_copy['node'] = node_features_copy['node'].astype(int)
            centrality_df['node'] = centrality_df['node'].astype(int)
            centrality_df = centrality_df.merge(node_features_copy, on='node', how='left')
            print("Merged with region features")
        
        # Print top nodes
        print(f"\nTop {self.config.centrality_topn} nodes by betweenness centrality:")
        print(centrality_df.sort_values('betweenness', ascending=False).head(self.config.centrality_topn))
        
        return centrality_df
    
    def detect_communities(
        self,
        G: nx.Graph,
        centrality_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect communities in the graph.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        centrality_df : pd.DataFrame
            Centrality DataFrame to update.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated centrality DataFrame and community summary DataFrame.
        """
        method = self.config.community_method
        
        # Detect communities
        if method == "louvain_if_available" and LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(
                G, weight='inv_weight', random_state=self.config.random_state
            )
            community_ids = [partition[n] for n in G.nodes]
            print("Community detection: Louvain algorithm")
        else:
            comms = list(greedy_modularity_communities(G, weight='inv_weight'))
            node_to_comm = {}
            for i, comm in enumerate(comms):
                for n in comm:
                    node_to_comm[n] = i
            community_ids = [node_to_comm[n] for n in G.nodes]
            print("Community detection: Greedy modularity")
        
        # Add community IDs to centrality_df
        centrality_df = centrality_df.copy()
        centrality_df['community_id'] = community_ids
        
        # Build community summary
        community_df = centrality_df.groupby('community_id').agg(
            size=('node', 'count'),
            mean_betweenness=('betweenness', 'mean'),
            mean_degree=('degree', 'mean')
        ).reset_index()
        
        # Add area stats if available
        if 'area' in centrality_df.columns:
            area_stats = centrality_df.groupby('community_id')['area'].agg(['mean', 'sum']).reset_index()
            area_stats.columns = ['community_id', 'mean_area', 'total_area']
            community_df = community_df.merge(area_stats, on='community_id')
        
        print(f"\nDetected {len(community_df)} communities")
        
        return centrality_df, community_df
    
    def compute_shortest_path(
        self,
        G: nx.Graph,
        weight_attr: str,
        centrality_df: pd.DataFrame
    ) -> Tuple[List[int], List[Tuple[int, int]], float, pd.DataFrame, pd.DataFrame]:
        """
        Compute shortest path between two nodes.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute.
        centrality_df : pd.DataFrame
            Centrality DataFrame with node information.
        
        Returns
        -------
        Tuple
            Path nodes, path edges, path length, path nodes DataFrame, path edges DataFrame.
        """
        print("Computing shortest paths...")
        
        # Choose endpoints
        node_a = self.config.path_node_a
        node_b = self.config.path_node_b
        
        if node_a is None or node_b is None:
            # Auto-select: pick two nodes with max Euclidean distance
            if 'centroid_row' in centrality_df.columns and 'centroid_col' in centrality_df.columns:
                coords = centrality_df[['node', 'centroid_row', 'centroid_col']].dropna()
                if len(coords) > 500:
                    coords = coords.sample(500, random_state=self.config.random_state)
                
                dist_matrix = cdist(
                    coords[['centroid_row', 'centroid_col']],
                    coords[['centroid_row', 'centroid_col']]
                )
                i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                node_a = int(coords.iloc[i]['node'])
                node_b = int(coords.iloc[j]['node'])
                print(f"Auto-selected endpoints (max distance): {node_a} -> {node_b}")
            else:
                nodes_list = list(G.nodes())
                node_a, node_b = nodes_list[0], nodes_list[-1]
                print(f"Fallback endpoints: {node_a} -> {node_b}")
        else:
            print(f"Using specified endpoints: {node_a} -> {node_b}")
        
        # Compute shortest path
        try:
            path_nodes = nx.shortest_path(G, source=node_a, target=node_b, weight=weight_attr)
            path_length = nx.shortest_path_length(G, source=node_a, target=node_b, weight=weight_attr)
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            
            print(f"\nShortest path: {len(path_nodes)} nodes, total weight: {path_length:.4f}")
            
            path_edges_df = pd.DataFrame(path_edges, columns=['source', 'target'])
            path_nodes_df = centrality_df[centrality_df['node'].isin(path_nodes)]
            
        except nx.NetworkXNoPath:
            print(f"No path exists between {node_a} and {node_b}")
            path_nodes, path_edges = [], []
            path_length = 0.0
            path_edges_df = pd.DataFrame(columns=['source', 'target'])
            path_nodes_df = pd.DataFrame()
        
        return path_nodes, path_edges, path_length, path_nodes_df, path_edges_df
    
    def compute_bottleneck_edges(
        self,
        G: nx.Graph,
        weight_attr: str,
        top_n: int = 20
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Compute edge betweenness centrality to find bottleneck edges.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute.
        top_n : int
            Number of top bottleneck edges to return.
        
        Returns
        -------
        Tuple[Dict, pd.DataFrame]
            Edge betweenness dictionary and DataFrame of top bottleneck edges.
        """
        print("\nComputing edge betweenness centrality...")
        
        edge_betweenness = nx.edge_betweenness_centrality(G, weight=weight_attr)
        bottleneck_edges = sorted(edge_betweenness.items(), key=lambda x: -x[1])[:top_n]
        
        bottleneck_edges_df = pd.DataFrame(
            [(u, v, b) for (u, v), b in bottleneck_edges],
            columns=['u', 'v', 'edge_betweenness']
        )
        
        print(f"\nTop {min(10, top_n)} bottleneck edges:")
        print(bottleneck_edges_df.head(10))
        
        return edge_betweenness, bottleneck_edges_df
    
    def compute_mst(
        self,
        G: nx.Graph,
        weight_attr: str
    ) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Compute Minimum Spanning Tree.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute.
        
        Returns
        -------
        Tuple[nx.Graph, pd.DataFrame]
            MST graph and DataFrame of MST edges.
        """
        print("Computing Minimum Spanning Tree...")
        
        mst = nx.minimum_spanning_tree(G, weight=weight_attr)
        mst_edges = list(mst.edges())
        mst_edges_df = pd.DataFrame(mst_edges, columns=['u', 'v'])
        
        total_mst_weight = mst.size(weight=weight_attr)
        print(f"\nMST Statistics:")
        print(f"  Total MST weight: {total_mst_weight:.4f}")
        print(f"  MST edges: {len(mst_edges)}")
        
        try:
            mst_diameter = nx.diameter(mst)
            print(f"  MST diameter (unweighted): {mst_diameter}")
        except Exception:
            print("  MST diameter: Could not compute")
        
        return mst, mst_edges_df
    
    def compute_mincut(
        self,
        G: nx.Graph,
        weight_attr: str
    ) -> Optional[pd.DataFrame]:
        """
        Compute min-cut partition if seeds are provided.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute.
        
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with cut assignments, or None if no seeds provided.
        """
        source_seeds = self.config.mincut_source_seeds
        sink_seeds = self.config.mincut_sink_seeds
        
        print("Min-Cut Segmentation Analysis")
        print("=" * 40)
        
        if not source_seeds or not sink_seeds:
            print("No min-cut seeds provided.")
            print("To enable min-cut, set mincut_source_seeds and mincut_sink_seeds.")
            return None
        
        print(f"Source seeds: {source_seeds}")
        print(f"Sink seeds: {sink_seeds}")
        
        # Build directed flow network
        flowG = nx.DiGraph()
        eps = 1e-8
        
        for u, v, d in G.edges(data=True):
            w = d.get(weight_attr, 1.0)
            cap = 1.0 / (eps + w)
            flowG.add_edge(u, v, capacity=cap)
            flowG.add_edge(v, u, capacity=cap)
        
        # Add super source (S) and super sink (T)
        S, T = 'S', 'T'
        for n in source_seeds:
            if n in G.nodes():
                flowG.add_edge(S, n, capacity=1e9)
        for n in sink_seeds:
            if n in G.nodes():
                flowG.add_edge(n, T, capacity=1e9)
        
        try:
            cut_value, (set0, set1) = nx.minimum_cut(flowG, S, T, capacity='capacity')
            print(f"\nMin-cut value: {cut_value:.4f}")
            
            # Remove super source/sink from sets
            set0.discard(S)
            set1.discard(T)
            
            cut_assignment = {n: 0 for n in set0}
            cut_assignment.update({n: 1 for n in set1})
            cut_assignment_df = pd.DataFrame(
                list(cut_assignment.items()),
                columns=['node', 'cut_side']
            )
            
            print(f"\nPartition sizes:")
            print(cut_assignment_df['cut_side'].value_counts())
            
            return cut_assignment_df
            
        except Exception as e:
            print(f"Min-cut computation failed: {e}")
            return None
    
    def process(
        self,
        enrichment_result: EnrichmentResult,
        graph_source: Optional[str] = None
    ) -> GraphBundle:
        """
        Process a single enrichment result through Phase 3.
        
        Parameters
        ----------
        enrichment_result : EnrichmentResult
            Result from Phase 2.
        graph_source : str, optional
            Which graph to use ('rag', 'g2', 'g_pruned').
        
        Returns
        -------
        GraphBundle
            Complete analysis result bundle.
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {enrichment_result.title or 'Image'}")
        print('='*60)
        
        # Select graph
        graph_source = graph_source or self.config.graph_source
        graphs = {
            'rag': enrichment_result.rag,
            'g2': enrichment_result.g2,
            'g_pruned': enrichment_result.g_pruned
        }
        
        G, G_work, weight_attr = select_graph(
            graphs,
            graph_source,
            self.config.weight_attr
        )
        
        print(f"\nUsing weight attribute: '{weight_attr}'")
        
        # Compute centrality
        centrality_df = self.compute_centrality(
            G_work,
            weight_attr,
            enrichment_result.node_features
        )
        
        # Detect communities
        centrality_df, community_df = self.detect_communities(G_work, centrality_df)
        
        # Compute shortest path
        path_nodes, path_edges, path_length, path_nodes_df, path_edges_df = \
            self.compute_shortest_path(G_work, weight_attr, centrality_df)
        
        # Compute bottleneck edges
        edge_betweenness, bottleneck_edges_df = self.compute_bottleneck_edges(G_work, weight_attr)
        
        # Compute MST
        mst, mst_edges_df = self.compute_mst(G_work, weight_attr)
        
        # Compute min-cut
        cut_assignment_df = self.compute_mincut(G_work, weight_attr)
        
        # Convert centroids to array
        max_label = max(enrichment_result.centroids.keys()) if enrichment_result.centroids else 0
        centroids_arr = np.zeros((max_label + 1, 2))
        for k, v in enrichment_result.centroids.items():
            centroids_arr[k] = v
            
        return GraphBundle(
            image_id=enrichment_result.image_id,
            title=enrichment_result.title,
            image=enrichment_result.image,
            labels=enrichment_result.labels,
            centroids=centroids_arr,
            G=G_work, # Use the working graph (LCC) or G? Let's use G_work as it has the analysis.
            graph_name=graph_source,
            node_df=enrichment_result.node_features,
            edge_df=enrichment_result.edge_df,
            centrality_df=centrality_df,
            community_df=community_df,
            path_nodes=path_nodes,
            path_edges=path_edges,
            mst=mst,
            bottleneck_edges_df=bottleneck_edges_df,
            cut_assignment_df=cut_assignment_df,
            meta={
                'weight_attr': weight_attr,
                'path_length': path_length,
                'edge_betweenness': edge_betweenness
            }
        )
    
    def export_results(
        self,
        bundle: GraphBundle,
        output_dir: str = '.',
        prefix: str = ''
    ) -> Dict[str, str]:
        """
        Export Phase 3 results to files.
        
        Parameters
        ----------
        bundle : GraphBundle
            GraphBundle containing analysis results.
        output_dir : str
            Output directory.
        prefix : str
            Prefix for output files.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping file types to file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Exporting Phase 3 results...")
        print("=" * 40)
        
        exported = {}
        p = f"{prefix}_" if prefix else ""
        
        # Export centrality data
        if bundle.centrality_df is not None:
            path = output_path / f'{p}centrality.csv'
            bundle.centrality_df.to_csv(path, index=False)
            exported['centrality'] = str(path)
            print(f"✓ {path.name}")
        
        # Export community assignments
        if bundle.community_df is not None:
            path = output_path / f'{p}communities.csv'
            bundle.community_df.to_csv(path, index=False)
            exported['communities'] = str(path)
            print(f"✓ {path.name}")
        
        # Export bottleneck edges
        if bundle.bottleneck_edges_df is not None:
            path = output_path / f'{p}bottleneck_edges.csv'
            bundle.bottleneck_edges_df.to_csv(path, index=False)
            exported['bottleneck_edges'] = str(path)
            print(f"✓ {path.name}")
        
        # Export MST
        if bundle.mst is not None:
            path = output_path / f'{p}mst.graphml'
            mst_copy = bundle.mst.copy()
            sanitize_attrs_for_graphml(mst_copy)
            nx.write_graphml(mst_copy, str(path))
            exported['mst'] = str(path)
            print(f"✓ {path.name}")
        
        # Export min-cut assignment
        if bundle.cut_assignment_df is not None:
            path = output_path / f'{p}mincut_assignment.csv'
            bundle.cut_assignment_df.to_csv(path, index=False)
            exported['mincut'] = str(path)
            print(f"✓ {path.name}")
        
        # Create enriched GraphML
        print("\nCreating enriched GraphML...")
        graphml_g = bundle.G.copy()
        
        # Add node attributes from centrality_df
        if bundle.centrality_df is not None:
            for _, row in bundle.centrality_df.iterrows():
                n = row.get('node') or row.get('label')
                if n in graphml_g.nodes:
                    for attr in ['betweenness', 'closeness', 'eigenvector', 'degree', 'weighted_degree', 'community_id']:
                        if attr in row.index and pd.notna(row[attr]):
                            graphml_g.nodes[n][attr] = float(row[attr]) if attr != 'community_id' else int(row[attr])
        
        sanitize_attrs_for_graphml(graphml_g)
        path = output_path / f'{p}rag_phase3.graphml'
        nx.write_graphml(graphml_g, str(path))
        exported['graphml'] = str(path)
        print(f"✓ {path.name}")
        
        print("\n" + "=" * 40)
        print("Phase 3 exports complete!")
        print(f"Output directory: {output_path}")
        
        return exported
