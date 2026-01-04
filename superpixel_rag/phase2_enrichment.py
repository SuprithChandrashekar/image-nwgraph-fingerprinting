"""
Phase 2: Graph Enrichment.

This module handles feature extraction, edge weight computation, and graph pruning/merging.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial.distance import euclidean
from skimage.measure import regionprops
from skimage.color import rgb2gray

from .config import Phase2Config
from .phase1_segmentation import SegmentationResult
from .utils import get_cache_key, load_from_cache, save_to_cache


@dataclass
class EnrichmentResult:
    """Result of Phase 2 enrichment for a single image."""
    image: np.ndarray
    labels: np.ndarray
    rag: nx.Graph  # Original RAG
    g2: nx.Graph   # Graph with enhanced edge weights
    g_pruned: nx.Graph  # Pruned graph
    node_features: pd.DataFrame
    edge_df: pd.DataFrame
    centroids: Dict[int, Tuple[float, float]]
    title: str = ""
    image_id: str = ""


class Phase2Enrichment:
    """
    Phase 2: Graph Enrichment.
    
    This class handles region feature extraction, enhanced edge weight computation,
    and graph pruning/merging operations.
    """
    
    def __init__(self, config: Optional[Phase2Config] = None, use_cache: bool = False, cache_dir: str = "./cache"):
        """
        Initialize Phase 2 with configuration.
        
        Parameters
        ----------
        config : Phase2Config, optional
            Configuration for enrichment. Uses defaults if not provided.
        use_cache : bool
            Whether to use caching.
        cache_dir : str
            Directory for cache files.
        """
        self.config = config or Phase2Config()
        self.use_cache = use_cache
        self.cache_dir = cache_dir
    
    def extract_region_features(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        rag: nx.Graph
    ) -> pd.DataFrame:
        """
        Extract region properties for each superpixel.
        
        Parameters
        ----------
        labels : np.ndarray
            Superpixel label map.
        image : np.ndarray
            Original image.
        rag : nx.Graph
            Region Adjacency Graph.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with features for each region.
        """
        gray = rgb2gray(image) if image.ndim == 3 else image
        
        # Shift labels to start from 1 for regionprops (it skips 0)
        labels_shifted = labels + 1
        props = regionprops(labels_shifted, intensity_image=gray)
        
        features = []
        for prop in props:
            node_id = prop.label - 1  # Convert back to 0-based indexing
            features.append({
                'node': node_id,
                'area': float(prop.area),
                'perimeter': float(prop.perimeter),
                'eccentricity': float(prop.eccentricity),
                'solidity': float(prop.solidity),
                'extent': float(prop.extent),
                'major_axis_length': float(prop.major_axis_length),
                'minor_axis_length': float(prop.minor_axis_length),
                'centroid_row': float(prop.centroid[0]),
                'centroid_col': float(prop.centroid[1]),
                'bbox': tuple(prop.bbox),
                'mean_intensity_gray': float(prop.mean_intensity)
            })
        
        node_df = pd.DataFrame(features)
        
        # Add mean color from RAG nodes
        mean_colors = {}
        for n, attrs in rag.nodes(data=True):
            mean_colors[n] = attrs.get('mean color', None)
        
        node_df['mean_color'] = node_df['node'].map(mean_colors)
        
        print(f"Extracted features for {len(node_df)} regions")
        return node_df
    
    def compute_boundary_contrast(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        u: int,
        v: int
    ) -> float:
        """
        Compute contrast between two adjacent regions at their boundary.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        u : int
            First region label.
        v : int
            Second region label.
        
        Returns
        -------
        float
            Boundary contrast value.
        """
        gray = rgb2gray(image) if image.ndim == 3 else image
        
        # Find boundary pixels between regions u and v
        mask_u = labels == u
        mask_v = labels == v
        
        # Dilate both masks and find intersection
        struct = generate_binary_structure(2, 1)
        dilated_u = binary_dilation(mask_u, structure=struct)
        dilated_v = binary_dilation(mask_v, structure=struct)
        
        boundary_u = dilated_v & mask_u  # Pixels in u adjacent to v
        boundary_v = dilated_u & mask_v  # Pixels in v adjacent to u
        
        if boundary_u.sum() == 0 or boundary_v.sum() == 0:
            return 0.0
        
        # Compute mean intensity difference at boundary
        mean_u = gray[boundary_u].mean()
        mean_v = gray[boundary_v].mean()
        
        return abs(mean_u - mean_v)
    
    def compute_edge_weights(
        self,
        g: nx.Graph,
        image: np.ndarray,
        labels: np.ndarray,
        node_df: pd.DataFrame
    ) -> nx.Graph:
        """
        Compute enhanced edge weights for the graph.
        
        Parameters
        ----------
        g : nx.Graph
            Input graph.
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        node_df : pd.DataFrame
            Node features DataFrame.
        
        Returns
        -------
        nx.Graph
            Graph with enhanced edge weights.
        """
        g2 = g.copy()
        mode = self.config.edge_weight_mode
        alpha_color = self.config.alpha_color
        alpha_boundary = self.config.alpha_boundary
        
        # Get mean colors from node attributes
        mean_colors = {}
        for n, attrs in g2.nodes(data=True):
            mc = attrs.get('mean color', None)
            if mc is not None:
                mean_colors[n] = np.array(mc)
            else:
                mean_colors[n] = np.array([128, 128, 128])  # Default gray
        
        for u, v, d in g2.edges(data=True):
            # Mean color distance
            if mode in ['mean_color_distance', 'combined']:
                color_dist = euclidean(
                    mean_colors.get(u, [0, 0, 0]),
                    mean_colors.get(v, [0, 0, 0])
                ) / 255.0
            else:
                color_dist = 0.0
            
            # Boundary contrast
            if mode in ['boundary_contrast', 'combined']:
                boundary_contrast = self.compute_boundary_contrast(image, labels, u, v)
            else:
                boundary_contrast = 0.0
            
            # Combine weights
            if mode == 'mean_color_distance':
                weight = color_dist
            elif mode == 'boundary_contrast':
                weight = boundary_contrast
            elif mode == 'combined':
                weight = alpha_color * color_dist + alpha_boundary * boundary_contrast
            else:
                weight = d.get('weight', 1.0)
            
            d['weight2'] = float(weight)
            d['color_dist'] = float(color_dist)
            d['boundary_contrast'] = float(boundary_contrast)
        
        print(f"Edge weights computed using mode: {mode}")
        weights = [d['weight2'] for _, _, d in g2.edges(data=True)]
        if weights:
            print(f"Weight stats: min={min(weights):.4f}, max={max(weights):.4f}, mean={np.mean(weights):.4f}")
        
        return g2
    
    def prune_graph(
        self,
        g: nx.Graph,
        weight_attr: str = 'weight2'
    ) -> nx.Graph:
        """
        Prune edges from the graph based on weight criteria.
        
        Parameters
        ----------
        g : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute to use.
        
        Returns
        -------
        nx.Graph
            Pruned graph.
        """
        g_pruned = g.copy()
        mode = self.config.prune_mode
        value = self.config.prune_value
        
        weights = [d.get(weight_attr, 1.0) for _, _, d in g_pruned.edges(data=True)]
        
        if mode == 'threshold':
            # Remove edges with weight below threshold
            edges_to_remove = [
                (u, v) for u, v, d in g_pruned.edges(data=True)
                if d.get(weight_attr, 1.0) < value
            ]
        
        elif mode == 'percentile':
            # Remove bottom percentile of edges
            threshold = np.percentile(weights, value * 100)
            edges_to_remove = [
                (u, v) for u, v, d in g_pruned.edges(data=True)
                if d.get(weight_attr, 1.0) < threshold
            ]
        
        elif mode == 'topk':
            # Keep only top-k weighted edges per node
            edges_to_remove = []
            for node in g_pruned.nodes():
                neighbors = list(g_pruned.neighbors(node))
                if len(neighbors) <= value:
                    continue
                # Sort neighbors by edge weight
                neighbor_weights = [
                    (n, g_pruned[node][n].get(weight_attr, 1.0))
                    for n in neighbors
                ]
                neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                # Mark edges to remove (beyond top k)
                for n, _ in neighbor_weights[int(value):]:
                    if (node, n) not in edges_to_remove and (n, node) not in edges_to_remove:
                        edges_to_remove.append((node, n))
        else:
            edges_to_remove = []
        
        g_pruned.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        isolates = list(nx.isolates(g_pruned))
        g_pruned.remove_nodes_from(isolates)
        
        print(f"Pruning mode: {mode}, value: {value}")
        print(f"Removed {len(edges_to_remove)} edges, {len(isolates)} isolated nodes")
        print(f"Pruned graph: {g_pruned.number_of_nodes()} nodes, {g_pruned.number_of_edges()} edges")
        
        return g_pruned
    
    def merge_similar_regions(
        self,
        g: nx.Graph,
        weight_attr: str = 'weight2'
    ) -> nx.Graph:
        """
        Merge adjacent regions with very similar colors (low edge weights).
        
        Parameters
        ----------
        g : nx.Graph
            Input graph.
        weight_attr : str
            Edge weight attribute to use.
        
        Returns
        -------
        nx.Graph
            Graph with merged regions.
        """
        g_merged = g.copy()
        threshold = self.config.merge_threshold
        max_merges = self.config.max_merges
        merges_done = 0
        
        while merges_done < max_merges:
            # Find edge with minimum weight
            if g_merged.number_of_edges() == 0:
                break
            
            min_edge = min(
                g_merged.edges(data=True),
                key=lambda x: x[2].get(weight_attr, float('inf'))
            )
            u, v, d = min_edge
            
            if d.get(weight_attr, float('inf')) > threshold:
                break  # No more edges below threshold
            
            # Merge v into u
            for neighbor in list(g_merged.neighbors(v)):
                if neighbor != u:
                    if g_merged.has_edge(u, neighbor):
                        # Average the weights
                        old_w = g_merged[u][neighbor].get(weight_attr, 1.0)
                        new_w = g_merged[v][neighbor].get(weight_attr, 1.0)
                        g_merged[u][neighbor][weight_attr] = (old_w + new_w) / 2
                    else:
                        # Add new edge
                        edge_data = g_merged[v][neighbor].copy()
                        g_merged.add_edge(u, neighbor, **edge_data)
            
            g_merged.remove_node(v)
            merges_done += 1
        
        print(f"Merged {merges_done} region pairs")
        print(f"After merging: {g_merged.number_of_nodes()} nodes, {g_merged.number_of_edges()} edges")
        
        return g_merged
    
    def process(self, seg_result: SegmentationResult) -> EnrichmentResult:
        """
        Process a single segmentation result through Phase 2.
        
        Parameters
        ----------
        seg_result : SegmentationResult
            Result from Phase 1.
        
        Returns
        -------
        EnrichmentResult
            Complete enrichment result.
        """
        print(f"\n{'='*40}")
        print(f"Enriching: {seg_result.title or 'Image'}")
        print('='*40)
        
        # Caching
        cache_key = None
        if self.use_cache and seg_result.image_id:
            params = self.config.__dict__.copy()
            # Add Phase 1 params implicitly via input hash if we wanted, but here we trust input is consistent
            # or we should include Phase 1 hash. For now, just Phase 2 params + image_id.
            cache_key = get_cache_key(seg_result.image_id, params)
            
            cached_node_df = load_from_cache(self.cache_dir, cache_key, 'node_df.parquet')
            cached_edge_df = load_from_cache(self.cache_dir, cache_key, 'edge_df.parquet')
            cached_g2 = load_from_cache(self.cache_dir, cache_key, 'g2.graphml')
            cached_g_pruned = load_from_cache(self.cache_dir, cache_key, 'g_pruned.graphml')
            
            if all(x is not None for x in [cached_node_df, cached_edge_df, cached_g2, cached_g_pruned]):
                print("Loaded Phase 2 results from cache.")
                # Recompute centroids from labels if needed, or cache them.
                # For now, recompute is fast.
                from .utils import compute_centroids
                centroids = compute_centroids(seg_result.labels)
                
                return EnrichmentResult(
                    image=seg_result.image,
                    labels=seg_result.labels,
                    rag=seg_result.rag,
                    g2=cached_g2,
                    g_pruned=cached_g_pruned,
                    node_features=cached_node_df,
                    edge_df=cached_edge_df,
                    centroids=centroids,
                    title=seg_result.title,
                    image_id=seg_result.image_id
                )

        # Extract features
        node_df = self.extract_region_features(
            seg_result.labels,
            seg_result.image,
            seg_result.rag
        )
        
        # Compute enhanced edge weights
        g2 = self.compute_edge_weights(
            seg_result.rag,
            seg_result.image,
            seg_result.labels,
            node_df
        )
        
        # Prune graph
        g_pruned = self.prune_graph(g2)
        
        # Optional merging
        if self.config.enable_merging:
            g_pruned = self.merge_similar_regions(g_pruned)
            
        # Extract edge_df from g2 (contains all edges and weights)
        edge_data = []
        for u, v, d in g2.edges(data=True):
            row = {'source': u, 'target': v}
            row.update(d)
            edge_data.append(row)
        edge_df = pd.DataFrame(edge_data)
        
        # Save to cache
        if self.use_cache and cache_key:
            save_to_cache(self.cache_dir, cache_key, 'node_df.parquet', node_df)
            save_to_cache(self.cache_dir, cache_key, 'edge_df.parquet', edge_df)
            save_to_cache(self.cache_dir, cache_key, 'g2.graphml', g2)
            save_to_cache(self.cache_dir, cache_key, 'g_pruned.graphml', g_pruned)
        
        return EnrichmentResult(
            image=seg_result.image,
            labels=seg_result.labels,
            rag=seg_result.rag,
            g2=g2,
            g_pruned=g_pruned,
            node_features=node_df,
            edge_df=edge_df,
            centroids=seg_result.centroids,
            title=seg_result.title,
            image_id=seg_result.image_id
        )
    
    def process_batch(
        self,
        seg_results: List[SegmentationResult]
    ) -> List[EnrichmentResult]:
        """
        Process multiple segmentation results through Phase 2.
        
        Parameters
        ----------
        seg_results : List[SegmentationResult]
            Results from Phase 1.
        
        Returns
        -------
        List[EnrichmentResult]
            List of enrichment results.
        """
        results = []
        for seg_result in seg_results:
            result = self.process(seg_result)
            results.append(result)
        
        print(f"\n{'='*40}")
        print(f"Phase 2 completed for {len(results)} images")
        print('='*40)
        
        return results
