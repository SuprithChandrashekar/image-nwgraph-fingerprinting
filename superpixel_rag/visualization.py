"""
Visualization module for Superpixel RAG Analysis.

Provides comprehensive visualization functions for all phases.
"""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from skimage import segmentation, color


class Visualizer:
    """
    Visualization utilities for Superpixel RAG Analysis.
    
    Provides methods to visualize segmentation results, graph overlays,
    centrality measures, communities, paths, and other analysis results.
    """
    
    def __init__(
        self,
        show_plots: bool = True,
        save_plots: bool = False,
        output_dir: str = '.',
        dpi: int = 100
    ):
        """
        Initialize the Visualizer.
        
        Parameters
        ----------
        show_plots : bool
            Whether to display plots.
        save_plots : bool
            Whether to save plots to files.
        output_dir : str
            Directory to save plots.
        dpi : int
            DPI for saved figures.
        """
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        
        if save_plots:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _finalize_plot(self, fig: plt.Figure, name: str):
        """Handle saving and showing of a plot."""
        if self.save_plots:
            path = self.output_dir / f"{name}.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {path}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_superpixels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        title: str = "SLIC Superpixels",
        save_name: Optional[str] = None
    ):
        """
        Display image with superpixel average colors.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(color.label2rgb(labels, image, kind='avg'))
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'superpixels')

    def plot_anomalies(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        anomaly_df: pd.DataFrame,
        centroids: np.ndarray,
        top_n: int = 30,
        title: str = "Top Anomaly Nodes",
        save_name: Optional[str] = None
    ):
        """
        Highlight top anomaly nodes on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel labels.
        anomaly_df : pd.DataFrame
            DataFrame with 'label' and 'anomaly_score'.
        centroids : np.ndarray
            Centroids of superpixels.
        top_n : int
            Number of top anomalies to highlight.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show boundaries
        ax.imshow(segmentation.mark_boundaries(image, labels))
        
        # Get top N anomalies
        top_anomalies = anomaly_df.head(top_n)
        
        # Plot markers
        for _, row in top_anomalies.iterrows():
            label = int(row['label'])
            if label < len(centroids):
                cy, cx = centroids[label]
                ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2)
                ax.text(cx, cy, f"{label}", color='yellow', fontsize=8)
                
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'anomalies')
    
    def plot_boundaries(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        title: str = "Superpixel Boundaries",
        save_name: Optional[str] = None
    ):
        """
        Display image with superpixel boundaries overlaid.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(segmentation.mark_boundaries(image, labels))
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'boundaries')
    
    def plot_rag_overlay(
        self,
        image: np.ndarray,
        rag: nx.Graph,
        centroids: Dict[int, Tuple[float, float]],
        title: str = "RAG Overlay",
        edge_color: str = 'yellow',
        node_color: str = 'red',
        edge_width: float = 1.0,
        node_size: float = 3.0,
        edge_sample_cap: int = 500,
        save_name: Optional[str] = None
    ):
        """
        Overlay RAG on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        rag : nx.Graph
            Region Adjacency Graph.
        centroids : Dict[int, Tuple[float, float]]
            Dictionary mapping node to (x, y) centroid.
        title : str
            Plot title.
        edge_color : str
            Color for edges.
        node_color : str
            Color for nodes.
        edge_width : float
            Width of edges.
        node_size : float
            Size of nodes.
        edge_sample_cap : int
            Maximum number of edges to draw.
        save_name : str, optional
            Name for saved file.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Sample edges if too many
        edges = list(rag.edges())
        if len(edges) > edge_sample_cap:
            np.random.seed(0)
            sampled_idx = np.random.choice(len(edges), edge_sample_cap, replace=False)
            edges = [edges[i] for i in sampled_idx]
        
        # Draw edges
        for u, v in edges:
            xy1 = centroids.get(u)
            xy2 = centroids.get(v)
            if xy1 and xy2:
                ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], 
                       color=edge_color, linewidth=edge_width, alpha=0.7)
        
        # Draw nodes
        for node, (x, y) in centroids.items():
            if node in rag.nodes():
                ax.plot(x, y, 'o', color=node_color, markersize=node_size)
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'rag_overlay')
    
    def plot_graph_overlay_with_df(
        self,
        image: np.ndarray,
        g: nx.Graph,
        node_df: pd.DataFrame,
        title: str = "Graph Overlay",
        edge_sample_cap: int = 500,
        save_name: Optional[str] = None
    ):
        """
        Overlay graph on image using node DataFrame for positions.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        g : nx.Graph
            Graph to overlay.
        node_df : pd.DataFrame
            DataFrame with 'node', 'centroid_col', 'centroid_row' columns.
        title : str
            Plot title.
        edge_sample_cap : int
            Maximum number of edges to draw.
        save_name : str, optional
            Name for saved file.
        """
        pos = {}
        for _, row in node_df.iterrows():
            pos[row['node']] = (row['centroid_col'], row['centroid_row'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Sample edges if too many
        edges = list(g.edges())
        if len(edges) > edge_sample_cap:
            np.random.seed(0)
            sampled_idx = np.random.choice(len(edges), edge_sample_cap, replace=False)
            edges = [edges[i] for i in sampled_idx]
        
        for u, v in edges:
            x0, y0 = pos.get(u, (None, None))
            x1, y1 = pos.get(v, (None, None))
            if None not in (x0, y0, x1, y1):
                ax.plot([x0, x1], [y0, y1], color='yellow', linewidth=1, alpha=0.7)
        
        for node, (x, y) in pos.items():
            if node in g.nodes():
                ax.plot(x, y, 'ro', markersize=3)
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'graph_overlay')
    
    def plot_slic_and_rag(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        rag: nx.Graph,
        centroids: Dict[int, Tuple[float, float]],
        title: str = "",
        save_name: Optional[str] = None
    ):
        """
        Side-by-side plot of SLIC superpixels and RAG overlay.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        rag : nx.Graph
            Region Adjacency Graph.
        centroids : Dict[int, Tuple[float, float]]
            Dictionary mapping node to (x, y) centroid.
        title : str
            Base title.
        save_name : str, optional
            Name for saved file.
        """
        n_superpixels = labels.max() + 1
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # SLIC
        axes[0].imshow(color.label2rgb(labels, image, kind='avg'))
        axes[0].set_title(f'{title}: SLIC ({n_superpixels} superpixels)')
        axes[0].axis('off')
        
        # RAG overlay
        axes[1].imshow(image)
        for n1, n2 in rag.edges:
            xy1 = centroids.get(n1)
            xy2 = centroids.get(n2)
            if xy1 and xy2:
                axes[1].plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], 
                           color='yellow', linewidth=1)
        for node, (x, y) in centroids.items():
            axes[1].plot(x, y, 'ro', markersize=3)
        axes[1].set_title(f'{title}: RAG Overlay')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'slic_and_rag')
    
    def plot_centrality_overlay(
        self,
        image: np.ndarray,
        centrality_df: pd.DataFrame,
        centrality_col: str = 'betweenness',
        top_n_labels: int = 10,
        cmap: str = 'viridis',
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Overlay nodes colored by centrality measure.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        centrality_df : pd.DataFrame
            DataFrame with centrality values.
        centrality_col : str
            Column to use for coloring.
        top_n_labels : int
            Number of top nodes to label.
        cmap : str
            Colormap name.
        title : str, optional
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns:
            print("Cannot plot centrality overlay: missing centroid columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        sc = ax.scatter(
            centrality_df['centroid_col'],
            centrality_df['centroid_row'],
            c=centrality_df[centrality_col],
            cmap=cmap,
            s=30,
            alpha=0.8
        )
        
        # Label top nodes
        for _, row in centrality_df.nlargest(top_n_labels, centrality_col).iterrows():
            ax.text(
                row['centroid_col'], row['centroid_row'],
                str(int(row['node'])),
                color='red', fontsize=8, fontweight='bold'
            )
        
        plt.colorbar(sc, ax=ax, label=centrality_col.replace('_', ' ').title())
        ax.set_title(title or f'Node {centrality_col.title()} Overlay')
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or f'{centrality_col}_overlay')
    
    def plot_community_overlay(
        self,
        image: np.ndarray,
        centrality_df: pd.DataFrame,
        cmap: str = 'tab20',
        title: str = "Community Detection Overlay",
        save_name: Optional[str] = None
    ):
        """
        Overlay nodes colored by community ID.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        centrality_df : pd.DataFrame
            DataFrame with community assignments.
        cmap : str
            Colormap name.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns:
            print("Cannot plot community overlay: missing centroid columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        sc = ax.scatter(
            centrality_df['centroid_col'],
            centrality_df['centroid_row'],
            c=centrality_df['community_id'],
            cmap=cmap,
            s=30,
            alpha=0.8
        )
        
        plt.colorbar(sc, ax=ax, label='Community ID')
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'community_overlay')
    
    def plot_community_bar(
        self,
        community_df: pd.DataFrame,
        title: str = "Community Size Distribution",
        save_name: Optional[str] = None
    ):
        """
        Bar chart of community sizes.
        
        Parameters
        ----------
        community_df : pd.DataFrame
            Community summary DataFrame.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        community_df.plot.bar(x='community_id', y='size', ax=ax, legend=False, color='steelblue')
        ax.set_ylabel('Community Size (# nodes)')
        ax.set_xlabel('Community ID')
        ax.set_title(title)
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'community_bar')
    
    def plot_shortest_path(
        self,
        image: np.ndarray,
        g: nx.Graph,
        centrality_df: pd.DataFrame,
        path_edges: List[Tuple[int, int]],
        path_nodes: List[int],
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Visualize shortest path on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        g : nx.Graph
            Graph.
        centrality_df : pd.DataFrame
            DataFrame with node positions.
        path_edges : List[Tuple[int, int]]
            Edges in the shortest path.
        path_nodes : List[int]
            Nodes in the shortest path.
        title : str, optional
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns or not path_nodes:
            print("Cannot plot shortest path: missing data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        # Draw all edges faintly
        for u, v in g.edges:
            row_u = centrality_df[centrality_df['node'] == u]
            row_v = centrality_df[centrality_df['node'] == v]
            if not row_u.empty and not row_v.empty:
                x0, y0 = row_u['centroid_col'].values[0], row_u['centroid_row'].values[0]
                x1, y1 = row_v['centroid_col'].values[0], row_v['centroid_row'].values[0]
                ax.plot([x0, x1], [y0, y1], color='gray', linewidth=0.3, alpha=0.2)
        
        # Draw path edges in red
        for u, v in path_edges:
            row_u = centrality_df[centrality_df['node'] == u]
            row_v = centrality_df[centrality_df['node'] == v]
            if not row_u.empty and not row_v.empty:
                x0, y0 = row_u['centroid_col'].values[0], row_u['centroid_row'].values[0]
                x1, y1 = row_v['centroid_col'].values[0], row_v['centroid_row'].values[0]
                ax.plot([x0, x1], [y0, y1], color='red', linewidth=3, alpha=0.9)
        
        node_a, node_b = path_nodes[0], path_nodes[-1]
        ax.set_title(title or f'Shortest Path: {node_a} → {node_b} ({len(path_nodes)} nodes)')
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'shortest_path')
    
    def plot_bottleneck_edges(
        self,
        image: np.ndarray,
        centrality_df: pd.DataFrame,
        bottleneck_edges_df: pd.DataFrame,
        title: str = "Top Bottleneck Edges",
        save_name: Optional[str] = None
    ):
        """
        Visualize bottleneck edges on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        centrality_df : pd.DataFrame
            DataFrame with node positions.
        bottleneck_edges_df : pd.DataFrame
            DataFrame with bottleneck edges.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns:
            print("Cannot plot bottleneck edges: missing centroid columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        for _, row in bottleneck_edges_df.iterrows():
            row_u = centrality_df[centrality_df['node'] == row['u']]
            row_v = centrality_df[centrality_df['node'] == row['v']]
            if not row_u.empty and not row_v.empty:
                x0, y0 = row_u['centroid_col'].values[0], row_u['centroid_row'].values[0]
                x1, y1 = row_v['centroid_col'].values[0], row_v['centroid_row'].values[0]
                ax.plot([x0, x1], [y0, y1], color='orange', linewidth=2, alpha=0.8)
        
        ax.set_title(f'{title} (Highest Edge Betweenness)')
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'bottleneck_edges')
    
    def plot_mst(
        self,
        image: np.ndarray,
        mst: nx.Graph,
        centrality_df: pd.DataFrame,
        title: str = "Minimum Spanning Tree",
        save_name: Optional[str] = None
    ):
        """
        Visualize Minimum Spanning Tree on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        mst : nx.Graph
            Minimum Spanning Tree.
        centrality_df : pd.DataFrame
            DataFrame with node positions.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns:
            print("Cannot plot MST: missing centroid columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        for u, v in mst.edges:
            row_u = centrality_df[centrality_df['node'] == u]
            row_v = centrality_df[centrality_df['node'] == v]
            if not row_u.empty and not row_v.empty:
                x0, y0 = row_u['centroid_col'].values[0], row_u['centroid_row'].values[0]
                x1, y1 = row_v['centroid_col'].values[0], row_v['centroid_row'].values[0]
                ax.plot([x0, x1], [y0, y1], color='cyan', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'{title} ({mst.number_of_edges()} edges)')
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'mst')
    
    def plot_mincut_partition(
        self,
        image: np.ndarray,
        centrality_df: pd.DataFrame,
        cut_assignment_df: pd.DataFrame,
        title: str = "Min-Cut Partition Overlay",
        save_name: Optional[str] = None
    ):
        """
        Visualize min-cut partition on the image.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        centrality_df : pd.DataFrame
            DataFrame with node positions.
        cut_assignment_df : pd.DataFrame
            DataFrame with cut assignments.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        if 'centroid_col' not in centrality_df.columns:
            print("Cannot plot min-cut: missing centroid columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image)
        
        for side, clr in zip([0, 1], ["blue", "red"]):
            nodes = cut_assignment_df[cut_assignment_df['cut_side'] == side]['node']
            sub = centrality_df[centrality_df['node'].isin(nodes)]
            if not sub.empty:
                ax.scatter(
                    sub['centroid_col'], sub['centroid_row'],
                    c=clr, label=f'Side {side} ({len(sub)} nodes)',
                    s=30, alpha=0.7
                )
        
        ax.legend()
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'mincut_partition')
    
    def plot_mincut_mask(
        self,
        labels: np.ndarray,
        cut_assignment_df: pd.DataFrame,
        title: str = "Min-Cut Pixel Mask",
        save_name: Optional[str] = None
    ):
        """
        Create and plot pixel mask from min-cut partition.
        
        Parameters
        ----------
        labels : np.ndarray
            Superpixel label map.
        cut_assignment_df : pd.DataFrame
            DataFrame with cut assignments.
        title : str
            Plot title.
        save_name : str, optional
            Name for saved file.
        """
        mask = np.zeros_like(labels, dtype=np.int32)
        
        for side in [0, 1]:
            nodes = cut_assignment_df[cut_assignment_df['cut_side'] == side]['node']
            for n in nodes:
                mask[labels == n] = side + 1
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='coolwarm', alpha=0.7)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        self._finalize_plot(fig, save_name or 'mincut_mask')
