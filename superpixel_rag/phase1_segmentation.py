"""
Phase 1: Superpixel Segmentation and RAG Construction.

This module handles SLIC superpixel segmentation and Region Adjacency Graph (RAG) construction.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import networkx as nx
from skimage import io, segmentation, color

try:
    from skimage.future import graph  # For older versions
except ImportError:
    from skimage import graph  # For skimage >=0.20

from .config import Phase1Config
from .utils import compute_centroids, rag_sanity_checks, get_cache_key, load_from_cache, save_to_cache


@dataclass
class SegmentationResult:
    """Result of Phase 1 segmentation for a single image."""
    image: np.ndarray
    labels: np.ndarray
    rag: nx.Graph
    n_superpixels: int
    centroids: Dict[int, Tuple[float, float]]
    title: str = ""
    image_id: str = ""


class Phase1Segmentation:
    """
    Phase 1: Superpixel Segmentation and RAG Construction.
    
    This class handles SLIC superpixel segmentation and builds Region Adjacency Graphs
    with mean-color edge weights.
    """
    
    def __init__(self, config: Optional[Phase1Config] = None, use_cache: bool = False, cache_dir: str = "./cache"):
        """
        Initialize Phase 1 with configuration.
        
        Parameters
        ----------
        config : Phase1Config, optional
            Configuration for segmentation. Uses defaults if not provided.
        use_cache : bool
            Whether to use caching.
        cache_dir : str
            Directory for cache files.
        """
        self.config = config or Phase1Config()
        self.use_cache = use_cache
        self.cache_dir = cache_dir
    
    def load_image(self, path: str) -> np.ndarray:
        """
        Load an image from a file path.
        
        Parameters
        ----------
        path : str
            Path to the image file.
        
        Returns
        -------
        np.ndarray
            Loaded image.
        """
        img = io.imread(path)
        print(f"Loaded image from: {path}")
        print(f"  Shape: {img.shape}, dtype: {img.dtype}")
        return img
    
    def segment_slic(self, image: np.ndarray) -> np.ndarray:
        """
        Perform SLIC superpixel segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
        
        Returns
        -------
        np.ndarray
            Label map where each pixel is assigned to a superpixel.
        """
        labels = segmentation.slic(
            image,
            n_segments=self.config.n_segments,
            compactness=self.config.compactness,
            sigma=self.config.sigma,
            start_label=self.config.start_label
        )
        n_superpixels = labels.max() + 1
        print(f"SLIC segmentation: {n_superpixels} superpixels")
        return labels
    
    def build_rag(self, image: np.ndarray, labels: np.ndarray) -> nx.Graph:
        """
        Build a Region Adjacency Graph from superpixel labels.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        
        Returns
        -------
        nx.Graph
            Region Adjacency Graph with mean color for each node.
        """
        rag = graph.rag_mean_color(image, labels, mode=self.config.rag_mode)
        print(f"RAG built: {rag.number_of_nodes()} nodes, {rag.number_of_edges()} edges")
        return rag
    
    def process_image(self, image: np.ndarray, title: str = "", image_id: str = "") -> SegmentationResult:
        """
        Process a single image through the complete Phase 1 pipeline.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
        title : str, optional
            Title/name for the image.
        image_id : str, optional
            Unique identifier for caching.
        
        Returns
        -------
        SegmentationResult
            Complete segmentation result.
        """
        print(f"\n{'='*40}")
        print(f"Processing: {title or 'Image'}")
        print('='*40)
        
        # Caching
        cache_key = None
        if self.use_cache and image_id:
            # Create hash based on config and image_id
            # We assume image content doesn't change for the same ID, or we could hash image content
            # For speed, we rely on ID + config
            params = {
                'n_segments': self.config.n_segments,
                'compactness': self.config.compactness,
                'sigma': self.config.sigma,
                'rag_mode': self.config.rag_mode
            }
            cache_key = get_cache_key(image_id, params)
            
            # Try load (using pickle for graphs to preserve complex types)
            cached_labels = load_from_cache(self.cache_dir, cache_key, 'labels.npy')
            cached_rag = load_from_cache(self.cache_dir, cache_key, 'rag.pkl')
            
            if cached_labels is not None and cached_rag is not None:
                print("Loaded Phase 1 results from cache.")
                n_superpixels = cached_labels.max() + 1
                centroids = compute_centroids(cached_labels)
                return SegmentationResult(
                    image=image,
                    labels=cached_labels,
                    rag=cached_rag,
                    n_superpixels=n_superpixels,
                    centroids=centroids,
                    title=title,
                    image_id=image_id
                )

        # Segmentation
        labels = self.segment_slic(image)
        n_superpixels = labels.max() + 1
        
        # RAG construction
        rag = self.build_rag(image, labels)
        
        # Compute centroids
        centroids = compute_centroids(labels)
        
        # Sanity checks
        rag_sanity_checks(rag)
        
        # Save to cache (using pickle for graphs to preserve complex types)
        if self.use_cache and cache_key:
            save_to_cache(self.cache_dir, cache_key, 'labels.npy', labels)
            save_to_cache(self.cache_dir, cache_key, 'rag.pkl', rag)
        
        return SegmentationResult(
            image=image,
            labels=labels,
            rag=rag,
            n_superpixels=n_superpixels,
            centroids=centroids,
            title=title,
            image_id=image_id
        )
    
    def process_images(
        self,
        image_paths: List[str],
        titles: Optional[List[str]] = None
    ) -> List[SegmentationResult]:
        """
        Process multiple images through Phase 1.
        
        Parameters
        ----------
        image_paths : List[str]
            List of paths to images.
        titles : List[str], optional
            List of titles for each image.
        
        Returns
        -------
        List[SegmentationResult]
            List of segmentation results.
        """
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(image_paths))]
        
        results = []
        for path, title in zip(image_paths, titles):
            image = self.load_image(path)
            # Use filename as ID
            image_id = Path(path).stem
            result = self.process_image(image, title, image_id)
            results.append(result)
        
        print(f"\n{'='*40}")
        print(f"Phase 1 completed for {len(results)} images")
        print('='*40)
        
        return results
    
    @staticmethod
    def get_superpixel_overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Create an image with superpixel average colors.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        
        Returns
        -------
        np.ndarray
            Image with superpixel average colors.
        """
        return color.label2rgb(labels, image, kind='avg')
    
    @staticmethod
    def get_boundary_image(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Create an image with superpixel boundaries overlaid.
        
        Parameters
        ----------
        image : np.ndarray
            Original image.
        labels : np.ndarray
            Superpixel label map.
        
        Returns
        -------
        np.ndarray
            Image with boundaries marked.
        """
        return segmentation.mark_boundaries(image, labels)
