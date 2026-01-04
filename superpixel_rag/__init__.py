"""
Superpixel RAG Analysis Package

A modular pipeline for superpixel segmentation, Region Adjacency Graph (RAG) construction,
graph enrichment, and classical graph analysis.

Phases:
    1. Segmentation: SLIC superpixels and RAG construction
    2. Enrichment: Feature extraction, edge weighting, and pruning
    3. Analysis: Centrality, communities, paths, MST, and min-cut
"""

__version__ = "1.0.0"
__author__ = "Research Project"

from .config import Config, Phase4Config
from .phase1_segmentation import Phase1Segmentation
from .phase2_enrichment import Phase2Enrichment
from .phase3_analysis import Phase3Analysis
from .phase4_ai import Phase4AI
from .visualization import Visualizer
from .utils import export_graph, sanitize_attrs_for_graphml

__all__ = [
    "Config",
    "Phase1Segmentation",
    "Phase2Enrichment",
    "Phase3Analysis",
    "Visualizer",
    "export_graph",
    "sanitize_attrs_for_graphml",
]
