"""
Configuration module for Superpixel RAG Analysis.

Contains all adjustable parameters for the three phases of the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Phase1Config:
    """Configuration for Phase 1: Segmentation."""
    n_segments: int = 200
    compactness: float = 10.0
    sigma: float = 1.0
    rag_mode: str = 'distance'  # 'distance' or 'similarity'
    start_label: int = 0


@dataclass
class Phase2Config:
    """Configuration for Phase 2: Graph Enrichment."""
    edge_weight_mode: str = 'mean_color_distance'  # 'mean_color_distance', 'boundary_contrast', 'combined'
    alpha_color: float = 0.5
    alpha_boundary: float = 0.5
    prune_mode: str = 'threshold'  # 'threshold', 'topk', 'percentile'
    prune_value: float = 0.2
    min_region_area: int = 50
    edge_sample_cap: int = 500
    merge_threshold: float = 0.1
    max_merges: int = 50
    enable_merging: bool = False


@dataclass
class Phase3Config:
    """Configuration for Phase 3: Graph Analysis."""
    graph_source: str = 'g_pruned'  # 'g2', 'g_pruned', 'rag'
    weight_attr: str = 'weight2'
    centrality_topn: int = 15
    community_method: str = 'greedy_modularity'  # 'greedy_modularity', 'louvain_if_available'
    path_node_a: Optional[int] = None
    path_node_b: Optional[int] = None
    mincut_source_seeds: List[int] = field(default_factory=list)
    mincut_sink_seeds: List[int] = field(default_factory=list)
    random_state: int = 42


@dataclass
class Phase4Config:
    """Configuration for Phase 4: AI Layer."""
    enable_phase4: bool = True
    ai_mode: str = "hybrid"  # "reasoning_only", "gnn_only", "hybrid"
    graph_source: str = "g_pruned"  # "rag", "g2", "g_pruned"
    weight_attr: str = "weight2"
    feature_set: str = "full"  # "basic", "full"
    task: str = "community_prediction"  # "community_prediction", "cut_prediction", "anomaly_scoring", "none"
    train_enabled: bool = False
    random_state: int = 0
    export_embeddings: bool = True
    export_reasoning: bool = True
    llm_enabled: bool = False
    llm_provider: str = "none"
    llm_model: str = "none"
    llm_max_nodes: int = 250
    llm_max_edges: int = 800


@dataclass
class Config:
    """Main configuration class combining all phase configurations."""
    
    # Image paths
    image_paths: List[str] = field(default_factory=list)
    image_titles: List[str] = field(default_factory=list)
    
    # Output directory
    output_dir: str = '.'
    
    # Phase configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    
    # Visualization settings
    show_plots: bool = True
    save_plots: bool = False
    figure_dpi: int = 100
    
    # Pipeline settings
    export_level: str = "standard"  # "minimal", "standard", "full"
    use_cache: bool = False
    cache_dir: str = "./cache"
    test_mode: bool = False
    
    @classmethod
    def default(cls) -> 'Config':
        """Create a default configuration."""
        return cls(
            image_paths=[
                r'Gemini Nano Banana\unnamed.jpg',
                r'GPT Image Gen\653601e7-f502-43f3-bc50-c682685b7c3a.png',
                r'Original Image\Histopathology Image.jpg'
            ],
            image_titles=["Image 1", "Image 2", "Image 3"],
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Create configuration from a dictionary."""
        phase1 = Phase1Config(**d.get('phase1', {}))
        phase2 = Phase2Config(**d.get('phase2', {}))
        phase3 = Phase3Config(**d.get('phase3', {}))
        phase4 = Phase4Config(**d.get('phase4', {}))
        
        return cls(
            image_paths=d.get('image_paths', []),
            image_titles=d.get('image_titles', []),
            output_dir=d.get('output_dir', '.'),
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            show_plots=d.get('show_plots', True),
            save_plots=d.get('save_plots', False),
            figure_dpi=d.get('figure_dpi', 100),
            export_level=d.get('export_level', 'standard'),
            use_cache=d.get('use_cache', False),
            cache_dir=d.get('cache_dir', './cache')
        )
    
    def validate(self):
        """Validate configuration consistency."""
        pass

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        print("\n[Phase 1 - Segmentation]")
        print(f"  n_segments: {self.phase1.n_segments}")
        print(f"  compactness: {self.phase1.compactness}")
        print(f"  sigma: {self.phase1.sigma}")
        print(f"  rag_mode: {self.phase1.rag_mode}")
        
        print("\n[Phase 2 - Enrichment]")
        print(f"  edge_weight_mode: {self.phase2.edge_weight_mode}")
        print(f"  alpha_color: {self.phase2.alpha_color}")
        print(f"  alpha_boundary: {self.phase2.alpha_boundary}")
        print(f"  prune_mode: {self.phase2.prune_mode}")
        print(f"  prune_value: {self.phase2.prune_value}")
        print(f"  enable_merging: {self.phase2.enable_merging}")
        
        print("\n[Phase 3 - Analysis]")
        print(f"  graph_source: {self.phase3.graph_source}")
        print(f"  weight_attr: {self.phase3.weight_attr}")
        print(f"  centrality_topn: {self.phase3.centrality_topn}")
        print(f"  community_method: {self.phase3.community_method}")
        print(f"  mincut_source_seeds: {self.phase3.mincut_source_seeds}")
        print(f"  mincut_sink_seeds: {self.phase3.mincut_sink_seeds}")
        
        print("\n" + "=" * 60)
