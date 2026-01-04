import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np

from .config import Config
from .data import GraphBundle
from .export_manager import ExportManager
from .phase1_segmentation import Phase1Segmentation
from .phase2_enrichment import Phase2Enrichment
from .phase3_analysis import Phase3Analysis
from .phase4_ai import Phase4AI
from .visualization import Visualizer

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Orchestrates the Superpixel RAG pipeline.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.export_manager = ExportManager(config)
        
        # Initialize Phases
        self.p1 = Phase1Segmentation(config.phase1, use_cache=config.use_cache, cache_dir=config.cache_dir)
        self.p2 = Phase2Enrichment(config.phase2)
        self.p3 = Phase3Analysis(config.phase3)
        self.p4 = Phase4AI(config.phase4)
        
        # Initialize Visualizer
        self.viz = Visualizer(
            show_plots=config.show_plots,
            save_plots=config.save_plots,
            output_dir=config.output_dir,
            dpi=config.figure_dpi
        )
        
    def run(self, image_path: str, title: Optional[str] = None, use_cache: bool = True) -> GraphBundle:
        """
        Run the full pipeline on an image.
        
        Parameters
        ----------
        image_path : str
            Path to the input image.
        title : str, optional
            Title for the image.
        use_cache : bool
            Whether to use cached results for expensive steps.
            
        Returns
        -------
        GraphBundle
            Final bundle containing all results.
        """
        start_time = time.time()
        logger.info(f"Starting pipeline for {image_path}")
        
        # --- Phase 1: Segmentation ---
        logger.info("Running Phase 1: Segmentation...")
        # Load image
        image = self.p1.load_image(image_path)
        image_id = Path(image_path).stem
        p1_result = self.p1.process_image(image, title=title or image_id, image_id=image_id)
        
        # Store image_path in result for later export
        p1_result.image_path = image_path
        
        # Viz Phase 1
        if self.config.save_plots or self.config.show_plots:
            self.viz.plot_slic_and_rag(
                p1_result.image,
                p1_result.labels,
                p1_result.rag,
                p1_result.centroids,
                title=p1_result.title,
                save_name=f"{p1_result.image_id}_phase1"
            )
        
        # --- Phase 2: Enrichment ---
        logger.info("Running Phase 2: Enrichment...")
        p2_result = self.p2.process(p1_result)
        
        # Viz Phase 2
        if self.config.save_plots or self.config.show_plots:
            # Visualize weights if needed, or just the graph
            pass # Add specific Phase 2 viz if needed
        
        # --- Phase 3: Analysis ---
        logger.info("Running Phase 3: Analysis...")
        # Phase 3 now returns a GraphBundle
        bundle = self.p3.process(p2_result)
        
        # Viz Phase 3 - use existing visualization methods
        if self.config.save_plots or self.config.show_plots:
            # Plot community overlay if centrality_df has community info
            if bundle.centrality_df is not None and 'community_id' in bundle.centrality_df.columns:
                self.viz.plot_community_overlay(
                    bundle.image,
                    bundle.centrality_df,
                    title=f"{bundle.title} - Communities",
                    save_name=f"{bundle.image_id}_communities"
                )
        
        # --- Phase 4: AI Layer ---
        if self.config.phase4.enable_phase4:
            logger.info("Running Phase 4: AI Layer...")
            bundle = self.p4.process(bundle)
            
            # Viz Phase 4
            if (self.config.save_plots or self.config.show_plots) and bundle.phase4_results:
                p4_res = bundle.phase4_results
                if p4_res.anomaly_df is not None:
                    # Reconstruct centroids list from node_df
                    # node_df has 'centroid_row', 'centroid_col'
                    # We need a list where index corresponds to label? 
                    # Visualizer expects centroids as list of (row, col) indexed by label?
                    # Let's check Visualizer.plot_anomalies
                    
                    # Assuming node_df is sorted by label or we create a map
                    # But Visualizer might expect a list where centroids[i] is centroid of label i?
                    # Or just a list matching the order of something?
                    # Let's look at Visualizer.plot_anomalies implementation if possible.
                    # For now, let's assume it takes a list of tuples.
                    
                    # Handle both 'label' and 'node' column names
                    node_id_col = 'label' if 'label' in bundle.node_df.columns else 'node'
                    
                    # Create a map from label to centroid
                    centroid_map = {}
                    for _, row in bundle.node_df.iterrows():
                        centroid_map[int(row[node_id_col])] = (row['centroid_row'], row['centroid_col'])
                    
                    # Create list. Max label might be larger than len(node_df) if filtered?
                    # But usually labels are 0..N-1 or 1..N.
                    # Let's just pass the map if Visualizer supports it, or a list where index=label.
                    max_label = int(bundle.node_df[node_id_col].max())
                    centroids_list = [(0.0, 0.0)] * (max_label + 1)
                    for label, cent in centroid_map.items():
                        centroids_list[label] = cent
                        
                    self.viz.plot_anomalies(
                        bundle.image,
                        bundle.labels,
                        p4_res.anomaly_df,
                        np.array(centroids_list),
                        title=f"{bundle.title} - Top Anomalies",
                        save_name=f"{bundle.image_id}_phase4_anomalies"
                    )
        else:
            logger.info("Phase 4 disabled, skipping.")
            
        # --- Export ---
        logger.info("Exporting results...")
        self.export_manager.export_bundle(bundle)
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.2f}s")
        
        return bundle

def run_pipeline(config: Config, image_path: str, title: Optional[str] = None, use_cache: bool = True) -> GraphBundle:
    """
    Helper function to run the pipeline.
    """
    pipeline = Pipeline(config)
    return pipeline.run(image_path, title=title, use_cache=use_cache)
