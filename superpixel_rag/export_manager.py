"""
Export Manager Module

Handles centralized exporting of analysis artifacts.
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .data import GraphBundle
from .utils import export_graph, sanitize_attrs_for_graphml, to_python_type
# Avoid circular import if possible, but we need Config for type hinting
# from .config import Config 


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    return obj


class ExportManager:
    """
    Manages file exports for the pipeline.
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.export_level = config.export_level
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_bundle(self, bundle: GraphBundle):
        """
        Export all relevant data from a GraphBundle based on export level.
        """
        safe_title = "".join([c if c.isalnum() else "_" for c in bundle.title])
        # Use a run_id based on timestamp or just title? 
        # Instructions say "output/<run_id>/manifest.json".
        # If we use title as folder, that's the run_id.
        run_id = safe_title
        run_dir = self.output_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": self._get_config_hash(),
            "image_path": str(bundle.image_path) if hasattr(bundle, 'image_path') else "",
            "title": bundle.title,
            "paths": {},
            "status": {
                "has_phase4": False,
                "has_mincut": False,
                "has_gnn": False
            }
        }
        
        # Always export core tables
        self._export_core(bundle, run_dir, manifest)
        
        # Export Phase 3 Analysis
        if self.export_level in ["standard", "full"]:
            self._export_analysis(bundle, run_dir, manifest)
            
        # Export Phase 4 AI
        if bundle.phase4_results:
            manifest["status"]["has_phase4"] = True
            self._export_ai(bundle, run_dir, manifest)
            
        # Write Manifest
        with open(run_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Exported results to {run_dir}")

    def _get_config_hash(self) -> str:
        """Generate a hash of the current configuration."""
        # Simple hash of the config dict
        try:
            c_dict = self.config.to_dict()
            s = json.dumps(c_dict, sort_keys=True, default=str)
            return hashlib.md5(s.encode()).hexdigest()
        except:
            return "unknown"

    def _export_core(self, bundle: GraphBundle, out_dir: Path, manifest: Dict):
        """Export core graph and attributes."""
        # Original Image (Copy or Link? Instructions say "paths for original_image")
        # We won't copy the original image to save space, just point to it.
        # But for the GUI to load it, it might need to be accessible.
        # Let's assume we just store the path.
        manifest["paths"]["original_image"] = str(bundle.image_path) if hasattr(bundle, 'image_path') else ""
        
        # Node Attributes
        node_csv = out_dir / 'nodes.csv'
        node_pq = out_dir / 'nodes.parquet'
        bundle.node_df.to_csv(node_csv, index=False)
        manifest["paths"]["nodes_csv"] = str(node_csv.relative_to(self.output_dir))
        try:
            bundle.node_df.to_parquet(node_pq)
            manifest["paths"]["nodes_parquet"] = str(node_pq.relative_to(self.output_dir))
        except ImportError:
            pass
            
        # Edge Attributes
        edge_csv = out_dir / 'edges.csv'
        edge_pq = out_dir / 'edges.parquet'
        bundle.edge_df.to_csv(edge_csv, index=False)
        manifest["paths"]["edges_csv"] = str(edge_csv.relative_to(self.output_dir))
        try:
            bundle.edge_df.to_parquet(edge_pq)
            manifest["paths"]["edges_parquet"] = str(edge_pq.relative_to(self.output_dir))
        except ImportError:
            pass
            
        # GraphML
        rag_path = out_dir / 'rag.graphml'
        export_graph(
            bundle.G,
            graphml_file='rag.graphml',
            output_dir=str(out_dir)
        )
        manifest["paths"]["rag_graphml"] = str(rag_path.relative_to(self.output_dir))

    def _export_analysis(self, bundle: GraphBundle, out_dir: Path, manifest: Dict):
        """Export Phase 3 analysis results."""
        if bundle.centrality_df is not None:
            p = out_dir / 'centrality.csv'
            bundle.centrality_df.to_csv(p)
            manifest["paths"]["centrality_csv"] = str(p.relative_to(self.output_dir))
            try:
                pq = out_dir / 'centrality.parquet'
                bundle.centrality_df.to_parquet(pq)
                manifest["paths"]["centrality_parquet"] = str(pq.relative_to(self.output_dir))
            except ImportError:
                pass
            
        if bundle.community_df is not None:
            p = out_dir / 'communities.csv'
            bundle.community_df.to_csv(p)
            manifest["paths"]["communities_csv"] = str(p.relative_to(self.output_dir))
            
        if bundle.bottleneck_edges_df is not None and self.export_level == "full":
            p = out_dir / 'bottleneck_edges.csv'
            bundle.bottleneck_edges_df.to_csv(p, index=False)
            manifest["paths"]["bottleneck_edges_csv"] = str(p.relative_to(self.output_dir))
            
        if bundle.mst is not None and self.export_level == "full":
            p = out_dir / 'mst.graphml'
            mst_copy = bundle.mst.copy()
            sanitize_attrs_for_graphml(mst_copy)
            nx.write_graphml(mst_copy, p)
            manifest["paths"]["mst_graphml"] = str(p.relative_to(self.output_dir))
            
        if bundle.cut_assignment_df is not None:
            p = out_dir / 'mincut_assignment.csv'
            bundle.cut_assignment_df.to_csv(p)
            manifest["paths"]["mincut_assignment_csv"] = str(p.relative_to(self.output_dir))
            manifest["status"]["has_mincut"] = True

    def _export_ai(self, bundle: GraphBundle, out_dir: Path, manifest: Dict):
        """Export Phase 4 AI results."""
        res = bundle.phase4_results
        if res is None:
            return
        
        # Digest
        if res.reasoning_report:
            p_digest = out_dir / 'phase4_digest.json'
            with open(p_digest, 'w') as f:
                # Convert numpy types to Python native types for JSON serialization
                digest = _make_json_serializable(res.reasoning_report.get('digest', {}))
                json.dump(digest, f, indent=2)
            manifest["paths"]["phase4_digest_json"] = str(p_digest.relative_to(self.output_dir))
                
            # Summaries
            s = res.reasoning_report.get('executive_summary', {})
            if s:
                p_exec = out_dir / 'phase4_exec_summary.txt'
                with open(p_exec, 'w') as f:
                    f.write(s.get('one_paragraph', '') + "\n\n")
                    f.write("Value:\n" + "\n".join(f"- {x}" for x in s.get('bullets_value', [])) + "\n")
                manifest["paths"]["phase4_exec_summary_txt"] = str(p_exec.relative_to(self.output_dir))
        
        # Anomalies
        if res.anomaly_df is not None:
            p_anom = out_dir / 'anomalies.csv'
            res.anomaly_df.to_csv(p_anom, index=False)
            manifest["paths"]["anomalies_csv"] = str(p_anom.relative_to(self.output_dir))
            
        # GNN Artifacts
        # Check if GNN dataset exists in the folder (Phase 4 might have written it)
        gnn_dir = out_dir / 'gnn_dataset'
        if gnn_dir.exists():
             manifest["status"]["has_gnn"] = True
             manifest["paths"]["gnn_dataset"] = str(gnn_dir.relative_to(self.output_dir))

