#!/usr/bin/env python
"""
Superpixel RAG Analysis - Main Entry Point

This script runs the complete three-phase pipeline for superpixel-based
Region Adjacency Graph analysis on images.

Usage:
    python main.py                    # Run with default settings
    python main.py --config config.json   # Run with custom config
    python main.py --image path/to/image.jpg  # Run on a single image
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from superpixel_rag.config import Config
from superpixel_rag.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Superpixel RAG Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --image myimage.jpg
  python main.py --config my_config.json
  python main.py --no-visualize --output ./results
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--image',
        type=str,
        action='append',
        help='Path to image file (can be used multiple times)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Disable file export'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save plots to files'
    )
    parser.add_argument(
        '--n-segments',
        type=int,
        default=200,
        help='Number of superpixels'
    )
    parser.add_argument(
        '--compactness',
        type=float,
        default=10.0,
        help='SLIC compactness parameter'
    )
    
    # Phase 4 Arguments
    parser.add_argument(
        '--enable-phase4',
        action='store_true',
        help='Enable Phase 4 AI Layer'
    )
    parser.add_argument(
        '--disable-phase4',
        action='store_true',
        help='Disable Phase 4 AI Layer'
    )
    parser.add_argument(
        '--ai-mode',
        type=str,
        default='hybrid',
        choices=['reasoning_only', 'gnn_only', 'hybrid'],
        help='AI Mode'
    )
    parser.add_argument(
        '--phase4-task',
        type=str,
        default='community_prediction',
        help='Phase 4 Task'
    )
    parser.add_argument(
        '--train-enabled',
        action='store_true',
        help='Enable GNN training'
    )
    parser.add_argument(
        '--llm-enabled',
        action='store_true',
        help='Enable LLM reasoning'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    if args.config:
        # Load from JSON file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        # Build from arguments or use defaults
        config = Config.default()
    
    # Override with command-line arguments
    if args.image:
        config.image_paths = args.image
        config.image_titles = [f"Image {i+1}" for i in range(len(args.image))]
    
    if args.output != '.':
        config.output_dir = args.output
        
    config.show_plots = not args.no_visualize
    if args.save_plots:
        config.save_plots = True
        
    config.phase1.n_segments = args.n_segments
    config.phase1.compactness = args.compactness
    
    # Phase 4 Overrides
    if args.enable_phase4:
        config.phase4.enable_phase4 = True
    if args.disable_phase4:
        config.phase4.enable_phase4 = False
        
    config.phase4.ai_mode = args.ai_mode
    config.phase4.task = args.phase4_task
    if args.train_enabled:
        config.phase4.train_enabled = True
    if args.llm_enabled:
        config.phase4.llm_enabled = True
        
    use_cache = not args.no_cache
    
    # Initialize Pipeline
    pipeline = Pipeline(config)
    
    print("=" * 60)
    print("Superpixel RAG Analysis Pipeline")
    print("=" * 60)
    config.print_summary()
    
    results = []
    
    # Run Pipeline
    for i, image_path in enumerate(config.image_paths):
        title = config.image_titles[i] if i < len(config.image_titles) else Path(image_path).stem
        try:
            bundle = pipeline.run(image_path, title=title, use_cache=use_cache)
            results.append(bundle)
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}", exc_info=True)
            
    print("\n" + "=" * 60)
    print("Pipeline Completed")
    print("=" * 60)
    print(f"Processed {len(results)} images successfully.")
    print(f"Output saved to: {config.output_dir}")

if __name__ == "__main__":
    main()
