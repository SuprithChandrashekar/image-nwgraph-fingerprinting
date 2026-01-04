import pytest
from pathlib import Path
from superpixel_rag.pipeline import Pipeline
from superpixel_rag.data import GraphBundle

def test_pipeline_end_to_end(test_config):
    """Test the full pipeline execution."""
    pipeline = Pipeline(test_config)
    
    # Run on the first image
    image_path = test_config.image_paths[0]
    bundle = pipeline.run(image_path, title="Test Run")
    
    assert isinstance(bundle, GraphBundle)
    assert bundle.G is not None
    assert bundle.node_df is not None
    assert bundle.edge_df is not None
    
    # Check Phase 1
    assert len(bundle.node_df) > 0
    assert 'mean_color' in bundle.node_df.columns
    
    # Check Phase 3
    assert bundle.centrality_df is not None
    assert bundle.community_df is not None
    
    # Check Phase 4
    if test_config.phase4.enable_phase4:
        assert bundle.phase4_results is not None
        assert bundle.phase4_results.anomaly_df is not None

def test_pipeline_caching(test_config):
    """Test that caching works."""
    test_config.use_cache = True
    pipeline = Pipeline(test_config)
    image_path = test_config.image_paths[0]
    
    # First run
    bundle1 = pipeline.run(image_path, title="Cache Test")
    
    # Second run
    bundle2 = pipeline.run(image_path, title="Cache Test")
    
    # Should be identical objects if cached in memory, or at least identical content
    assert len(bundle1.node_df) == len(bundle2.node_df)
