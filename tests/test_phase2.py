import pytest
from superpixel_rag.phase1_segmentation import Phase1Segmentation
from superpixel_rag.phase2_enrichment import Phase2Enrichment

def test_phase2_enrichment(test_config):
    """Test Phase 2 logic."""
    # Run Phase 1 first
    p1 = Phase1Segmentation(test_config.phase1, use_cache=False)
    image_path = test_config.image_paths[0]
    image = p1.load_image(image_path)
    p1_res = p1.process_image(image, title="Test", image_id="test")
    
    # Run Phase 2
    p2 = Phase2Enrichment(test_config.phase2)
    p2_res = p2.process(p1_res)
    
    assert p2_res.node_features is not None
    assert p2_res.edge_df is not None
    
    # Check columns
    required_cols = ['area', 'perimeter', 'mean_color']
    for col in required_cols:
        assert col in p2_res.node_features.columns
        
    assert 'weight' in p2_res.edge_df.columns or 'weight2' in p2_res.edge_df.columns
