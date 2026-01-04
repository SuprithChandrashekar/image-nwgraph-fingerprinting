import pytest
import numpy as np
from superpixel_rag.phase1_segmentation import Phase1Segmentation
from superpixel_rag.config import Phase1Config

def test_phase1_segmentation(test_config):
    """Test Phase 1 logic."""
    p1 = Phase1Segmentation(test_config.phase1, use_cache=False)
    image_path = test_config.image_paths[0]
    
    # Load image first
    image = p1.load_image(image_path)
    result = p1.process_image(image, title="Test", image_id="test")
    
    assert result.image is not None
    assert result.labels is not None
    assert result.rag is not None
    
    # Check labels shape
    assert result.labels.shape[:2] == result.image.shape[:2]
    
    # Check RAG nodes match unique labels
    unique_labels = np.unique(result.labels)
    assert len(unique_labels) == result.rag.number_of_nodes()
