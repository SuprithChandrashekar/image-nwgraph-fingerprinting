import pytest
import json
from pathlib import Path
from superpixel_rag.pipeline import Pipeline
from superpixel_rag.export_manager import ExportManager

def test_manifest_generation(test_config):
    """Test that manifest.json is created and valid."""
    pipeline = Pipeline(test_config)
    image_path = test_config.image_paths[0]
    bundle = pipeline.run(image_path, title="Manifest Test")
    
    # Check output directory
    out_dir = Path(test_config.output_dir)
    # Find the run folder (it's named after title "Manifest_Test")
    run_dir = out_dir / "Manifest_Test"
    
    assert run_dir.exists()
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    assert manifest['title'] == "Manifest Test"
    assert 'paths' in manifest
    assert 'nodes_csv' in manifest['paths']
    assert 'rag_graphml' in manifest['paths']
    
    # Check if files exist
    nodes_path = out_dir / manifest['paths']['nodes_csv']
    assert nodes_path.exists()
