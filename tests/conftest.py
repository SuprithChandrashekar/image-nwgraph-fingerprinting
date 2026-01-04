import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import tempfile
from superpixel_rag.config import Config

@pytest.fixture(scope="session")
def test_assets_dir():
    """Create a temporary directory with test assets."""
    tmp_dir = Path(tempfile.mkdtemp())
    assets_dir = tmp_dir / "assets"
    assets_dir.mkdir()
    
    # Create a deterministic synthetic image
    # 64x64 image with 4 quadrants of different colors
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[0:32, 0:32] = [255, 0, 0]     # Red
    img[0:32, 32:64] = [0, 255, 0]    # Green
    img[32:64, 0:32] = [0, 0, 255]    # Blue
    img[32:64, 32:64] = [255, 255, 0] # Yellow
    
    img_path = assets_dir / "test_image.png"
    Image.fromarray(img).save(img_path)
    
    yield assets_dir
    
    shutil.rmtree(tmp_dir)

@pytest.fixture
def test_config(test_assets_dir):
    """Return a test configuration."""
    config = Config.default()
    config.test_mode = True
    config.image_paths = [str(test_assets_dir / "test_image.png")]
    config.image_titles = ["Test Image"]
    config.output_dir = str(test_assets_dir.parent / "output")
    config.cache_dir = str(test_assets_dir.parent / "cache")  # Use temp cache dir
    config.phase1.n_segments = 10 # Small number for speed
    config.phase4.enable_phase4 = True
    config.phase4.train_enabled = False # Skip training in unit tests usually
    return config
