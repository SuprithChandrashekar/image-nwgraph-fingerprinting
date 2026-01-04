# Documentation Assets

This directory contains images and documentation assets for the README.

## Required Images

To complete the documentation, add the following images:

1. **pipeline_overview.png** - Diagram showing the 4-phase pipeline flow
2. **segmentation_example.png** - Example of superpixel segmentation output
3. **community_example.png** - Example of community detection visualization
4. **anomaly_example.png** - Example of anomaly detection output

## Generating Example Images

You can generate example images by running the pipeline:

```python
from superpixel_rag.config import Config
from superpixel_rag.pipeline import Pipeline

config = Config.default()
config.image_paths = ['Original Image/Satellite Image.webp']
config.save_plots = True
config.output_dir = './docs'

pipeline = Pipeline(config)
pipeline.run(config.image_paths[0], title='Example')
```

Then move the generated plots to this directory.
