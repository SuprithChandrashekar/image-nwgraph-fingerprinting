# Superpixel RAG Analysis

A modular Python pipeline for superpixel-based image segmentation and Region Adjacency Graph (RAG) analysis.

## Overview

This package implements a four-phase pipeline for analyzing images using superpixel segmentation and graph-based methods:

1. **Phase 1: Segmentation** - SLIC superpixel segmentation and RAG construction
2. **Phase 2: Enrichment** - Feature extraction, edge weighting, and graph pruning
3. **Phase 3: Analysis** - Centrality measures, community detection, shortest paths, MST, and min-cut partitioning
4. **Phase 4: AI Layer** - Predictive learning (GNN), graph reasoning, and explainability artifacts

## Project Structure

```
superpixel_rag/
├── __init__.py           # Package initialization
├── config.py             # Configuration dataclasses
├── pipeline.py           # Main pipeline orchestrator
├── data.py               # Data contracts (GraphBundle)
├── export_manager.py     # Centralized export logic
├── utils.py              # Shared utility functions & Caching
├── phase1_segmentation.py # SLIC and RAG construction
├── phase2_enrichment.py   # Feature extraction and pruning
├── phase3_analysis.py     # Graph analysis algorithms
├── phase4_ai.py           # AI Layer orchestrator
├── ai_features.py         # Feature matrix construction
├── ai_reasoning.py        # Reasoning and LLM hooks
├── ai_gnn.py              # GNN training (PyTorch/PyG)
├── ai_export.py           # Dataset export utilities
├── visualization.py       # Plotting and visualization
├── main.py               # CLI entry point
└── requirements.txt      # Dependencies
```

## Installation

```bash
# Install dependencies
pip install -r superpixel_rag/requirements.txt

# Optional: Install Louvain community detection
pip install python-louvain

# Optional: Install PyTorch and PyTorch Geometric for GNN training
# See https://pytorch.org/get-started/locally/
# See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Quick Start

### Option 1: Run the pre-configured script

```bash
python superpixel_rag/main.py
```

### Option 2: Use the Python API (New Pipeline)

```python
from superpixel_rag.config import Config
from superpixel_rag.pipeline import Pipeline

# Create configuration
config = Config.default()
config.image_paths = ['path/to/image.jpg']
config.image_titles = ['My Image']
config.phase4.enable_phase4 = True

# Initialize Pipeline
pipeline = Pipeline(config)

# Run Pipeline
for i, image_path in enumerate(config.image_paths):
    bundle = pipeline.run(image_path, title=config.image_titles[i])
    print(f"Processed {bundle.graph_name} with {bundle.G.number_of_nodes()} nodes")
```

### Option 3: Use the CLI

```bash
# Run with default settings
python superpixel_rag/main.py

# Run on specific images
python superpixel_rag/main.py --image image1.jpg --image image2.png

# Customize parameters
python superpixel_rag/main.py --n-segments 300 --compactness 15 --output ./results

# Enable Phase 4 (AI Layer)
python superpixel_rag/main.py --enable-phase4

# Disable visualization (faster)
python superpixel_rag/main.py --no-visualize

# Disable caching (force re-computation)
python superpixel_rag/main.py --no-cache
```

## Key Features

### Unified Pipeline
The system is orchestrated by a central `Pipeline` class that manages data flow between phases using a `GraphBundle` object. This ensures efficient memory usage and consistent state management.

### Caching System
Expensive operations (SLIC segmentation, Graph construction) are automatically cached based on configuration parameters and image IDs. This significantly speeds up re-runs when tweaking downstream parameters.

### Parquet & GraphML Exports
Data is exported in efficient formats:
- **Parquet**: For large DataFrames (node features, edge lists).
- **GraphML**: For graph structures compatible with Gephi/Cytoscape.
- **CSV**: For human-readable summaries.

## Configuration Options

### Phase 1 (Segmentation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_segments` | 200 | Number of superpixels |
| `compactness` | 10.0 | Balance between color and space proximity |
| `sigma` | 1.0 | Gaussian smoothing kernel width |
| `rag_mode` | 'distance' | Edge weight mode: 'distance' or 'similarity' |

### Phase 2 (Enrichment)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_weight_mode` | 'mean_color_distance' | Weight mode: 'mean_color_distance', 'boundary_contrast', 'combined' |
| `alpha_color` | 0.5 | Weight for color distance in combined mode |
| `alpha_boundary` | 0.5 | Weight for boundary contrast in combined mode |
| # Phase 4 (AI Layer)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase4` | True | Enable AI layer |
| `ai_mode` | 'hybrid' | 'reasoning_only', 'gnn_only', 'hybrid' |
| `task` | 'community_prediction' | GNN task: 'community_prediction', 'cut_prediction' |
| `train_enabled` | False | Enable GNN training (requires PyTorch) |
| `llm_enabled` | False | Enable LLM reasoning hooks |
| `feature_set` | 'full' | Feature set for ML: 'basic', 'full' |

## Outputs

The pipeline generates:

### Data Files (Parquet & CSV)
- `nodes.parquet` / `nodes.csv` - Node attributes (mean color, features)
- `edges.parquet` / `edges.csv` - Edge attributes (weights)
- `centrality.parquet` / `centrality.csv` - Node centrality measures
- `communities.csv` - Community assignments
- `bottleneck_edges.csv` - Top bottleneck edges
- `mst_edges.csv` - Minimum Spanning Tree edges
- `shortest_path_*.csv` - Shortest path nodes/edges
- `mincut_assignment.csv` - Min-cut partition (if seeds provided)
- `anomalies.csv` - Top anomalous superpixels (Phase 4)

### GraphML Files
- `rag.graphml` - Phase 1 RAG exports
- `rag_enriched.graphml` - Enriched graph with all attributes

### AI Artifacts (Phase 4)
- `gnn_dataset/` - Numpy files for GNN training (`X_nodes.npy`, `edge_index.npy`, etc.)
- `phase4_digest.json` - Structured graph statistics
- `phase4_exec_summary.txt` - Executive summary of graph analysis

### Visualizations
- SLIC superpixel overlays
- RAG graph overlays
- Centrality heatmaps
- Community coloring
- Shortest path highlights
- MST visualization
- Bottleneck edge highlighting
- Min-cut partitions
- Anomaly node highlighting (Phase 4)

## GUI Application

A Streamlit-based GUI is available for running the pipeline and visualizing results.

```bash
# Install GUI dependencies
pip install -r requirements-gui.txt

# Run the App
streamlit run apps/gui/app.py
```

## Development & Testing

This project uses `pytest` for testing and `ruff`/`black` for code quality.

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0
- networkx >= 2.6.0
- matplotlib >= 3.4.0
- Pillow >= 8.0.0

Optional:
- python-louvain >= 0.16 (for Louvain community detection)
- torch >= 1.10.0 (for GNN training)
- torch-geometric >= 2.0.0 (for GNN training
- Bottleneck edge highlighting
- Min-cut partitions

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-image >= 0.19.0
- networkx >= 2.6.0
- matplotlib >= 3.4.0
- Pillow >= 8.0.0

Optional:
- python-louvain >= 0.16 (for Louvain community detection)

## License

Research Project - University of Illinois Urbana-Champaign
