# 🔬 Superpixel RAG Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular Python pipeline for **superpixel-based image segmentation** and **Region Adjacency Graph (RAG) analysis** with AI-powered insights.

> **Research Project** - University of Illinois Urbana-Champaign

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Interactive Notebooks](#-interactive-notebooks)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Pipeline Phases](#-pipeline-phases)
- [Configuration](#-configuration)
- [GUI Application](#-gui-application)
- [CLI Usage](#-cli-usage)
- [Outputs](#-outputs)
- [Development](#-development)
- [License](#-license)

---

## 🎯 Overview

This package implements a **four-phase pipeline** for analyzing images using superpixel segmentation and graph-based methods:

| Phase | Name | Description |
|-------|------|-------------|
| **1** | Segmentation | SLIC superpixel segmentation and RAG construction |
| **2** | Enrichment | Feature extraction, edge weighting, and graph pruning |
| **3** | Analysis | Centrality measures, community detection, shortest paths, MST, min-cut |
| **4** | AI Layer | Graph reasoning, anomaly detection, GNN training, LLM hooks |

> 📌 See the [pipeline_viewer.ipynb](notebooks/pipeline_viewer.ipynb) notebook for interactive visualizations of each phase.

---

## ✨ Features

- 🖼️ **SLIC Superpixel Segmentation** - Efficient image oversegmentation
- 🔗 **Region Adjacency Graphs** - Graph-based image representation
- 📊 **Graph Analytics** - Centrality, communities, paths, MST, min-cut
- 🤖 **AI-Powered Insights** - Anomaly detection, reasoning, GNN support
- 💾 **Smart Caching** - Automatic caching of expensive operations
- 📁 **Multi-Format Export** - CSV, Parquet, GraphML, JSON
- 🖥️ **Streamlit GUI** - Interactive web interface
- ⚡ **CLI Interface** - Command-line automation
- ✅ **Tested & Quality** - pytest, ruff, black, GitHub Actions CI

---

## � Interactive Notebooks

Explore the analysis through our Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| **[pipeline_viewer.ipynb](notebooks/pipeline_viewer.ipynb)** | **Main comparative analysis notebook** - Side-by-side comparison of original vs AI-generated images with executive inferences, spectral clustering (Fiedler vector), community detection, centrality analysis, and anomaly detection |
| [network_graph_comparison.ipynb](notebooks/network_graph_comparison.ipynb) | Network graph visualization and comparison utilities |

### Quick Start with Notebooks

```bash
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook pipeline_viewer.ipynb
```

> **📌 Recommended**: Start with `pipeline_viewer.ipynb` for the complete comparative analysis of original histopathology images vs AI-generated images (GPT and Gemini).

---

## �🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/superpixel-rag-analysis.git
cd superpixel-rag-analysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Or using conda
conda create -n superpixel-rag python=3.10
conda activate superpixel-rag
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r superpixel_rag/requirements.txt

# Optional: Louvain community detection
pip install python-louvain

# Optional: GUI application
pip install -r requirements-gui.txt

# Optional: Development tools
pip install -r requirements-dev.txt
```

### Step 4: Install PyTorch (Optional - for GNN Training)

```bash
# CPU only
pip install torch torchvision

# With CUDA (check your version at https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric
pip install torch-geometric
```

---

## ⚡ Quick Start

### Option 1: Python API

```python
from superpixel_rag.config import Config
from superpixel_rag.pipeline import Pipeline

# Create configuration
config = Config.default()
config.image_paths = ['path/to/image.jpg']
config.image_titles = ['My Image']
config.output_dir = './output'

# Optional: Enable AI features
config.phase4.enable_phase4 = True

# Initialize and run pipeline
pipeline = Pipeline(config)
bundle = pipeline.run(config.image_paths[0], title='My Image')

# Access results
print(f"Nodes: {bundle.G.number_of_nodes()}")
print(f"Edges: {bundle.G.number_of_edges()}")
print(f"Communities: {bundle.community_df['community_id'].nunique()}")
```

### Option 2: Command Line

```bash
# Run with default settings
python -m superpixel_rag.main

# Process specific images
python -m superpixel_rag.main --image image1.jpg --image image2.png

# Customize parameters
python -m superpixel_rag.main --n-segments 300 --compactness 15 --output ./results

# Enable AI layer
python -m superpixel_rag.main --enable-phase4
```

### Option 3: GUI Application

```bash
streamlit run apps/gui/app.py
```

Then open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
superpixel-rag-analysis/
├── superpixel_rag/           # Core Python package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration dataclasses
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── data.py               # Data contracts (GraphBundle)
│   ├── export_manager.py     # Centralized export logic
│   ├── utils.py              # Shared utilities & caching
│   ├── visualization.py      # Plotting functions
│   │
│   ├── phase1_segmentation.py  # Phase 1: SLIC + RAG
│   ├── phase2_enrichment.py    # Phase 2: Features + Pruning
│   ├── phase3_analysis.py      # Phase 3: Graph algorithms
│   ├── phase4_ai.py            # Phase 4: AI orchestrator
│   │
│   ├── ai_features.py        # Feature matrix construction
│   ├── ai_reasoning.py       # Reasoning & LLM hooks
│   ├── ai_gnn.py             # GNN training (PyTorch)
│   ├── ai_export.py          # Dataset export utilities
│   │
│   ├── main.py               # CLI entry point
│   └── requirements.txt      # Core dependencies
│
├── apps/
│   └── gui/                  # Streamlit GUI application
│       ├── app.py            # Main app entry
│       ├── components/       # Reusable UI components
│       ├── pages/            # Multi-page app pages
│       └── assets/           # Static assets
│
├── tests/                    # Test suite
│   ├── conftest.py           # Pytest fixtures
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_pipeline.py
│   └── test_export_manager.py
│
├── Original Image/           # Sample input images
│
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI
│
├── .gitignore
├── .pre-commit-config.yaml   # Pre-commit hooks
├── pyproject.toml            # Project metadata
├── pytest.ini                # Pytest configuration
├── requirements-dev.txt      # Development dependencies
├── requirements-gui.txt      # GUI dependencies
└── README.md                 # This file
```

---

## 🔄 Pipeline Phases

### Phase 1: Segmentation

- **SLIC Algorithm**: Clusters pixels into superpixels based on color and spatial proximity
- **RAG Construction**: Builds a Region Adjacency Graph connecting neighboring superpixels
- **Caching**: Results are cached for fast re-runs

### Phase 2: Enrichment

- **Region Features**: Extracts area, perimeter, mean color, texture, shape descriptors
- **Edge Weights**: Computes edge weights based on color distance, boundary contrast
- **Graph Pruning**: Removes weak edges, merges small regions

### Phase 3: Analysis

- **Centrality Measures**: Betweenness, closeness, eigenvector, degree centrality
- **Community Detection**: Louvain or greedy modularity algorithms
- **Path Analysis**: Shortest paths between regions
- **MST**: Minimum Spanning Tree extraction
- **Min-Cut**: Graph partitioning with seed nodes

### Phase 4: AI Layer

- **Feature Matrices**: Constructs node/edge feature tensors for ML
- **Anomaly Detection**: Z-score based anomaly scoring
- **Graph Digest**: Structured statistics for reasoning
- **LLM Hooks**: Integration points for language model reasoning
- **GNN Training**: Optional Graph Neural Network training (PyTorch Geometric)

---

## ⚙️ Configuration

### Main Configuration

```python
from superpixel_rag.config import Config

config = Config.default()

# General settings
config.output_dir = './output'
config.use_cache = True
config.cache_dir = './cache'
config.save_plots = True
config.show_plots = False
```

### Phase 1 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_segments` | 200 | Number of superpixels |
| `compactness` | 10.0 | Color vs space trade-off |
| `sigma` | 1.0 | Gaussian smoothing |
| `rag_mode` | 'distance' | Edge weight mode |

### Phase 2 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_weight_mode` | 'mean_color_distance' | Weight computation method |
| `prune_mode` | 'threshold' | Pruning strategy |
| `prune_threshold` | 0.2 | Edge removal threshold |

### Phase 3 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `centrality_weight` | 'weight2' | Weight attr for centrality |
| `community_method` | 'greedy' | Community detection algorithm |
| `path_source` / `path_target` | None | Auto-select if None |

### Phase 4 Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase4` | True | Enable AI layer |
| `ai_mode` | 'hybrid' | 'reasoning_only', 'gnn_only', 'hybrid' |
| `train_enabled` | False | Train GNN model |
| `llm_enabled` | False | Enable LLM hooks |

---

## 🖥️ GUI Application

The Streamlit-based GUI provides an interactive interface for:

- **Running the Pipeline**: Upload images and configure parameters
- **Viewing Results**: Browse segmentation overlays and graphs
- **Graph Analysis**: Explore centrality, communities, paths
- **AI Insights**: View anomalies and reasoning summaries
- **Exports**: Download all generated files

### Running the GUI

```bash
# Install GUI dependencies
pip install -r requirements-gui.txt

# Start the application
streamlit run apps/gui/app.py
```

---

## 💻 CLI Usage

```bash
# Basic usage
python -m superpixel_rag.main --image path/to/image.jpg

# Multiple images
python -m superpixel_rag.main --image img1.jpg --image img2.png

# Full configuration
python -m superpixel_rag.main \
    --image image.jpg \
    --n-segments 300 \
    --compactness 15 \
    --output ./results \
    --enable-phase4 \
    --no-visualize \
    --no-cache
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--image` | Input image path(s) |
| `--output` | Output directory |
| `--n-segments` | Number of superpixels |
| `--compactness` | SLIC compactness |
| `--enable-phase4` | Enable AI layer |
| `--no-visualize` | Skip plot generation |
| `--no-cache` | Disable caching |

---

## 📤 Outputs

The pipeline generates the following outputs:

### Data Files

| File | Format | Description |
|------|--------|-------------|
| `rag_nodes.csv` | CSV | Node attributes |
| `rag_edges.csv` | CSV | Edge attributes |
| `centrality.csv` | CSV/Parquet | Centrality measures |
| `communities.csv` | CSV | Community assignments |
| `bottleneck_edges.csv` | CSV | Top bottleneck edges |
| `mst.graphml` | GraphML | Minimum Spanning Tree |
| `anomalies.csv` | CSV | Anomaly scores |

### Graph Files

| File | Format | Description |
|------|--------|-------------|
| `rag.graphml` | GraphML | Base RAG structure |
| `rag_phase3.graphml` | GraphML | Enriched graph |

### AI Artifacts

| File | Description |
|------|-------------|
| `phase4_digest.json` | Structured graph statistics |
| `phase4_exec_summary.txt` | Executive summary |
| `manifest.json` | Export inventory |

### Visualizations

- Superpixel overlays
- RAG graph overlays
- Centrality heatmaps
- Community coloring
- Shortest path visualization
- MST visualization
- Anomaly highlighting

---

## 🧪 Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=superpixel_rag --cov-report=html

# Run specific test
pytest tests/test_pipeline.py -v
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy superpixel_rag/

# Pre-commit hooks (install once)
pre-commit install
```

### CI/CD

This project uses GitHub Actions for continuous integration. On every push:

1. ✅ Runs tests on Python 3.10 and 3.11
2. ✅ Checks code formatting with black
3. ✅ Runs linting with ruff
4. ✅ Generates coverage reports

---

## 📊 Sample Results

For interactive visualizations and comparative analysis results, see the **[pipeline_viewer.ipynb](notebooks/pipeline_viewer.ipynb)** notebook, which includes:

| Analysis | Description |
|----------|-------------|
| **Superpixel Segmentation** | RAG overlay on original vs AI-generated images |
| **Community Detection** | Louvain community clustering visualization |
| **Spectral Clustering** | Fiedler vector bipartition analysis |
| **Centrality Analysis** | Betweenness centrality comparison |
| **Anomaly Detection** | Structural anomaly identification |

> 💡 Run the notebook locally or view it on GitHub to see all visualizations.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- University of Illinois Urbana-Champaign
- scikit-image for SLIC implementation
- NetworkX for graph algorithms
- PyTorch Geometric for GNN support

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ at UIUC
</p>
