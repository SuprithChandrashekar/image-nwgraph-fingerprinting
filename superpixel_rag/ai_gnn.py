"""
AI GNN Module

Optional GNN training using PyTorch and PyTorch Geometric.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Optional imports
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class GCN(torch.nn.Module if TORCH_AVAILABLE else object):  # type: ignore
    def __init__(self, num_features, num_classes, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x  # Penultimate layer embeddings
        x = self.conv2(x, edge_index)
        return x, embeddings

def train_gnn(
    X_nodes: np.ndarray,
    edge_index_pairs: np.ndarray,
    y: Optional[np.ndarray] = None,
    epochs: int = 50,
    hidden_channels: int = 64,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a simple GCN if PyTorch/PyG are available.
    
    Parameters
    ----------
    X_nodes : np.ndarray
        Node features.
    edge_index_pairs : np.ndarray
        Edge indices (2, E).
    y : np.ndarray, optional
        Labels.
        
    Returns
    -------
    Dict containing embeddings, predictions, and metrics.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch or PyTorch Geometric not found. Skipping GNN training.")
        return {}
        
    if y is None:
        logger.info("No labels provided. Skipping GNN training.")
        return {}

    # Set seed
    torch.manual_seed(random_state)
    
    # Prepare data
    x = torch.tensor(X_nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index_pairs.T, dtype=torch.long) # (2, E)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    num_classes = len(np.unique(y))
    num_features = X_nodes.shape[1]
    
    # Create masks (simple random split)
    num_nodes = X_nodes.shape[0]
    indices = np.random.permutation(num_nodes)
    train_idx = torch.tensor(indices[:int(0.8*num_nodes)], dtype=torch.long)
    test_idx = torch.tensor(indices[int(0.8*num_nodes):], dtype=torch.long)
    
    # Model
    model = GCN(num_features, num_classes, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out, _ = model(x, edge_index)
        loss = F.cross_entropy(out[train_idx], y_tensor[train_idx])
        loss.backward()
        optimizer.step()
        
    # Eval
    model.eval()
    with torch.no_grad():
        out, embeddings = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[test_idx] == y_tensor[test_idx]
        test_acc = int(test_correct.sum()) / int(len(test_idx)) if len(test_idx) > 0 else 0
        
    return {
        'embeddings': embeddings.numpy(),
        'predictions': pred.numpy(),
        'probabilities': F.softmax(out, dim=1).numpy(),
        'test_accuracy': test_acc,
        'model_state': model.state_dict()
    }
