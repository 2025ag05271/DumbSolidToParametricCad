"""UV-Net style face-adjacency graph features for the pipeline.

The sampling and graph construction live in ``step_to_graph.py`` (the script version of
STEP_to_UVNet_Graph.ipynb) and are re-exported here, so the notebook logic, dataset generation,
training data and prediction all use one implementation. Output is pure numpy (no DGL needed in
generator worker processes); ``to_dgl`` turns the arrays into a DGLGraph when DGL is available.
"""

from __future__ import annotations

import numpy as np

from step_to_graph import (  # noqa: F401  (re-exported)
    MAX_RANGE,
    brep_to_graph_arrays,
    is_connected,
    sample_edge_uv_grid,
    sample_face_uv_grid,
)


def to_dgl(arrays: dict[str, np.ndarray], flatten: bool = True):
    """DGLGraph with ndata['x'] / edata['x'].

    ``flatten=True`` matches the notebook layout (F, N*N*7) / (E, N*6); ``flatten=False``
    keeps the grids (F, N, N, 7) / (E, N, 6) as expected by UV-Net style CNN encoders.
    """
    import dgl
    import torch

    node_x, edge_index, edge_x = arrays["node_x"], arrays["edge_index"], arrays["edge_x"]
    g = dgl.graph((torch.from_numpy(edge_index[0]), torch.from_numpy(edge_index[1])),
                  num_nodes=node_x.shape[0])
    if flatten:
        node_x = node_x.reshape(node_x.shape[0], -1)
        edge_x = edge_x.reshape(edge_x.shape[0], -1)
    g.ndata["x"] = torch.from_numpy(np.ascontiguousarray(node_x, dtype=np.float32))
    g.edata["x"] = torch.from_numpy(np.ascontiguousarray(edge_x, dtype=np.float32))
    return g
