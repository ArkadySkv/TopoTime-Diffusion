# tests/test_pipeline.py
"""Provide integration tests for the TopoTime-Diffusion preprocessing layers.

Verify the mathematical and geometric consistency of DSP filtering,
persistence homology extractions, and radius graph construction.
"""

import numpy as np
from darts import TimeSeries
import pytest
import torch

from src.preprocessing.dataloader import ProteinTrajectoryLoader
from src.preprocessing.graph_builder import ProteinGraphBuilder
from src.preprocessing.topological_analyzer import ProteinTopologicalAnalyzer


@pytest.fixture
def mock_trajectory():
    """Generate synthetic 3D atom trajectory for testing."""
    n_steps, n_atoms = 20, 10
    return np.random.normal(loc=10.0, scale=1.5, size=(n_steps, n_atoms, 3))


def test_dataloader_dsp_filter(mock_trajectory):
    """Verify DSP filtering and Darts TimeSeries encapsulation."""
    loader = ProteinTrajectoryLoader(fs=1.0, cutoff_idx=4)
    series = loader.process_trajectory_to_darts(mock_trajectory)

    assert isinstance(series, TimeSeries)
    assert series.n_components == mock_trajectory.shape[1] * 3


def test_topological_analyzer(mock_trajectory):
    """Verify TDA persistence homology computations using giotto-tda."""
    analyzer = ProteinTopologicalAnalyzer(
        max_edge_length=10.0, homologies=(2,)
    )
    diagrams = analyzer.compute_persistence_diagrams(mock_trajectory)
    lifetimes = analyzer.get_pocket_lifetime_metrics(diagrams)

    assert diagrams.ndim == 3
    assert len(lifetimes) == mock_trajectory.shape[0]
    assert np.all(lifetimes >= 0)


def test_graph_builder():
    """Verify PyG spatial radius graph construction under cutoff threshold."""
    mock_coords = np.random.uniform(0, 10, size=(15, 3))
    mock_types = np.random.randint(0, 20, size=15)

    builder = ProteinGraphBuilder(radius_threshold=8.0)
    graph = builder.build_frame_graph(mock_coords, mock_types)

    assert graph.num_nodes == 15
    assert hasattr(graph, "edge_index")
    assert graph.pos.shape == (15, 3)
