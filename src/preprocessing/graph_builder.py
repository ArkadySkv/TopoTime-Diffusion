# src/preprocessing/graph_builder.py
"""Build equivariant spatial 3D graphs from protein atomic coordinates.

Transform static point cloud frames into unified PyTorch Geometric Data
structures using radius-based spatial cutoff criteria.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph


class ProteinGraphBuilder:
    """Processor for constructing 3D spatial graphs of proteins."""

    def __init__(self, radius_threshold: float = 5.0):
        """Initialize graph builder with cutoff radius in Angstroms.

        Args:
            radius_threshold: Maximum distance to connect two atoms.
        """
        self.radius_threshold = radius_threshold

    def build_frame_graph(
        self, atom_coordinates: np.ndarray, residue_types: list = None
    ) -> Data:
        """Convert static point cloud frame into a PyG Data object.

        Args:
            atom_coordinates: Array of shape (N_atoms, 3) with positions.
            residue_types: List of integer amino acid identifiers.

        Returns:
            Unified torch_geometric.data.Data graph object.
        """
        pos = torch.tensor(atom_coordinates, dtype=torch.float32)
        n_atoms = pos.size(0)

        # Инициализируем базовые признаки узлов (аминокислот)
        if residue_types is None:
            x = torch.ones((n_atoms, 1), dtype=torch.float32)
        else:
            x = torch.tensor(residue_types, dtype=torch.long).view(-1, 1)

        # Вычисляем ребра по радиусу отсечения (Stanford CS224W)
        edge_index = radius_graph(
            pos, r=self.radius_threshold, loop=False
        )

        graph_data = Data(x=x, pos=pos, edge_index=edge_index)
        return graph_data


if __name__ == "__main__":
    # Тестовый прогон: 50 ключевых атомов остова вирусного белка
    mock_frame_coords = np.random.uniform(
        low=0.0, high=20.0, size=(50, 3)
    )
    mock_residues = np.random.randint(0, 20, size=50)

    builder = ProteinGraphBuilder(radius_threshold=6.0)
    graph = builder.build_frame_graph(mock_frame_coords, mock_residues)

    print("Молекулярный граф успешно собран.")
    print(f"Количество узлов (атомов): {graph.num_nodes}")
    print(f"Количество связей (ребер) в 3D: {graph.num_edges}")
    print(f"Форма тензора позиций: {graph.pos.shape}")
