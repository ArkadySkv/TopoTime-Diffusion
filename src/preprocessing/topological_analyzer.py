# src/preprocessing/topological_analyzer.py
"""Analyze topological features of protein trajectory data frames.

Extract persistence diagrams and calculate the lifetime stability metrics
of H2 cavities representing latent binding pockets using giotto-tda.
"""

from darts import TimeSeries
from gtda.homology import VietorisRipsPersistence
import numpy as np


class ProteinTopologicalAnalyzer:
    """Compute persistence homology metrics for molecular dynamics."""

    def __init__(
        self, max_edge_length: float = 10.0, homologies: tuple = (0, 1, 2)
    ):
        """Initialize Vietoris-Rips persistence filtration engine.

        Args:
            max_edge_length: Maximum filtration distance in Angstroms.
            homologies: Tuple of target homology dimensions to extract.
                0 - components, 1 - loops, 2 - voids/pockets (H2).
        """
        self.homology_dimensions = homologies
        self.vr = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            max_edge_length=max_edge_length,
            n_jobs=-1,
        )

    def extract_trajectory_coordinates(
        self, series: TimeSeries, n_atoms: int
    ) -> np.ndarray:
        """Reconstruct 3D matrix array from flat darts TimeSeries objects."""
        df_values = series.values()
        n_steps = df_values.shape[0]
        return df_values.reshape(n_steps, n_atoms, 3)

    def compute_persistence_diagrams(
        self, trajectory_3d: np.ndarray
    ) -> np.ndarray:
        """Compute persistence diagrams for each batch trajectory frame.

        Args:
            trajectory_3d: Array of shape (N_steps, N_atoms, 3).

        Returns:
            Persistence feature diagrams array of shape (N_steps, M, 3).
        """
        print(
            f"[Шаг 2] Расчет персистентных гомологий для "
            f"{trajectory_3d.shape[0]} кадров динамики белка..."
        )

        persistence_diagrams = self.vr.fit_transform(trajectory_3d)

        print("[Шаг 2] Расчет TDA диаграмм успешно завершен.")
        return persistence_diagrams

    def get_pocket_lifetime_metrics(self, diagrams: np.ndarray) -> np.ndarray:
        """Calculate total persistence lifetime of H2 cavities/voids."""
        n_steps = diagrams.shape[0]
        pocket_lifetimes = np.zeros(n_steps)

        for step in range(n_steps):
            step_diagram = diagrams[step]
            # Отбираем признаки трехмерных полостей кармана (H2 = 2)
            h2_features = step_diagram[step_diagram[:, 2] == 2]

            if len(h2_features) > 0:
                lifetimes = h2_features[:, 1] - h2_features[:, 0]
                pocket_lifetimes[step] = np.sum(lifetimes)

        return pocket_lifetimes


if __name__ == "__main__":
    from dataloader import ProteinTrajectoryLoader

    n_steps, n_atoms = 10, 40
    mock_raw_data = np.random.normal(
        loc=10.0, scale=3.0, size=(n_steps, n_atoms, 3)
    )

    loader = ProteinTrajectoryLoader(fs=1.0, cutoff_idx=4)
    darts_series = loader.process_trajectory_to_darts(mock_raw_data)

    analyzer = ProteinTopologicalAnalyzer(
        max_edge_length=12.0, homologies=(0, 1, 2)
    )
    trajectory_3d = analyzer.extract_trajectory_coordinates(
        darts_series, n_atoms=n_atoms
    )
    diagrams = analyzer.compute_persistence_diagrams(trajectory_3d)
    pocket_metrics = analyzer.get_pocket_lifetime_metrics(diagrams)

    print(f"Персистентные диаграммы рассчитаны. Форма: {diagrams.shape}")
    print(f"Профиль стабильности вирусных карманов: \n{pocket_metrics}")
