# tests/test_pipeline.py
import pytest
import numpy as np
import torch
from darts import TimeSeries

from src.preprocessing.dataloader import ProteinTrajectoryLoader
from src.preprocessing.topological_analyzer import ProteinTopologicalAnalyzer
from src.preprocessing.graph_builder import ProteinGraphBuilder

@pytest.fixture
def mock_trajectory():
    """Фикстура для генерации синтетической 3D траектории движения атомов белка"""
    n_steps, n_atoms = 20, 10
    return np.random.normal(loc=10.0, scale=1.5, size=(n_steps, n_atoms, 3))

def test_dataloader_dsp_filter(mock_trajectory):
    """Тест Шага 1: Проверка ЦОС фильтрации и создания объекта Darts TimeSeries"""
    loader = ProteinTrajectoryLoader(fs=1.0, cutoff_idx=4)
    series = loader.process_trajectory_to_darts(mock_trajectory)
    
    assert isinstance(series, TimeSeries), "Выход должен быть объектом Darts TimeSeries"
    assert series.n_components == mock_trajectory.shape[1] * 3, "Неверное количество компонентов временного ряда"

def test_topological_analyzer(mock_trajectory):
    """Тест Шага 2: Проверка TDA вычислений персистентных гомологий"""
    analyzer = ProteinTopologicalAnalyzer(max_edge_length=10.0, homologies=(2,))
    diagrams = analyzer.compute_persistence_diagrams(mock_trajectory)
    lifetimes = analyzer.get_pocket_lifetime_metrics(diagrams)
    
    assert diagrams.ndim == 3, "Матрица персистентных диаграмм должна быть трехмерной"
    assert len(lifetimes) == mock_trajectory.shape[0], "Длина профиля стабильности должна быть равна числу шагов времени"
    assert np.all(lifetimes >= 0), "Время жизни вирусного кармана не может быть отрицательным"

def test_graph_builder():
    """Тест Шага 3: Проверка сборки 3D молекулярного графа через PyTorch Geometric"""
    mock_coords = np.random.uniform(0, 10, size=(15, 3))
    mock_types = np.random.randint(0, 20, size=15)
    
    builder = ProteinGraphBuilder(radius_threshold=8.0)
    graph = builder.build_frame_graph(mock_coords, mock_types)
    
    assert graph.num_nodes == 15, "Количество узлов графа должно быть равно количеству атомов"
    assert hasattr(graph, 'edge_index'), "Граф должен содержать матрицу инцидентности ребер"
    assert graph.pos.shape == (15, 3), "Матрица 3D координат повреждена или имеет неверную размерность"
