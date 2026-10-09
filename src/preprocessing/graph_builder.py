import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph  # Инструмент автоматического построения ребер по радиусу

class ProteinGraphBuilder:
    def __init__(self, radius_threshold: float = 5.0):
        """
        Процессор сборки 3D-графов белковых структур.
        
        :param radius_threshold: Максимальное расстояние в Ангстремах для создания ребра между атомами.
        """
        self.radius_threshold = radius_threshold

    def build_frame_graph(self, atom_coordinates: np.ndarray, residue_types: list = None) -> Data:
        """
        Превращает статическое облако точек (кадр молекулярной динамики) в объект графа PyTorch Geometric.
        
        :param atom_coordinates: Массив формы (N_atoms, 3) — координаты атомов текущего кадра.
        :param residue_types: Список типов аминокислот (целочисленные идентификаторы) длины N_atoms.
        :return: Объект torch_geometric.data.Data
        """
        # Переводим координаты в тензор PyTorch
        pos = torch.tensor(atom_coordinates, dtype=torch.float32)
        n_atoms = pos.size(0)
        
        # Если типы аминокислот не заданы, инициализируем их дефолтными признаками (единичная матрица)
        if residue_types is None:
            # Создаем фичи узлов: размерность (N_atoms, 1) — например, индекс атома или константа
            x = torch.ones((n_atoms, 1), dtype=torch.float32)
        else:
            x = torch.tensor(residue_types, dtype=torch.long).view(-1, 1)
            
        # Строим матрицу инцидентности ребер по радиусу отсечения (концепция Stanford CS224W)
        # Алгоритм находит все пары атомов, расстояние между которыми <= radius_threshold
        edge_index = radius_graph(pos, r=self.radius_threshold, loop=False)
        
        # Собираем унифицированный объект данных PyG графа
        graph_data = Data(x=x, pos=pos, edge_index=edge_index)
        
        return graph_data

if __name__ == "__main__":
    # Тестовый прогон: один кадр молекулярной динамики, содержащий 50 ключевых атомов остова белка
    mock_frame_coords = np.random.uniform(low=0.0, high=20.0, size=(50, 3))
    
    # Симулируем типы 50 аминокислот (индексы от 0 до 19 для 20 стандартных аминокислот)
    mock_residue_types = np.random.randint(0, 20, size=50)
    
    builder = ProteinGraphBuilder(radius_threshold=6.0)
    graph = builder.build_frame_graph(mock_frame_coords, mock_residue_types)
    
    print(f"Молекулярный граф успешно собран.")
    print(f"Количество узлов (атомов): {graph.num_nodes}")
    print(f"Количество связей (ребер) в 3D-пространстве: {graph.num_edges}")
    print(f"Форма тензора позиций: {graph.pos.shape}")
