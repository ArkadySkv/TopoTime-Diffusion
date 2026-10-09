import numpy as np
from darts import TimeSeries
from gtda.homology import VietorisRipsPersistence  # Импорт движка TDA из giotto-tda

class ProteinTopologicalAnalyzer:
    def __init__(self, max_edge_length: float = 10.0, homologies: tuple = (0, 1, 2)):
        """
        Инициализация топологического процессора.
        
        :param max_edge_length: Максимальное расстояние (в Ангстремах) для построения симплексов.
        :param homologies: Извлекаемые типы гомологий. 
                           0 - компоненты связности, 1 - петли/туннели, 2 - полости/карманы (H2).
        """
        self.homology_dimensions = homologies
        # Инициализируем пайплайн фильтрации Виеториса-Рипса на базе Ripser
        self.vr = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            max_edge_length=max_edge_length,
            n_jobs=-1 # Используем все доступные ядра процессора
        )

    def extract_trajectory_coordinates(self, series: TimeSeries, n_atoms: int) -> np.ndarray:
        """
        Вспомогательный метод: восстанавливает матрицу (N_steps, N_atoms, 3) из плоского ряда Darts.
        """
        df_values = series.values() # Получаем чистый numpy массив из объекта Darts
        n_steps = df_values.shape[0]
        return df_values.reshape(n_steps, n_atoms, 3)

    def compute_persistence_diagrams(self, trajectory_3d: np.ndarray) -> np.ndarray:
        """
        Вычисляет персистентные диаграммы для каждого временного фрейма траектории.
        
        :param trajectory_3d: Массив (N_steps, N_atoms, 3) с координатами остова белка.
        :return: Массив диаграмм персистентности формы (N_steps, M_features, 3).
                 Каждая строка фичи содержит: [birth, death, homology_dimension]
        """
        print(f"[Шаг 2] Расчет персистентных гомологий для {trajectory_3d.shape[0]} кадров динамики белка...")
        
        # giotto-tda нативно принимает трехмерные облака точек (батч из N_steps кадров)
        persistence_diagrams = self.vr.fit_transform(trajectory_3d)
        
        print("[Шаг 2] Расчет TDA диаграмм успешно завершен.")
        return persistence_diagrams

    def get_pocket_lifetime_metrics(self, diagrams: np.ndarray) -> np.ndarray:
        """
        Метрика оценки объема/стабильности скрытых карманов.
        Вычисляет суммарное "время жизни" (persistence = death - birth) для пустот размерности H2.
        """
        n_steps = diagrams.shape[0]
        pocket_lifetimes = np.zeros(n_steps)

        for step in range(n_steps):
            step_diagram = diagrams[step]
            # Фильтруем только топологические признаки, относящиеся к трехмерным полостям (H2 = 2)
            h2_features = step_diagram[step_diagram[:, 2] == 2]
            
            if len(h2_features) > 0:
                # Время жизни полости: момент исчезновения минус момент зарождения
                lifetimes = h2_features[:, 1] - h2_features[:, 0]
                # Суммируем персистентность всех найденных полостей в этом кадре
                pocket_lifetimes[step] = np.sum(lifetimes)
                
        return pocket_lifetimes

if __name__ == "__main__":
    from dataloader import ProteinTrajectoryLoader
    
    # 1. Генерируем сырые данные и прогоняем через (ЦОС + Darts)
    n_steps, n_atoms = 10, 40
    mock_raw_data = np.random.normal(loc=10.0, scale=3.0, size=(n_steps, n_atoms, 3))
    
    loader = ProteinTrajectoryLoader(fs=1.0, cutoff_idx=4)
    darts_series = loader.process_trajectory_to_darts(mock_raw_data)
    
    # 2. Передаем результат в Шаг 2 (Topological ML)
    analyzer = ProteinTopologicalAnalyzer(max_edge_length=12.0, homologies=(0, 1, 2))
    
    # Восстанавливаем 3D структуру и считаем топологию
    trajectory_3d = analyzer.extract_trajectory_coordinates(darts_series, n_atoms=n_atoms)
    diagrams = analyzer.compute_persistence_diagrams(trajectory_3d)
    
    # Считаем профиль стабильности скрытых карманов
    pocket_metrics = analyzer.get_pocket_lifetime_metrics(diagrams)
    
    print(f"Персистентные диаграммы рассчитаны. Форма: {diagrams.shape}")
    print(f"Профиль стабильности вирусных карманов по кадрам (H2 сума персистентности): \n{pocket_metrics}")
