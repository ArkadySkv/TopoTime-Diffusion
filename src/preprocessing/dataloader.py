import numpy as np
import pandas as pd
from darts import TimeSeries
from scipy.signal import stft, istft

class ProteinTrajectoryLoader:
    def __init__(self, fs: float = 1.0, cutoff_idx: int = 4):
        """
        Инициализация загрузчика и препроцессора траекторий.
        
        :param fs: Частота дискретизации временного ряда (фреймы в пикосекунду).
        :param cutoff_idx: Частотный порог для фильтрации теплового шума (ЦОС).
        """
        self.fs = fs
        self.cutoff_idx = cutoff_idx

    def _apply_dsp_filter(self, trajectory_3d: np.ndarray) -> np.ndarray:
        """
        Внутренний метод: спектральная очистка траекторий атомов (ЦОС).
        Убирает высокочастотный тепловой шум на основе алгоритмов STFT.
        """
        n_steps, n_atoms, coords = trajectory_3d.shape
        filtered_trajectory = np.zeros_like(trajectory_3d)
        
        for atom in range(n_atoms):
            for c in range(coords):
                signal = trajectory_3d[:, atom, c]
                
                # Применяем коротковременное преобразование Фурье
                f, t, Zxx = stft(signal, fs=self.fs, nperseg=min(128, n_steps))
                
                # Фильтрация: зануляем высокие частоты (шум флуктуаций)
                Zxx[self.cutoff_idx:, :] = 0
                
                # Восстановление очищенного пространственного сигнала
                _, filtered_signal = istft(Zxx, fs=self.fs)
                filtered_trajectory[:, atom, c] = filtered_signal[:n_steps]
                
        return filtered_trajectory

    def process_trajectory_to_darts(self, raw_trajectory: np.ndarray) -> TimeSeries:
        """
        Основной метод пайплайна Шага 1. 
        Принимает сырую матрицу координат, фильтрует её и оборачивает в многомерный ряд Darts.
        
        :param raw_trajectory: Массив формы (N_steps, N_atoms, 3)
        :return: Объект TimeSeries из библиотеки darts
        """
        print("Запуск ЦОС фильтрации теплового шума...")
        cleaned_trajectory = self._apply_dsp_filter(raw_trajectory)
        
        n_steps, n_atoms, _ = cleaned_trajectory.shape
        # Сплющиваем 3D координаты атомов в плоскую матрицу признаков для временного ряда
        flattened_features = cleaned_trajectory.reshape(n_steps, n_atoms * 3)
        
        # Генерируем читаемые названия колонок для MLOps-логирования
        columns = [f"atom_{i}_{axis}" for i in range(n_atoms) for axis in ['x', 'y', 'z']]
        
        df = pd.DataFrame(flattened_features, columns=columns)
        # Создаем искусственную временную шкалу для совместимости с движком Darts
        df['timestamp'] = pd.date_range(start="2025-12-31", periods=n_steps, freq="s")
        
        print(f"Успешно сформирован TimeSeries. Всего признаков: {len(columns)}")
        return TimeSeries.from_dataframe(df, time_col='timestamp', value_cols=columns)

if __name__ == "__main__":
    print("Генерация симуляции траектории вирусного белка...")
    mock_raw_trajectory = np.random.normal(loc=15.0, scale=2.5, size=(200, 30, 3))
    
    loader = ProteinTrajectoryLoader(fs=1.0, cutoff_idx=4)
    darts_series = loader.process_trajectory_to_darts(mock_raw_trajectory)
    
    print(f"Объект Darts готов. Количество компонентов ряда: {darts_series.n_components}")
