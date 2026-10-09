import numpy as np
# Импортируем официальный класс прогнозирования для TimesFM 3.0
from timesfm3 import TimesFM3Forecaster

class ProteinKineticsPredictor:
    def __init__(self, context_len: int = 32, horizon_len: int = 16):
        """
        Инициализация предиктора кинетики на базе актуальной модели Google TimesFM 3.0.
        """
        self.context_len = context_len
        self.horizon_len = horizon_len
        
        print("[TimesFM] Загрузка предобученных весов Google TimesFM 3.0 из Hugging Face...")

        # Модель инициализируется напрямую через загрузку весов Hugging Face.
        self.forecaster = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch")

    def predict_future_pocket_trend(self, pocket_lifetime_series: np.ndarray) -> np.ndarray:
        """
        Предсказание будущего поведения вирусного кармана с использованием Zero-Shot возможностей TimesFM 3.0.
        """
        print(f"[TimesFM] Анализ временного ряда. Контекст: {len(pocket_lifetime_series)} кадров...")
        
        # Обрезаем входной ряд под необходимый нам размер контекста
        context = pocket_lifetime_series[-self.context_len:]
        
        # Функция принимает одномерный или многомерный массив контекста напрямую
        output = self.forecaster.predict(
            context=context,
            horizon=self.horizon_len,
        )
        
        # Извлекаем точечный прогноз (медиану/среднее) из результирующего объекта
        future_trend = output.forecast
        
        # Если модель возвращает батч/матрицу, берем первый элемент для нашего одномерного тренда
        if future_trend.ndim > 1:
            future_trend = future_trend[0]
            
        print(f"[TimesFM] Прогноз кинетики успешно сформирован на {self.horizon_len} шагов.")
        return future_trend

if __name__ == "__main__":
    # Тестовая проверка работоспособности обновленного API 3.0
    mock_pocket_lifetimes = np.abs(np.sin(np.linspace(0, 10, 64))) + np.random.normal(0, 0.1, 64)
    
    predictor = ProteinKineticsPredictor(context_len=32, horizon_len=16)
    predicted_trend = predictor.predict_future_pocket_trend(mock_pocket_lifetimes)
    
    print(f"Предсказанный TimesFM 3.0 тренд кармана в будущем: \n{predicted_trend}")
