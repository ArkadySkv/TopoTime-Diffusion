# inference.py
"""Run standalone inference pipeline for TopoTime-Diffusion backbones.

Load pre-trained weights from checkpoints and generate stable 3D atomic
conformations under the guidance of times-series trends and topology.
"""

import os
import numpy as np
import torch
from src.models.topological_diffusion import TopoTimeDiffusionEngine


def run_standalone_inference(
    checkpoint_name: str = "topotime_diffusion_latest.pt",
):
    """Generate 3D viral protein conformations using saved model weights."""
    print("=== Запуск автономного инференса TopoTime-Diffusion ===")
    checkpoint_path = os.path.join("checkpoints", checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Критическая ошибка: Веса модели не обнаружены по пути "
            f"{checkpoint_path}. Сначала запустите обучение через main.py"
        )

    n_atoms = 40
    timesfm_horizon = 16

    # 1. Инициализируем генеративный движок диффузии Point-E
    engine = TopoTimeDiffusionEngine(
        d_model=64, timesfm_horizon=timesfm_horizon
    )

    # 2. Подгружаем сохраненный мозг модели из чекпоинта
    engine.load_checkpoint(checkpoint_path)

    # 3. Имитируем входящий предикт кинетики вируса от Google TimesFM 3.0
    mock_future_trend = np.random.uniform(0.8, 1.5, size=timesfm_horizon)

    # 4. Генерируем чистое облако атомов из случайного гауссова шума
    x_noisy = torch.randn(1, n_atoms, 3)

    print("[Inference] Развертывание марковской цепи обратной диффузии...")
    x_generated = x_noisy

    for _ in range(20):
        x_generated = engine.denoise_step(
            x_generated, mock_future_trend, t=0.5, lambda_topo=0.02
        )

    final_coords = x_generated.squeeze(0).detach().cpu().numpy()
    print(
        f"[Inference] Генерация завершена. Сформирована 3D конформация. "
        f"Матрица координат: {final_coords.shape}"
    )
    return final_coords


if __name__ == "__main__":
    run_standalone_inference()
