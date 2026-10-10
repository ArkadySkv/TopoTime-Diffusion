# main.py
"""Orchestrate the end-to-end MLOps pipeline for TopoTime-Diffusion.

Execute data loading, DSP noise filtering, topological analysis,
kinetics forecasting, equivariant graph building, point-cloud diffusion,
and reinforcement learning drug design loop integration.
"""

import hydra
import numpy as np
from omegaconf import DictConfig
import torch
import wandb

# Импорт всех компонентов пайплайна по стандартам PEP 8
from src.models.rl_drug_designer import ProductionRLDrugDesigner
from src.models.timesfm_predictor import ProteinKineticsPredictor
from src.models.topological_diffusion import TopoTimeDiffusionEngine
from src.preprocessing.dataloader import ProteinTrajectoryLoader
from src.preprocessing.graph_builder import ProteinGraphBuilder
from src.preprocessing.topological_analyzer import ProteinTopologicalAnalyzer
from src.validation.cheminformatics_validator import ProteinLigandValidator
from src.validation.report_generator import MLOpsReportGenerator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_pipeline(cfg: DictConfig):
    """Run the complete deep learning drug discovery pipeline workflow."""
    print(f"=== Запуск MLOps пайплайна проекта {cfg.project_name} ===")

    wandb.init(
        project=cfg.project_name,
        name=cfg.experiment_name,
        config=dict(cfg),
    )

    # Симуляция входных данных молекулярной динамики вируса
    n_steps, n_atoms = 100, 40
    raw_trajectory_data = np.random.normal(
        loc=12.0, scale=2.0, size=(n_steps, n_atoms, 3)
    )
    mock_residue_types = np.random.randint(0, 20, size=n_atoms)

    # --- ШАГ 1: Загрузка и ЦОС фильтрация шума ---
    loader = ProteinTrajectoryLoader(
        fs=cfg.dataloader.fs, cutoff_idx=cfg.dataloader.cutoff_idx
    )
    darts_series = loader.process_trajectory_to_darts(raw_trajectory_data)

    # --- ШАГ 2: Топологический анализ данных (TDA) ---
    analyzer = ProteinTopologicalAnalyzer(
        max_edge_length=cfg.topological_analyzer.max_edge_length,
        homologies=tuple(cfg.topological_analyzer.homologies),
    )
    trajectory_3d = analyzer.extract_trajectory_coordinates(
        darts_series, n_atoms=n_atoms
    )
    diagrams = analyzer.compute_persistence_diagrams(trajectory_3d)
    pocket_lifetimes = analyzer.get_pocket_lifetime_metrics(diagrams)

    # --- ШАГ 2.5: Прогнозирование кинетики вируса (TimesFM 3.0) ---
    kinetics_predictor = ProteinKineticsPredictor(
        context_len=cfg.timesfm.context_len,
        horizon_len=cfg.timesfm.horizon_len,
    )
    predicted_future_trend = (
        kinetics_predictor.predict_future_pocket_trend(pocket_lifetimes)
    )

    # --- ШАГ 3: Построение 3D-графов белка (PyG) ---
    graph_builder = ProteinGraphBuilder(
        radius_threshold=cfg.graph_builder.radius_threshold
    )
    current_frame_graph = graph_builder.build_frame_graph(
        trajectory_3d[-1], mock_residue_types
    )

    # --- ШАГ 4: Управляемая диффузия (Point-E движок) ---
    diffusion_engine = TopoTimeDiffusionEngine(
        d_model=cfg.diffusion.d_model,
        timesfm_horizon=cfg.timesfm.horizon_len,
    )
    x_generated = torch.randn(1, n_atoms, 3)
    for _ in range(10):  # Оптимизированный шаг для MVP
        x_generated = diffusion_engine.denoise_step(
            x_generated,
            predicted_future_trend,
            t=0.5,
            lambda_topo=cfg.diffusion.lambda_topo,
        )
    final_protein_positions = x_generated.squeeze(0).detach().cpu().numpy()

    # --- ШАГ 5: Валидация хемоинформатики (RDKit) ---
    validator = ProteinLigandValidator()
    pocket_center = validator.calculate_pocket_center(
        final_protein_positions, list(cfg.validation.pocket_indices)
    )
    fit_score = validator.evaluate_ligand_fit(
        cfg.validation.test_smiles,
        pocket_center=pocket_center,
        search_radius=cfg.validation.search_radius,
    )

    # --- ОБУЧЕНИЕ RL-АГЕНТА REINFORCE С ФИКСАЦИЕЙ МАТЕМАТИЧЕСКИХ ЛОГОВ ---
    print(
        "\n[MLOps] Запуск цикла Policy Gradient "
        "оптимизации тяжелых ингибиторов..."
    )
    chemical_vocab = [
        "C", "O", "N", "S", "F", "Cl", "c", "n", "s",
        "(", ")", "=", "1", "2", "3", "$",
    ]
    rl_designer = ProductionRLDrugDesigner(vocabulary=chemical_vocab)

    rl_rewards_history = []
    rl_loss_history = []

    # Прогоняем полноценный цикл из 10 итераций обучения REINFORCE
    for epoch in range(10):
        smiles_candidate, log_probs = rl_designer.sample_molecule()
        if not smiles_candidate:
            smiles_candidate = cfg.validation.test_smiles

        reward = rl_designer.evaluate_multi_objective_reward(
            smiles_candidate, pocket_center
        )
        loss_val = rl_designer.update_policy(log_probs, reward=reward)

        # Сохраняем логи оптимизации
        rl_rewards_history.append(reward)
        rl_loss_history.append(loss_val)

        wandb.log(
            {"rl/iteration_loss": loss_val, "rl/iteration_reward": reward}
        )
        print(
            f"[RL Обучение] Итерация {epoch+1}/10 | "
            f"Сгенерирован SMILES: {smiles_candidate} -> Reward: {reward:.4f}"
        )

    # --- Генерация финального PDF-отчёта (fmin.xyz стандарты) ---
    report_generator = MLOpsReportGenerator()
    report_generator.generate_pdf_report(
        tda_history=list(pocket_lifetimes[:20]),
        rl_rewards=rl_rewards_history,
        loss_history=rl_loss_history,
        filename="topotime_diffusion_report.pdf",
    )

    # Сохранение чекпоинта весов
    diffusion_engine.save_checkpoint(
        directory=cfg.checkpointing.dir,
        filename=cfg.checkpointing.filename,
        epoch=1,
        score=fit_score,
    )
    print("\n Сквозной продакшн пайплайн успешно завершен!")
    wandb.finish()


if __name__ == "__main__":
    run_pipeline()
