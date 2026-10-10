# src/validation/report_generator.py
"""Generate comprehensive analytical PDF reports for MLOps tracking.

Visualize the non-smooth subgradient descent trends from fmin.xyz
optimization along with multi-objective reinforcement learning rewards.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


class MLOpsReportGenerator:
    """Automated report engine for saving mathematical metrics logs."""

    def __init__(self, output_dir: str = "outputs"):
        """Initialize the generator and prepare the output directory."""
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_pdf_report(
        self,
        tda_history: list,
        rl_rewards: list,
        loss_history: list,
        filename: str = "pipeline_report.pdf",
    ) -> str:
        """Generate a multi-panel validation plot saved as high-res PDF.

        Args:
            tda_history: History logs of persistent homology cavities.
            rl_rewards: Rewards trend from the REINFORCE chemical agent.
            loss_history: Optimization loss history logs.
            filename: Destination name of the PDF report file.

        Returns:
            String filesystem path to the generated document.
        """
        filepath = os.path.join(self.output_dir, filename)

        # Создаем мульти-панельный график (Паттерны hse26)
        fig, axs = plt.subplots(3, 1, figsize=(10, 14))
        fig.suptitle(
            "TopoTime-Diffusion: Отчет оптимизации и RL Drug Discovery",
            fontsize=16,
            fontweight="bold",
        )

        # 1. График динамики объема кармана TDA
        axs[0].plot(
            tda_history,
            color="teal",
            linewidth=2,
            marker="o",
            label="Суммарная персистентность H₂",
        )
        axs[0].set_title(
            "1. Лог топологических полостей (Subgradient Descent)"
        )
        axs[0].set_xlabel("Временной шаг (Фреймы молекулярной динамики)")
        axs[0].set_ylabel("Объем полости (Persistence Lifetime)")
        axs[0].grid(True, linestyle="--")
        axs[0].legend()

        # 2. График сходимости Reward функции REINFORCE агента
        axs[1].plot(
            rl_rewards,
            color="darkorange",
            linewidth=2,
            linestyle="-",
            label="Многокритериальный Reward (RDKit+QED)",
        )
        axs[1].axhline(
            y=0.5, color="red", linestyle=":", label="Порог легитимности"
        )
        axs[1].set_title("2. Сходимость целевой функции награды R(τ)")
        axs[1].set_xlabel("Итерация обучения Policy Gradient")
        axs[1].set_ylabel("Значение функции вознаграждения")
        axs[1].grid(True, linestyle="--")
        axs[1].legend()

        # 3. График минимизации функции потерь REINFORCE Loss
        axs[2].plot(
            loss_history,
            color="crimson",
            linewidth=2,
            label="- log π_θ(a_t | s_t) * R(τ)",
        )
        axs[2].set_title("3. Отрицательный логарифм правдоподобия")
        axs[2].set_xlabel("Эпоха градиентного спуска")
        axs[2].set_ylabel("Значение лосса")
        axs[2].grid(True, linestyle="--")
        axs[2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.savefig(filepath, format="pdf", dpi=300)
        plt.close()

        print(f"[MLOps] PDF-отчет успешно сохранен на диск: {filepath}")
        return filepath


if __name__ == "__main__":
    generator = MLOpsReportGenerator()
    generator.generate_pdf_report(
        tda_history=list(np.abs(np.sin(np.linspace(0, 5, 10))) + 0.01),
        rl_rewards=list(
            np.linspace(0.1, 0.8, 10) + np.random.normal(0, 0.05, 10)
        ),
        loss_history=list(
            np.linspace(5.0, 1.2, 10) + np.random.normal(0, 0.2, 10)
        ),
    )
