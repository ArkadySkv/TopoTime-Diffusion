# src/models/rl_drug_designer.py
"""Provide reinforcement learning agent for generative drug design.

Utilize Policy Gradient (REINFORCE) to optimize SMILES sequences targeted
at binding pockets discovered by the TopoTime-Diffusion framework.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import torch
import torch.nn as nn
import torch.nn.functional as F

# Импорт детерминированного валидатора геометрии кармана
from src.validation.cheminformatics_validator import ProteinLigandValidator


class MolecularRNN(nn.Module):
    """Recurrent neural network (GRU) for character-by-character generation."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        """Initialize embedding, GRU cells and projection layers.

        Mathematical foundation matches Markov processes of sequential
        generative scripts.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """Perform a forward pass through the RNN layers."""
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        logits = self.fc(out)
        return logits, hidden


class ProductionRLDrugDesigner:
    """Manage the policy gradient loop for chemical compound generation."""

    def __init__(self, vocabulary: list, lr: float = 1e-4):
        """Set up model token tables, networks and MLOps optimizer."""
        self.vocab = vocabulary
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        self.model = MolecularRNN(vocab_size=self.vocab_size)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.validator = ProteinLigandValidator()

    def sample_molecule(self, max_len: int = 60, device: str = "cpu"):
        """Sample a chemical trajectory string from the policy distribution.

        Keep tracking of the computational graph gradients for autograd.
        """
        self.model.to(device)
        self.model.train()

        input_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        hidden = None

        generated_chars = []
        log_probabilities = []

        for _ in range(max_len):
            logits, hidden = self.model(input_idx, hidden)
            probs = F.softmax(logits.squeeze(1), dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            char_idx = action.item()
            if char_idx == self.char_to_idx.get("$", -1):
                break

            generated_chars.append(self.idx_to_char[char_idx])
            log_probabilities.append(dist.log_prob(action))

            input_idx = action.view(1, 1)

        smiles = "".join(generated_chars)
        return smiles, log_probabilities

    def evaluate_multi_objective_reward(
        self, smiles: str, pocket_center: np.ndarray
    ) -> float:
        """Calculate multi-objective reward function for deep inhibitors.

        Evaluate 3D pocket matching score, drug-likeness (QED), and mass.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0

        try:
            # 1. Скор соответствия 3D геометрии кармана
            spatial_score = self.validator.evaluate_ligand_fit(
                smiles, pocket_center, search_radius=7.5
            )

            # 2. Расчет QED (Quantitative Estimate of Drug-likeness)
            qed_score = QED.qed(mol)

            # 3. Фильтрация молекулярного веса тяжелых ингибиторов (>500 Da)
            mol_weight = Descriptors.MolWt(mol)
            weight_bonus = 1.0 if mol_weight > 500 else (mol_weight / 500.0)

            total_reward = (
                0.5 * spatial_score + 0.3 * qed_score + 0.2 * weight_bonus
            )
            return max(0.1, total_reward)

        except Exception:
            return 0.05

    def update_policy(self, log_probs, reward: float):
        """Execute standard REINFORCE policy gradient optimization step."""
        if len(log_probs) == 0:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        log_probs_tensor = torch.stack(log_probs)
        loss = -torch.sum(log_probs_tensor) * reward

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


def train_production_rl_loop(pocket_center: np.ndarray, iterations: int = 10):
    """Run an isolated policy gradient optimization cycle for tests."""
    print("=== Запуск Продакшн-контура REINFORCE Policy Gradient ===")

    vocab = [
        "C", "O", "N", "S", "F", "Cl", "c", "n", "s",
        "(", ")", "=", "1", "2", "3", "$",
    ]
    designer = ProductionRLDrugDesigner(vocabulary=vocab)

    for i in range(iterations):
        smiles, log_probs = designer.sample_molecule()

        if not smiles:
            smiles = "CC(=O)NC1=CC=C(C=C1)O"

        reward = designer.evaluate_multi_objective_reward(smiles, pocket_center)
        loss_val = designer.update_policy(log_probs, reward)

        test_mol = Chem.MolFromSmiles(smiles)
        mol_wt = Descriptors.MolWt(test_mol) if test_mol else 0.0

        print(
            f"[Итерация {i+1}/{iterations}] Молекула: {smiles} | "
            f"Вес: {mol_wt:.1f} | REWARD: {reward:.4f} | LOSS: {loss_val:.4f}"
        )


if __name__ == "__main__":
    mock_pocket_center = np.array([14.1, 11.2, 8.5])
    train_production_rl_loop(mock_pocket_center)

