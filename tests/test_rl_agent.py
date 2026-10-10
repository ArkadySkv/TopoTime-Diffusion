# tests/test_rl_agent.py
"""Provide unit tests for the reinforcement learning ligand designer agent.

Validate chemical syntax exception gating and REINFORCE policy loss
gradient convergence vectors using synthetic logs.
"""

import numpy as np
import pytest
import torch

from src.models.rl_drug_designer import ProductionRLDrugDesigner


@pytest.fixture
def rl_designer():
    """Initialize a production-ready REINFORCE chemical designer agent."""
    chemical_vocab = [
        "C", "O", "N", "S", "F", "Cl", "c", "n", "s",
        "(", ")", "=", "1", "2", "3", "$",
    ]
    return ProductionRLDrugDesigner(vocabulary=chemical_vocab)


def test_invalid_smiles_handling_penalty(rl_designer):
    """Verify strict mathematical fallback penalty for incorrect chemistry."""
    mock_pocket_center = np.array([10.0, 10.0, 10.0])
    broken_smiles = "Cln)(C)3Cl22Ff"

    reward = rl_designer.evaluate_multi_objective_reward(
        broken_smiles, mock_pocket_center
    )

    assert reward == -1.0


def test_valid_heavy_inhibitor_scoring(rl_designer):
    """Verify multi-objective scoring for chemically valid structures."""
    mock_pocket_center = np.array([12.0, 11.0, 8.5])
    valid_smiles = "CC(=O)NC1=CC=C(C=C1)O"

    reward = rl_designer.evaluate_multi_objective_reward(
        valid_smiles, mock_pocket_center
    )

    assert reward > 0
    assert isinstance(reward, float)


def test_reinforce_policy_loss_computation(rl_designer):
    """Verify correct calculation of the REINFORCE policy loss vector."""
    mock_log_probs = [
        torch.tensor(-0.5, requires_grad=True),
        torch.tensor(-1.2, requires_grad=True),
        torch.tensor(-0.8, requires_grad=True),
    ]
    mock_reward = 0.75

    loss_val = rl_designer.update_policy(mock_log_probs, reward=mock_reward)

    expected_loss = 1.875
    assert np.isclose(loss_val, expected_loss, atol=1e-3)
