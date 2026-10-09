import pytest
import numpy as np
import torch

from src.models.rl_drug_designer import ProductionRLDrugDesigner

@pytest.fixture
def rl_designer():
    """Фикстура инициализации продакшн RL-дизайнера"""
    chemical_vocab = ['C', 'O', 'N', 'S', 'F', 'Cl', 'c', 'n', 's', '(', ')', '=', '1', '2', '3', '$']
    return ProductionRLDrugDesigner(vocabulary=chemical_vocab)

def test_invalid_smiles_handling_penalty(rl_designer):
    """Тест защиты от невалидного синтаксиса химии (SMILES Parse Error)"""
    mock_pocket_center = np.array([10.0, 10.0, 10.0])
    
    # Симулируем синтаксически некорректную строку (из логов ошибок: лишняя скобка)
    broken_smiles = "Cln)(C)3Cl22Ff"
    
    reward = rl_designer.evaluate_multi_objective_reward(broken_smiles, mock_pocket_center)
    
    # Математический штраф по классике RL должен быть равен -1.0
    assert reward == -1.0, "Невалидный SMILES должен получать жесткий штраф -1.0"

def test_valid_heavy_inhibitor_scoring(rl_designer):
    """Тест расчета многокритериального вознаграждения для легитимной молекулы"""
    mock_pocket_center = np.array([12.0, 11.0, 8.5])
    
    # Берем валидный SMILES парацетамола (PubChem CID 1983)
    valid_smiles = "CC(=O)NC1=CC=C(C=C1)O"
    
    reward = rl_designer.evaluate_multi_objective_reward(valid_smiles, mock_pocket_center)
    
    assert reward > 0, "Валидная молекула должна получать строго положительную награду"
    assert isinstance(reward, float), "Результат вознаграждения должен быть вещественным числом"

def test_reinforce_policy_loss_computation(rl_designer):
    """Тест корректности вычисления градиентного лосса REINFORCE"""
    # Симулируем логарифмы вероятностей для батча из 3 токенов
    mock_log_probs = [
        torch.tensor(-0.5, requires_grad=True),
        torch.tensor(-1.2, requires_grad=True),
        torch.tensor(-0.8, requires_grad=True)
    ]
    mock_reward = 0.75
    
    loss_val = rl_designer.update_policy(mock_log_probs, reward=mock_reward)
    
    # Математика: loss = -(-0.5 - 1.2 - 0.8) * 0.75 = -( -2.5 ) * 0.75 = 1.875
    expected_loss = 1.875
    assert np.isclose(loss_val, expected_loss, atol=1e-3), f"Ожидался лосс {expected_loss}, получено {loss_val}"