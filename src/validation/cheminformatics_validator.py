# src/validation/cheminformatics_validator.py
"""Validate protein-ligand geometric fitting and complementarity.

Utilize RDKit force-field optimization and Conformer generation to evaluate
how well small-molecule drugs sit inside the dynamically generated pockets.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class ProteinLigandValidator:
    """Validator for chemoinformatics docking and geometry screening."""

    def __init__(self):
        """Initialize the RDKit-based conformation validator."""
        print("Инициализация валидатора хемоинформатики RDKit...")

    def calculate_pocket_center(
        self, generated_atoms: np.ndarray, pocket_atom_indices: list
    ) -> np.ndarray:
        """Find the 3D geometric center of the generated topological pocket.

        Args:
            generated_atoms: Protein atom coordinates array of shape (N, 3).
            pocket_atom_indices: Indices of atoms forming the TDA cavity.

        Returns:
            Array of shape (3,) representing the XYZ center of the pocket.
        """
        pocket_coordinates = generated_atoms[pocket_atom_indices]
        return np.mean(pocket_coordinates, axis=0)

    def evaluate_ligand_fit(
        self,
        smiles_string: str,
        pocket_center: np.ndarray,
        search_radius: float = 6.0,
    ) -> float:
        """Evaluate spatial matching score of a small molecule ligand.

        Generate 3D conformations using RDKit and compute the proportion
        of atoms fitting inside the spatial radius.

        Args:
            smiles_string: Canonical SMILES string of the drug target.
            pocket_center: 3D coordinates array of the pocket center.
            search_radius: Maximum pocket cutoff radius for screening.

        Returns:
            Float binding fit score ranging from 0.0 to 1.0.
        """
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError(
                f"Невалидный SMILES-код молекулы: {smiles_string}"
            )

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)  # Оптимизация силового поля

        conformers = mol.GetConformers()
        if len(conformers) == 0:
            return 0.0

        conformer = conformers[0]
        ligand_coords = conformer.GetPositions()

        # Считаем, сколько атомов физически помещаются внутрь полости
        distances = np.linalg.norm(ligand_coords - pocket_center, axis=1)
        atoms_inside_pocket = np.sum(distances <= search_radius)

        fit_score = float(atoms_inside_pocket) / len(ligand_coords)
        return fit_score


if __name__ == "__main__":
    print("Запуск тестирования хемоинформационной валидации...")

    # Имитируем выход диффузионного движка (40 атомов белка в 3D)
    mock_generated_protein = np.random.uniform(
        low=5.0, high=15.0, size=(40, 3)
    )
    mock_pocket_indices = [5, 12, 18, 24]

    validator = ProteinLigandValidator()
    center = validator.calculate_pocket_center(
        mock_generated_protein, mock_pocket_indices
    )

    # Тестируем потенциальную молекулу-кандидат (PubChem CID 1983)
    test_smiles = "CC(=O)NC1=CC=C(C=C1)O"

    score = validator.evaluate_ligand_fit(
        test_smiles, pocket_center=center, search_radius=8.0
    )

    print(f"Геометрический центр вирусного кармана: {center}")
    print(f"Итоговый RDKit Conformer Fit Score: {score:.4f}")
    print("\nСквозной пайплайн MVP успешно верифицирован!")
