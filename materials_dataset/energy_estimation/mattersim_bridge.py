import numpy as np
from ..crystal_dataset import CrystalDataset, CrystalEntry
from materials_tools.NN_energy_estimators import mattersim_estimator

def estimate_batch(dataset: CrystalDataset):
    with mattersim_estimator.EnergyStream() as es:
        energies = np.zeros(len(dataset))
        for i, e in enumerate(dataset):
            energies[i] = es.calc(e.structure.to_ase_atoms())
    return energies

def estimate_entry_energy(e: CrystalEntry):
    return mattersim_estimator.get_energy(e.structure.to_ase_atoms())