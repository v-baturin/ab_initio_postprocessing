from dataclasses import dataclass
from typing import Callable, ClassVar, Dict
import types
from ..crystal_dataset import CrystalDataset, CrystalEntry


@dataclass(slots=True)
class NNEstimator:

    known_estimators: ClassVar[Dict[str, types.ModuleType]] = {}

    def estimate_dataset_energies(self, dataset: CrystalDataset, estimator_name: str | None = "mattersim"):
        if estimator_name not in self.known_estimators:
            raise ValueError(f"Unknown estimator '{estimator_name}', available: {list(self.known_estimators.keys())}")
        estimator = self.known_estimators[estimator_name]
        new_energies = estimator.estimate_batch(dataset)
        new_entries = []
        for entry, estimation in zip(dataset, new_energies):
            new_entries.append(entry.copy_with(**{"energy": estimation}))
        msg = f"Energies estimated with {estimator_name}"
        return CrystalDataset.from_parents(new_entries, (dataset,), message=msg)

    def estimate_entry_energy(self, e: CrystalEntry, estimator: str | None = "mattersim"):
        return self.known_estimators[estimator].estimate_entry_energy(e)

    @classmethod
    def register_estimator(cls, name: str, module: types.ModuleType):
        cls.known_estimators[name] = module
