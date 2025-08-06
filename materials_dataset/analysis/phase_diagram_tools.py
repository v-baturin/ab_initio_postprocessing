from functools import lru_cache
from ..crystal_entry import CrystalEntry
from ..crystal_dataset import CrystalDataset
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry




class PhaseDiagramTools:

    def __init__(self, dataset: CrystalDataset):
        self.dataset = dataset
        self._phdiag = None
        assert all([e.energy is not None for e in self.dataset]), "Energies are missing in dataset entries"

    def phase_diagram(self):
        self._phdiag = self._phdiag or PhaseDiagram([self.as_pdentry(e) for e in self.dataset])
        return self._phdiag

    @lru_cache(maxsize=1000)
    def height_above_hull_pa(self, e: CrystalEntry):
        return self.phase_diagram().get_e_above_hull(PhaseDiagramTools.as_pdentry(e))

    @lru_cache(maxsize=1000)
    def formation_energy(self, e: CrystalEntry) -> float:
        return self.phase_diagram().get_form_energy(PhaseDiagramTools.as_pdentry(e))

    def get_e_hull_pa_list(self):
        return [self.height_above_hull_pa(e) for e in self.dataset]


    @staticmethod
    def as_pdentry(e: CrystalEntry):
        return PDEntry(composition=e.composition, energy=e.energy, attribute=e.id)
