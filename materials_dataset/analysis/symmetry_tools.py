from functools import lru_cache
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..crystal_dataset import CrystalDataset
from ..crystal_entry import CrystalEntry

ENFORCING_SYMPREC = 1e-2
ANALYZING_SYMPREC =  1e-4

class SymmetryToolkit:

    analyzing_symprec = ANALYZING_SYMPREC
    enforcing_symprec = ENFORCING_SYMPREC

    @classmethod
    def set_symprecs(cls, a_sym_prec=None, e_sym_prec=None):
        cls.analyzing_symprec = a_sym_prec or cls.analyzing_symprec
        cls.enforcing_symprec = e_sym_prec or cls.enforcing_symprec

    def __init__(self, a_sym_prec=None, e_sym_prec=None):
        self.analyzing_symprec = a_sym_prec or self.analyzing_symprec
        self.enforcing_symprec = e_sym_prec or self.enforcing_symprec

    @lru_cache(maxsize=1000)
    def _sga(self, e: CrystalEntry, symprec=None):
        return SpacegroupAnalyzer(e.structure, symprec=symprec or self.analyzing_symprec)

    def sym_group_no(self, e: CrystalEntry) -> int:
        return self._sga(e).get_space_group_number()


    def sym_group_symbol(self, e: CrystalEntry) -> str:
        return self._sga(e).get_space_group_symbol()

    def nonequivalent_sites(self, e: CrystalEntry) -> dict:
        symm_struct = self._sga(e).get_symmetrized_structure()
        nonequivalent_positions = {}
        for group in symm_struct.equivalent_sites:
            element = group[0].specie.symbol
            nonequivalent_positions[element] = nonequivalent_positions.get(element, 0) + 1
        return nonequivalent_positions

    def get_symmetrized_entry(self, e: CrystalEntry, primitive=True, drop_energy=True):
        sga = self._sga(e, symprec=self.enforcing_symprec)
        if primitive:
            struc = sga.get_primitive_standard_structure()
        else:
            struc = sga.get_refined_structure()
        energy = None if drop_energy else e.energy * len(struc) / len(e.structure)

        new_symmetry = SpacegroupAnalyzer(struc, self.enforcing_symprec).get_space_group_symbol()
        msg = f'symmetrized from {self.sym_group_symbol(e)} to {new_symmetry}'
        kw = {"structure": struc, "energy": energy}
        new_entry = e.copy_with(**kw)
        new_entry.log_message(msg)
        return new_entry

    def get_symmetrized_dataset(self, ds: CrystalDataset, primitive=True, drop_energy=True):
        new_entries = [self.get_symmetrized_entry(e, primitive=primitive, drop_energy=drop_energy) for e in ds]
        message = f'Symmetrized with symprec = {self.enforcing_symprec}'
        return CrystalDataset.from_parents(new_entries, parents=(ds,), message=message)