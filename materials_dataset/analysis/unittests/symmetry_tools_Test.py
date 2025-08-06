import unittest
from pathlib import Path
from pymatgen.core import Structure
from ...crystal_entry import CrystalEntry
from ...crystal_dataset import CrystalDataset
from ..symmetry_tools import SymmetryToolkit

PATH_WITH_DATASETS = Path(__file__).parent / "../../unittests_datasets"


class symmetry_tools_Test(unittest.TestCase):
    def setUp(self):
        sample_poscars_file = PATH_WITH_DATASETS /  "POSCARS"
        entries = [CrystalEntry(id = str(i), structure=Structure.from_file(f), energy = -i/10) \
                      for i, f in enumerate(sample_poscars_file.rglob('*POSCAR'))]
        self.ds = CrystalDataset(entries)
        SymmetryToolkit.set_symprecs(1e-4, 1e-2)
        self.stk = SymmetryToolkit()

    def test_sym_group_no(self):
        self.assertEqual(self.stk.sym_group_no(self.ds[8]), 1)

    def test_get_symmetrized_entry(self):
        original = self.ds[8]
        symmetrized = self.stk.get_symmetrized_entry(original)
        self.assertEqual('P1', self.stk.sym_group_symbol(original))
        self.assertEqual('Cmcm', self.stk.sym_group_symbol(symmetrized))

