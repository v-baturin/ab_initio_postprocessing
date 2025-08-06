import unittest
from pathlib import Path
from pymatgen.core import Structure
from ...crystal_entry import CrystalEntry
from ...crystal_dataset import CrystalDataset
from ..storage import load, save

PATH_WITH_TESTS = Path(__file__).parent

class storage_Test(unittest.TestCase):
    def setUp(self):
        sample_poscars_file = PATH_WITH_TESTS /  "../../io/unittests/POSCARS"
        entries = [CrystalEntry(id = str(i), structure=Structure.from_file(f), energy = -i/10) \
                      for i, f in enumerate(sample_poscars_file.rglob('*POSCAR'))]
        self.ds = CrystalDataset(entries)

    def test_saveload(self):
        save(self.ds, 'tmp.pkl')
        ds = load('tmp.pkl')
        print(len(ds))


