import unittest
import pickle as pkl
from pathlib import Path
from materials_tools.materials_dataset.crystalDataset import CrystalDataset, CrystalEntry
from materials_tools.materials_dataset.db_clients.oqmd_client import OQMDClient


class CrystalDataset_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.poscars_folder = Path("poscars")
        self.cifs_folder = Path("cifs")

    def test_readCifs(self):
        cifs = CrystalDataset.from_struct_folder(self.cifs_folder, search_pattern='*.cif', skip_dump=True)
        self.assertEqual(len(cifs), 62)

    def test_elements(self):
        ds = CrystalDataset.from_struct_folder(self.poscars_folder, search_pattern='*.POSCAR', skip_dump=True)
        self.assertEqual(ds.elements, {'N', 'Na', 'Al', 'Ni', 'Fe'})

    def test_deduplication(self):
        ds = CrystalDataset.from_struct_folder(self.poscars_folder, search_pattern='*.POSCAR', skip_dump=True)
        assert len(ds) == 348, "Dataset should contain 348 entries before deduplication"
        ds.tol_FP = 0.07
        deduped_ds, _, _ = ds.deduplicated(enforce_compositions_separation=True,
                                            fitness_list=None,
                                            reset_entries=False, skip_dump=True)
        self.assertEqual(len(deduped_ds),300)

    def test_filtering(self):
        ds = CrystalDataset.from_client(OQMDClient(), elements=['Al', 'Fe', 'Ni'], skip_dump=True)
        fs = ds.filter(predicate_fn=lambda e: e.e_above_hull < 0.02, reset_entries=False, skip_dump=True)
        print(fs.metadata["message"])

    def test_rw(self):
        ds = CrystalDataset.from_struct_folder(self.poscars_folder, search_pattern='*.POSCAR', skip_dump=True)
        ds.dump()
        with open(ds.pkl_path, 'rb') as f:
            ds2 = pkl.load(f)
        print(f"ds1.id: {ds.id}, ds2.id: {ds2.id}")
        self.assertEqual(ds.id, ds2.id)
        Path(ds.pkl_path).unlink()
        Path(ds.regfile).unlink()
        Path(ds.treefile).unlink()

