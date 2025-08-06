import unittest
from pathlib import Path
from ..structures_dataset_io import StructureDatasetIO

PATH_WITH_TESTS = Path(__file__).parent


class yaml_csv_poscars_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.zipped_cifs = PATH_WITH_TESTS / 'zipped_cifs'

    def test_unzip(self):
        self.utils = StructureDatasetIO(self.zipped_cifs)
        ds = self.utils.load_from_directory()
        print(len(ds))
        self.assertEqual(len(ds), 696)

