import unittest
from pathlib import Path
from ..uspex_bridge import USPEXBridge
from ..structures_dataset_io import StructureDatasetIO

PATH_WITH_TESTS = Path(__file__).parent
PATH_WITH_DATASETS = PATH_WITH_TESTS / "../../unittests_datasets"

class USPEXBridge_Test(unittest.TestCase):

    def setUp(self):
        self.two_systems = StructureDatasetIO(PATH_WITH_DATASETS/"two_systems").load_from_directory()
        self.ub = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True)


    def test_distance(self):
        self.assertAlmostEqual(self.ub.fp_dist(*self.two_systems), 0.1822, places=4)