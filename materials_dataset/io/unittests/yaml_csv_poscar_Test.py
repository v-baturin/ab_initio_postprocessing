import unittest
import filecmp
import shutil
from pathlib import Path
from ..yaml_csv_poscars import read, write
from ..preset_loaders import load_mattersim_estimated_set

PATH_WITH_TESTS = Path(__file__).parent


class yaml_csv_poscars_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.csv = PATH_WITH_TESTS / 'mattersim_res_for_POSCARS.csv'
        PATH_WITH_DATASETS = PATH_WITH_TESTS / "../../unittests_datasets"
        self.poscars_folder = PATH_WITH_DATASETS / "POSCARS"
        self.ref_dump_results = PATH_WITH_DATASETS / "ref_yaml_csv_poscars"


    def test_write_read(self):
        ds = load_mattersim_estimated_set(self.csv, self.poscars_folder)
        ds.metadata = {}
        ds.dataset_id = '0'
        res_path_1 = PATH_WITH_TESTS / "tmp1"
        ds.override_base_path(res_path_1)
        write(ds)
        dcmp = filecmp.dircmp(self.ref_dump_results, res_path_1)
        mismatch = dcmp.diff_files or dcmp.left_only or dcmp.right_only
        for common_dir in dcmp.common_dirs:
            mismatch = mismatch or dcmp.subdirs[common_dir].diff_files
        self.assertFalse(mismatch)
        res_path_2 = PATH_WITH_TESTS / "tmp2"
        ds2 = read(ds.base_path / f"manifest.yaml")
        ds2.override_base_path(res_path_2)
        write(ds2)
        dcmp = filecmp.dircmp(res_path_1, res_path_2)
        mismatch = dcmp.diff_files
        for common_dir in dcmp.common_dirs:
            mismatch = mismatch or dcmp.subdirs[common_dir].diff_files
        self.assertFalse(mismatch)
        shutil.rmtree(res_path_1)
        shutil.rmtree(res_path_2)