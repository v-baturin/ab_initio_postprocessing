from __future__ import annotations
import os
from pathlib import Path
import pickle as pkl
from my_packages.materials_tools.materials_dataset.crystalDataset import CrystalDataset, CrystalEntry
from my_packages.materials_tools.materials_dataset.db_preset_scenarios import gather_entries_from_databases

DEBUG = False

class GenerationPostprocessingPipeline:
    """
    A class to handle the postprocessing of generated structures from CIF files or POSCARs.
    It reads raw generated structures, deduplicates them, and saves the processed dataset.
    """

    def __init__(self,
                 repository: Path = None,
                 raw_cif_folder=Path("cif_files"),
                 raw_poscars_folder=Path("POSCARs"),
                 rgs_pkl=None,
                 srgs_pkl=None,
                 rs_pkl=None,
                 ms_pkl=None,
                 dft_input_pkl=None,
                 single_point_pkl=None,
                 dft_relaxed_pkl=None,
                 elements=None,
                 max_eCHull=0.02,
                 ):
        self.repository = repository or Path(os.getcwd())  # Assuming this script is in a subfolder of the repository root
        self.raw_cif_folder = raw_cif_folder
        self.raw_poscars_folder = raw_poscars_folder
        self.rgs_pkl = rgs_pkl or self.repository / "rgs.pkl"
        self.rgs = None # Raw Generated Structures
        self.srgs_pkl = srgs_pkl or self.repository / "srgs.pkl"
        self.srgs = None # Symmetrized Raw Generated Structures
        self.rs_pkl = rs_pkl or self.repository / "rs.pkl"
        self.rs = None # Reference Structures
        self.merged_set_pkl = ms_pkl or self.repository / "merged_set.pkl"
        self.ms = None # Merged Set of Structures
        self.dft_input_pkl = dft_input_pkl or self.repository / "dft_input.pkl"
        self.dft_input = None # DFT Input Structures
        self.single_point_pkl = single_point_pkl or self.repository / "single_point.pkl"
        self.single_point = None # Single Point Calculations Results
        self.dft_relaxed_pkl = dft_relaxed_pkl or self.repository / "dft_relaxed.pkl"
        self.dft_relaxed = None # DFT Relaxed Structures
        self._elements = elements
        self.maxCH = max_eCHull


    def get_rgs(self, refresh=False):
        """
        Returns the raw generated structures, reading from a pickle file or generating them from cif files or POSCARs.
        """
        if self.rgs is None or refresh:
            self.rgs = self._read_raw_generated_structures(refresh)
        return self.rgs

    def get_srgs(self, refresh=False):
        """
        Returns the symmetrized raw generated structures, reading from a pickle file or generating them from raw generated structures.
        """
        if self.srgs is None:
            if self.srgs_pkl.exists() and not refresh:
                print("Symmetrized raw generated structures pickle file exists. Use refresh=True to regenerate.")
                with open(self.srgs_pkl, 'br') as fh:
                    self.srgs = pkl.load(fh)
                self.srgs.pkl_path = self.srgs_pkl
            else:
                rgs = self.get_rgs(refresh)
                self.srgs : CrystalDataset = rgs.copy()
                self.srgs.symmetrize(reset_compromised_parameters=True)
                self.srgs.pkl_path = self.srgs_pkl
                self.srgs.dump()
        return self.srgs

    def get_rs(self, refresh=False):
        """
        Returns the reference structures, reading from a pickle file or generating them from symmetrized raw generated structures.
        """
        if self.rs is None:
            if self.rs_pkl.exists() and not refresh:
                print("Reference structures pickle file exists. Use refresh=True to regenerate.")
                with open(self.rs_pkl, 'br') as fh:
                    self.rs = pkl.load(fh)
            else:
                if DEBUG:
                    self.rs = gather_entries_from_databases(self.elements, do_deduplication=True,
                                                            do_ehull_filtering=True, max_eCHull=self.maxCH,
                                                            database_names=['oqmd', 'MatProj'],
                                                            repository=self.repository)
                else:
                    self.rs = gather_entries_from_databases(self.elements, do_deduplication=True,
                                                            do_ehull_filtering=True, max_eCHull=self.maxCH,
                                                            repository=self.repository)
                self.rs.pkl_path = self.rs_pkl
                self.rs.dump()
        return self.rs

    def get_ms(self, refresh=False):
        """
        Merges symmetrized raw generated structures into reference structures with reproductivity statistics
        """
        if self.ms is None:
            if self.merged_set_pkl.exists() and not refresh:
                print("Merged set of structures pickle file exists. Use refresh=True to regenerate.")
                with open(self.merged_set_pkl, 'br') as fh:
                    self.ms = pkl.load(fh)
            else:
                self.ms = self.get_rs().merge_from(self.get_srgs(),
                                                   reset_entry_caches=False, reset_caches=True,
                                                   check_duplicates=True, tol_FP=0.08, skip_dump=True)
                self.ms.pkl_path = self.merged_set_pkl
                self.ms.dump()
        return self.ms



    def _read_raw_generated_structures(self, refresh=False):
        """
        Reads the raw generated structures from a pickle file or generates them from cif files or POSCARs.
        """
        if self.rgs_pkl.exists() and not refresh:
            print("Raw generated structures pickle file exists. Use refresh=True to regenerate.")
            with open(self.rgs_pkl, 'br') as fh:
                rgs = pkl.load(fh)
                return rgs

        # Check if the cif_files folder exists
        if not self.raw_cif_folder.exists() and not self.raw_poscars_folder.exists():
            raise FileNotFoundError("Neither cif_files nor POSCARs folder exists to generate the dataset.")

        rgs = CrystalDataset.from_struct_folder(self.raw_poscars_folder, search_pattern='*POSCAR',
                                                repository=self.repository,
                                                skip_dump=True) if self.raw_poscars_folder.exists() \
            else CrystalDataset.from_struct_folder(self.raw_cif_folder, search_pattern='*.cif',
                                                   repository=self.repository, skip_dump=True)
        rgs.pkl_path = self.rgs_pkl  # Set the pickle path for the raw generated structures

        # Save the dataset to a pickle file

        rgs.dump()

        return rgs

    # -------------------------------------------------------------------
    # Helpers and properties
    # -------------------------------------------------------------------

    @property
    def elements(self):
        """
        Returns the elements of the raw generated structures.
        If not set, it will be inferred from the raw generated structures.
        """
        if self._elements is None:
            for dataset in [self.rgs, self.srgs, self.rs, self.ms, self.dft_input, self.single_point, self.dft_relaxed]:
                if dataset is not None:
                    self._elements = dataset.elements
                    break
            else:
                # If no dataset is available, read the raw generated structures
                print("No dataset available. Reading raw generated structures to infer elements.")
                self._elements = self.get_rgs().elements
        return self._elements






# if not rgs_pkl.exists():
#     if not raw_poscars_folder.exists():
#         assert raw_cif_folder.exists(), "Either cif_files or POSCARs folder must exist to generate the dataset."
#         rgs = CrystalDataset.from_struct_folder(raw_cif_folder, search_pattern='*.cif')
#     else:
#         rgs = CrystalDataset.from_struct_folder(raw_poscars_folder, search_pattern='*POSCAR')
# else:
#     with open(rgs_pkl, 'br') as fh:
#         rgs = pkl.load(fh)
#
#
#
#
#     def run(self, refresh=False):
#         return self.read_raw_generated_structures(self.raw_cif_folder, self.raw_poscars_folder, self.rgs_pkl, refresh)
