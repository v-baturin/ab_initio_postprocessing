from __future__ import annotations
import os
import pickle as pkl
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Callable, Tuple, Iterable, List
import logging
from prettytable import PrettyTable
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from USPEX.components import Atomistic
from USPEX.DataModel.Engine import Engine
from USPEX.DataModel.Flavour import Flavour
from USPEX.DataModel.Entry import Entry
from materials_tools.uspex_toolkit.remove_duplicates import remove_duplicate_entries, prepare_dist_function
from materials_tools.NN_energy_estimators import mattersim_estimator
from genutils.misc import describe_predicate

Engine.createEngine(":memory:")
tdy = datetime.today().strftime('%Y%m%d')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

atomistic = Atomistic()
uspex_entry_extensions = dict(atomistic=(atomistic, atomistic.propertyExtension.propertyTable))

# Representation preset
ENTRY_ATTRIBUTE_LABELS = {"id": "ID", "composition": "Composition",
                          "energy_total": "Total energy (eV)",
                          "e_above_hull": "Energy above hull (eV/atom)",
                          "sym_group_no": "Symmetry group No",
                          "sym_group_symbol": "Symmetry group Symbol",
                          "nonequivalent_sites": "Non-equivalent sites"}

# DATASET_ATTRIBUTE_LABELS = {"e_above_hull_list": "Energy above hull (eV/atom)"}
DATASET_ATTRIBUTE_LABELS = {}
DEFAULT_SYMPREC = 1e-4
DEFAULT_TOL_FP = 0.08

ESTIMATORS = {"mattersim": mattersim_estimator}


class CrystalEntry:

    def __init__(self,
                 id: str,
                 structure: Structure | None = None,
                 composition: Composition | None = None,
                 energy_total: float | None = None,
                 origin: str = 'NA',
                 calc_settings: Mapping[str, Any] | None = None,
                 parent_set: CrystalDataset | None = None,
                 symprec: float = None, **kwargs):
        self.id = id
        self.structure = structure  # in‑memory structure
        self.composition = composition or getattr(structure, 'composition', None)
        self.energy_total = energy_total  # eV per formula unit
        self.origin = origin  # provenance tag ("oqmd", "mp", …)
        self.calc_settings = calc_settings
        self.parent_set = parent_set
        self.estimator = "mattersim"
        self._sga = None
        self._ehull_height_cache = None  # cache for energies above hull
        self.symprec = symprec or DEFAULT_SYMPREC

    # ---------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------
    @property
    def natoms(self) -> int:
        return len(self.structure)

    @property
    def formula(self) -> str:
        """Integer formula string, e.g. "Al4Fe2Ni2"."""
        return self.composition.formula.replace(" ", "")

    @property
    def e_above_hull(self) -> float:
        if self._ehull_height_cache is None:
            self._ehull_height_cache = self.parent_set.energies_above_hull_pa()[self]
        return self._ehull_height_cache

    @property
    def formation_energy(self) -> float:
        return self.parent_set.phase_diagram.get_form_energy(self.as_pd_entry())

    @property
    def formation_energy_pa(self) -> float:
        return self.parent_set.phase_diagram.get_form_energy(self.as_pd_entry()) / self.natoms

    @property
    def sga(self):
        if not hasattr(self, '_sga'):
            self._sga = None
        self._sga = self._sga or SpacegroupAnalyzer(self.structure, symprec=self.symprec if hasattr(self,
                                                                                                    'symprec') else DEFAULT_SYMPREC)
        return self._sga

    @property
    def sym_group_no(self) -> int:
        return self.sga.get_space_group_number()

    @property
    def sym_group_symbol(self) -> str:
        return self.sga.get_space_group_symbol()

    @property
    def nonequivalent_sites(self) -> dict:
        symm_struct = self.sga.get_symmetrized_structure()
        nonequivalent_positions = {}
        for group in symm_struct.equivalent_sites:
            element = group[0].specie.symbol
            nonequivalent_positions[element] = nonequivalent_positions.get(element, 0) + 1
        return nonequivalent_positions

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------
    def get_symmetrized_structure(self, symprec=1e-4, update_sga=False, primitive=True):
        # debug-------------
        sg_before = self.sym_group_no
        # end debug---------
        sga = SpacegroupAnalyzer(self.structure, symprec=symprec)
        if update_sga:
            self._sga = sga
        # debug-------------
        sg_after = self.sym_group_no
        if sg_before != sg_after:
            print(f"{self.id}: sg_before {sg_before} != sg_after {sg_after}")
        # end debug---------
        if primitive:
            return sga.get_primitive_standard_structure()
        else:
            return sga.get_refined_structure()

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def as_pd_entry(self) -> PDEntry | None:
        """Return a PhaseDiagram-compatible entry (total energy)."""
        if self.energy_total is not None:
            return PDEntry(self.composition, self.energy_total, attribute=self.id)
        else:
            warnings.warn("No energy provided, thermodynamic properties unavailable")
            return None

    def as_uspex_entry(self):
        atoms = AseAtomsAdaptor.get_atoms(self.structure)
        uspex_structure = atomistic.AtomicStructureRepresentation.fromAtoms(atoms)
        return Entry.newEntry(Flavour(extensions=uspex_entry_extensions,
                                      **{'.howCome': 'Seeds', '.parent': None, '.label': self.id},
                                      **atomistic.atomicDisassemblerType(
                                          np.arange(len(uspex_structure)).reshape((-1, 1))).disassemble(
                                          uspex_structure)))

    # ------------------------------------------------------------------
    # Hash / equality
    # ------------------------------------------------------------------
    def __hash__(self):  # noqa: D401
        return hash((self.origin, self.id))

    def __eq__(self, other: object):  # noqa: D401
        if not isinstance(other, CrystalEntry):
            return NotImplemented
        return (
                self.origin == other.origin
                and self.id == other.id
        )

    # ------------------------------------------------------------------
    # Constructors from external sources
    # ------------------------------------------------------------------
    @classmethod
    def from_tuple(cls, row: Mapping[str, Any] | pd.Series | Any, source: str) -> "CrystalEntry":
        """Create a `ThermoEntry` from a DataFrame row / dict / namedtuple."""

        # helper that works for dict, Series, or namedtuple
        def _get(obj: Any, key: str, default: Any = None):
            if isinstance(obj, Mapping):
                return obj.get(key, default)
            return getattr(obj, key, default)

        sid = str(_get(row, "id"))

        # ---------------structure and composition ---------------------
        structure: Structure | None = _get(row, "structure")
        composition: Composition | None = Composition(_get(row, "formula"))
        if structure is None:
            spath = _get(row, "structure_path")
            if spath is None:
                warnings.warn("Row lacks structure object or path")
            else:
                structure = Structure.from_file(Path(spath))

        if structure and composition:
            assert composition.reduced_formula == structure.composition.reduced_formula, f"{composition} != {structure.composition}"
            composition = structure.composition
        elif structure:
            composition = structure.composition
        else:
            raise ValueError("Row lacks both composition and structure")

        # ----------------------- energy -------------------------------
        if (et := _get(row, "e_total")) is not None:
            energy_total = float(et)
        elif (epa := _get(row, "e_pa")) is not None:
            energy_total = float(epa) * len(structure)
        else:
            raise ValueError("Row lacks energy information")

        calc_settings = _get(row, "settings") or _get(row, "calc_settings")

        return cls(
            id=sid,
            energy_total=energy_total,
            composition=composition,
            structure=structure,
            origin=source,
            calc_settings=calc_settings,
        )

    @classmethod
    def from_uspex_entry(cls, uspex_entry: Entry, **kwargs):
        atoms = atomistic.AtomicStructureRepresentation.toAtoms(
            uspex_entry["atomistic.structure.origin"])
        return cls(
            structure=AseAtomsAdaptor.get_structure(atoms),
            **kwargs
        )

    @classmethod
    def from_struc_file(cls, file_path, **kwargs):
        struct = Structure.from_file(file_path)
        return cls(structure=struct, **kwargs)


# ---------------------------------------------------------------------------
# CrystalDataset
# ---------------------------------------------------------------------------


class CrystalDataset(list[CrystalEntry]):
    """Collection of entries from one provenance (total energies).
    All pkl's of the same tree are stored in the same repository.
    """

    def __init__(self, entries: Optional[list[CrystalEntry]] = None, elements=None, reset_entries=True,
                 tol_FP: float = None,
                 estimator: str = "mattersim",
                 message: str = None, skip_dump=False, repository: str | Path = '', parents: list | None = None,
                 id=None,
                 regfile: str = "registry.txt", treefile: str = "tree.txt",
                 **kwargs):
        super().__init__(entries or [])
        self.estimator = estimator
        self._pd_cache: PhaseDiagram | None = None
        self._ehull_cache: Dict[CrystalEntry, float] | None = None
        self._rdf_utility = None
        self._uspex_entries = None
        self._elements: Set[str] = elements
        if reset_entries:
            self._reset_entry_caches()
        self._reset_caches()
        self.tol_FP = tol_FP or DEFAULT_TOL_FP
        self.metadata: dict = {
            "elements": self.elements,
            "message": message or f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - Crystal dataset created",
        }
        self.repository = Path(repository).expanduser().resolve() if repository else Path(
            os.getcwd())  # Repository path for storing dataset files
        self.id = id or self.update_id()
        self.pkl_path = self.repository / f"{self.id}.pkl"  # Path to save the dataset as a pickle file
        self.parents = parents  # List of parent datasets, if any
        self.regfile = self.repository / regfile  # Path to the registry file
        self.treefile = self.repository / treefile  # Path to the tree file
        if not skip_dump:
            self.dump()

    # ------------------------------------------------------------------
    # Infrastructure / validation / cache helpers
    # ------------------------------------------------------------------

    def update_id(self) -> str:
        self.id = hex(hash((id(self), datetime.today(), self.metadata["message"])))[2:]
        self.pkl_path = self.repository / f"{self.id}.pkl"  # Path to save the dataset as a pickle file
        return self.id

    def dump(self) -> None:
        with open(self.pkl_path, 'wb') as fh:
            pkl.dump(self, fh)
        with (open(self.regfile, 'a') as reg_file,
              open(self.treefile, 'a') as tree_file):
            reg_file.write(f"{self.id:<20} - {self.metadata["message"]}" +
                           (f" - from parents: {self.parents}" if self.parents else ' - origin')
                           + f" - saved in {self.pkl_path.name}"
                           + '\n')
            tree_file.write(f"{self.id:10s}"
                            + (f" from {' and '.join(self.parents)} " if self.parents else ' origin')
                            + '\n')

    def _reset_caches(self):
        self._pd_cache: PhaseDiagram | None = None
        self._ehull_cache: Dict[CrystalEntry, float] | None = None
        self._rdf_utility = None
        self._uspex_entries = None

    def _reset_entry_caches(self):
        for e in self:
            e.parent_set = self
            e._ehull_height_cache = None

    def load_parents(self) -> List[CrystalDataset]:
        """Load parent datasets from the repository."""
        parents = []
        for parent_id in self.parents:
            parent_path = self.repository / f"{parent_id}.pkl"
            if parent_path.exists():
                with open(parent_path, 'rb') as fh:
                    parents.append(pkl.load(fh))
            else:
                warnings.warn(f"Parent dataset {parent_id} not found in repository.")
        return parents

    def extend(self, other, check_duplicates: bool = False, tol_FP=None,
               reset_caches=True, reset_entry_caches=True, verbose=True, **kwargs):
        if 'id' not in self.__dict__:  # for unpickling
            return super().extend(other)
        """Extend the dataset with new entries, optionally checking for duplicates."""
        if check_duplicates:
            tol_FP = tol_FP or self.tol_FP
            new_entries = []
            duplicates_counter = set()
            logger.info("Preparing distance function for duplicate checking...")
            rho = prepare_dist_function(self.uspex_entry_list + other.uspex_entry_list,
                                        elements=self.elements | other.elements,
                                        storeDistances=False)
            for i, e in enumerate(other.uspex_entry_list):
                logger.info(f"Checking entry {i + 1}/{len(other.uspex_entry_list)} of added dataset for duplicates...")
                for j, e2 in enumerate(self.uspex_entry_list):
                    if rho(e, e2) <= tol_FP:
                        logger.info(f"{other[i].id} in extension list is a duplicate of {self[j].id}")
                        duplicates_counter.add(j)
                        break
                else:
                    new_entries.append(other[i])
            if verbose:
                reproductibility = len(duplicates_counter) / len(self)
                logger.info(f"Extension contains {reproductibility:.2%} of initial entries")
                other.metadata["reproducibility"] = reproductibility

        else:
            new_entries = other
        super().extend(new_entries)
        old_id = self.id
        self.update_id()
        self.parents = [old_id, other.id] if isinstance(other, CrystalDataset) else [old_id]
        self.metadata["message"] = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - "
                                    f"Extension of {old_id} by "
                                    f"{other.id if isinstance(other, CrystalDataset) else str(len(other)) + 'entries'}"
                                    f"{' by NEW structures only' if check_duplicates else ''}")
        if reset_caches:
            self._reset_caches()
        if reset_entry_caches:
            self._reset_entry_caches()

    def merge_from(self, other: CrystalDataset, check_duplicates: bool = False, tol_FP: float = None,
                   reset_caches=True, reset_entry_caches=True, verbose=True, skip_dump=False,
                   **kwargs) -> CrystalDataset:
        """Merge another dataset into this one."""
        ours = self.copy()
        theirs = other.copy()
        ours.extend(theirs, check_duplicates=check_duplicates, tol_FP=tol_FP,
                    reset_caches=reset_caches, reset_entry_caches=reset_entry_caches,
                    verbose=verbose)
        reproducibility = ours.metadata['reproducibility'] if 'reproducibility' in ours.metadata else 0.0
        ours.parents = [self.id, other.id]
        ours.metadata["message"] = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - "
                                    f"Merge of {self.id} by {other.id} {f'by NEW structures only ({reproducibility:.2%} reproduced)' if check_duplicates else ''}"
                                    )
        ours.refresh_id()
        if not skip_dump:
            ours.dump()
        return ours

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_client(cls, client: Any, elements: Iterable[str], *,
                    repository: Path | str | None = None,
                    label: str | None = None, **kwargs) -> "CrystalDataset":
        if hasattr(client, "query"):
            df = client.query(elements)
        else:
            raise AttributeError("Client lacks 'query' method")
        tag = label or client.__class__.__name__.replace("Client", "").lower()
        entries = [CrystalEntry.from_tuple(r, tag) for r in df.itertuples(index=False)]
        dataset = cls(entries, message=f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - Created from {tag}",
                      repository=repository, **kwargs)
        dataset._reset_caches()
        dataset._reset_entry_caches()
        return dataset

    @classmethod
    def from_struct_folder(cls,
                           structures_path: Path | str,
                           search_pattern: str = '*POSCAR',
                           parameters_dict: Dict[str, list] | None = None,
                           **kwargs) -> "CrystalDataset":
        structures_path = Path(structures_path)
        found_files = sorted(structures_path.rglob(search_pattern))
        if parameters_dict is not None:
            for val in parameters_dict.values():
                assert len(val) == len(list(found_files))
            params = [dict(zip(parameters_dict.keys(), values)) for values in zip(*parameters_dict.values())]
        else:
            params = [{'id': str(i)} for i in range(len(found_files))]
        entry_list = []
        for i, f in enumerate(found_files):
            try:
                entry_list.append(CrystalEntry.from_struc_file(f, origin=f.name, **params[i]))
            except ValueError:
                continue
        repo = (kwargs.get("repository", '/'))
        dataset = cls(entry_list, message=f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - "
                                          f"Created from structures in {os.path.relpath(str(structures_path), str(repo))}",
                      **kwargs)
        dataset._reset_entry_caches()
        return dataset

    # ------------------------------------------------------------------
    # Properties & heavy operations
    # ------------------------------------------------------------------
    @property
    def source(self) -> Optional[str]:
        return ', '.join(list({e.origin for e in self})) if self else None

    @property
    def elements(self) -> Set[str]:
        """Set of elements in the dataset."""
        if not self._elements:
            self._elements = set(
                el for e in self if e.structure for el in e.structure.composition.as_data_dict()['elements'])
        return self._elements

    @property
    def uspex_entry_list(self) -> list[Entry]:
        if getattr(self, "_uspex_entries", None) is None:
            self._uspex_entries = [e.as_uspex_entry() for e in self]
        return self._uspex_entries

    @property
    def phase_diagram(self) -> PhaseDiagram | None:
        if not any([e.energy_total for e in self]):
            warnings.warn("No energies provided, thermodynamics unavailable")
            return None
        if getattr(self, "_pd_cache", None) is None:
            self._pd_cache = PhaseDiagram([e.as_pd_entry() for e in self])
        return self._pd_cache  # type: ignore[return-value]

    @property
    def energies_above_hull_list(self):
        return [self.energies_above_hull_pa()[e] for e in self]

    # ------------------------------------------------------------------
    # Energy utilities
    # ------------------------------------------------------------------
    def energies_above_hull_pa(self, refresh: bool = False) -> Dict[CrystalEntry, float | None]:
        if getattr(self, "_ehull_cache", None) is None or refresh:
            pd = self.phase_diagram
            if pd is None:
                return {e: None for e in self}
            self._ehull_cache = {e: pd.get_e_above_hull(e.as_pd_entry()) for e in self}
        return self._ehull_cache  # type: ignore[return-value]

    def energy_above_hull(self, entry: CrystalEntry):
        return self.phase_diagram.get_e_above_hull(entry.as_pd_entry())

    #------------------------------------------------------------------
    # Filtering
    #------------------------------------------------------------------
    def filter(self, predicate_fn: Callable[[CrystalEntry], bool], **kwargs) -> CrystalDataset:
        message = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - parent filtered by "
                   f"{describe_predicate(predicate_fn)}")
        filtered_set = self.__class__([e for e in self if predicate_fn(e)], repository=self.repository,
                                      parents=[self.id],
                                      message=message, **kwargs)
        return filtered_set

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def __repr__(self):  # noqa: D401
        return f"<CrystalDataset n={len(self)} source='{self.source}'>"

    def as_pandas_df(self,
                     entry_attribute_labels: dict | None = None,
                     dataset_attribute_labels: dict | None = None):
        # labels = {parameter_label: parameter_name}
        if entry_attribute_labels is None:
            entry_attribute_labels = ENTRY_ATTRIBUTE_LABELS
        if dataset_attribute_labels is None:
            dataset_attribute_labels = DATASET_ATTRIBUTE_LABELS
        data = {v: [getattr(e, k) for e in self] for k, v in entry_attribute_labels.items()}
        data.update({v: getattr(self, k) for k, v in dataset_attribute_labels.items()})
        df = pd.DataFrame(data)
        # Identify and drop all-None columns
        all_none_cols = [col for col in df.columns if df[col].isna().all()]
        df.drop(columns=all_none_cols, inplace=True)
        # Report dropped columns
        if all_none_cols:
            print(f"Dropped columns with only None values: {', '.join(all_none_cols)}")
        return df

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def present_as_table(self, dump_path: Path | str | None = None, sort_by=None, columns=None):
        df = self.as_pandas_df()
        df["_int_ID"] = df["ID"].str.extract(r'(\d+)(?!.*\d)')[0].astype(int)
        if isinstance(sort_by, list):
            sort_by.append("_int_ID")
        elif sort_by:
            sort_by = [sort_by, "_int_ID"]
        if not sort_by:
            df = df.sort_values(by=sort_by)
        df.drop(columns="_int_ID", inplace=True)
        if not columns:
            columns = list(df.columns)
        else:
            # Validate that each requested column exists
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns {missing} are not in the DataFrame ({df.columns})")
        pt = PrettyTable()
        pt.field_names = columns

        # Slice df to just the chosen columns, then iterate
        for row in df[columns].itertuples(index=False, name=None):
            pt.add_row(row)

        # Write the table string to a file
        if dump_path:
            with open(dump_path, "w") as file:
                file.write(pt.get_string())
        else:
            print(pt)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def write_poscar_files(self, directory: Path | str | None = None, file_id_mapping_file: Path | str | None = None):
        if directory is None:
            directory = os.getcwd()
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        fnames = []
        for e in self:
            fname = f"{e.id.replace('POSCAR', '')}-{e.formula}_POSCAR".split("/")[-1]
            fnames.append(fname)
            e.structure.to(fmt="poscar", filename=directory / fname, comment=e.id)
        if file_id_mapping_file:
            with open(directory / file_id_mapping_file, 'wt') as fid:
                for i, e in enumerate(self):
                    fid.write(f"{fnames[i]} {e.id}\n")

    def write_multistructure_poscar(self, filename: Path | str | None = None, sort_key=None,
                                    entries_order_file: Path | str | None = None):
        if filename is None:
            filename = Path(os.getcwd()) / "POSCARS"
        with open(filename, 'wt') as fid:
            sequence = self if not sort_key else sorted(self, key=sort_key)
            for e in sequence:
                poscar_str = Poscar(e.structure, comment=e.id).get_str()
                fid.write(poscar_str)
            if entries_order_file:
                with open(entries_order_file, 'wt') as order_file:
                    for e in sequence:
                        order_file.write(f"{e.id}\n")

    def to_energy_csv(self, path: Path | str | None = None, *, unique: bool = False) -> str:
        df = pd.DataFrame({
            "id": [e.id for e in self],
            "composition": [e.formula for e in self],
            "e_total": [e.energy_total for e in self],
        })
        if unique:
            df = df.loc[df.groupby("composition").e_total.idxmin()].reset_index(drop=True)
        csv_txt = df.to_csv(index=False)
        if path is not None:
            Path(path).write_text(csv_txt)
        return csv_txt

    # ------------------------------------------------------------------
    # USPEX-based structural filtration
    # ------------------------------------------------------------------
    def deduplicated(self,
                     check_clusters_file=False,
                     check_dist_matrix_file=False,
                     tol_FP=None,
                     enforce_compositions_separation=True,
                     fitness_list=None,
                     **kwargs):
        """
        Remove duplicates from the dataset using USPEX's remove_duplicates function.
        :param check_clusters_file: If True, will write clusters to a file.
        :param check_dist_matrix_file: If True, will write distance matrix to a file.
        :param tol_FP: Tolerance for fingerprint distance.
        :param enforce_compositions_separation: If True, will enforce separation of compositions in clusters.
        :param fitness_list: List of fitness values (e.g., energies) for each entry.
        """
        tol_FP = tol_FP or self.tol_FP
        try:
            fitness_list = fitness_list or [e.energy_total / e.natoms for e in self]
        except TypeError:
            fitness_list = None

        if enforce_compositions_separation:
            reduced_compositions = [e.composition.reduced_formula for e in self]
        else:
            reduced_compositions = None

        best_representatives, clusters, best_idx = remove_duplicate_entries(self.uspex_entry_list, fitness_list,
                                                                            check_clusters_file=check_clusters_file,
                                                                            check_dist_matrix_file=check_dist_matrix_file,
                                                                            tol_Fp=tol_FP,
                                                                            enforce_compositions_separation=enforce_compositions_separation,
                                                                            compositions_list=reduced_compositions,
                                                                            elements=self.elements,
                                                                            **kwargs)
        filtered_list = [self[i] for i in best_idx]
        message = f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - Parent deduplicated with tol_FP={tol_FP} "
        return CrystalDataset(filtered_list, message=message, parents=[self.id], **kwargs), clusters, best_idx

    def contains_structure(self, crystal_entry: CrystalEntry, tol_FP=None) -> Tuple[list, list]:
        tol_FP = tol_FP or self.tol_FP
        uspex_entry_ref = crystal_entry.as_uspex_entry()
        rho = prepare_dist_function(self.uspex_entry_list + [uspex_entry_ref],
                                    elements=self.elements | {crystal_entry.composition.as_data_dict()['elements']})
        true_idcs = [i for i, uspex_entry in enumerate(self.uspex_entry_list) if
                     rho(uspex_entry_ref, uspex_entry) <= tol_FP]
        return true_idcs, [self[i].id for i in true_idcs]

    # ------------------------------------------------------------------
    # Symmetrization
    # ------------------------------------------------------------------
    def symmetrize(self, symprec: float = DEFAULT_SYMPREC, update_sga=False,
                   primitive=True, reset_compromised_parameters=True, update_id=True, dump=False) -> CrystalDataset:
        """Return a list of symmetrized structures."""
        for e in self:
            e.structure = e.get_symmetrized_structure(symprec=symprec, update_sga=update_sga, primitive=primitive)
        if reset_compromised_parameters:
            self._reset_entry_caches()
            self._reset_caches()
            for e in self:
                e.energy_total = None
                e._ehull_height_cache = None
        if update_id:
            old_id = self.id
            self.update_id()
            self.parents = [old_id]
            self.metadata["message"] = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - Parent symmetrized with "
                                        f"symprec={symprec}."
                                        f"{' Primitive cells saved.' if primitive else ''}"
                                        )
        if dump:
            self.dump()

    # ------------------------------------------------------------------
    # Batch energy estimation
    # ------------------------------------------------------------------
    def estimate_energies(self, estimator: str | None = None, update_id=True, dump=False, **kwargs):
        if estimator is None:
            estimator = self.estimator
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator '{estimator}', available: {list(ESTIMATORS.keys())}")
        estimator = ESTIMATORS[estimator]
        with estimator.EnergyStream() as es:
            for e in self:
                e.energy_total = es.calc(e.structure.to_ase_atoms())
        if update_id:
            old_id = self.id
            self.update_id()
            self.parents = [old_id]
            self.metadata["message"] = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')} - Energies estimated with "
                                        f"{estimator}."
                                        )
        if dump:
            self.dump()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def __getstate__(self):
        memo = {}
        data_copy = []
        for e in self:
            entry_dict = e.__dict__.copy()
            if "structure" in entry_dict:
                entry_dict["structure_dict"] = entry_dict["structure"].as_dict()
            if "composition" in entry_dict:
                entry_dict["composition_dict"] = entry_dict["composition"].as_dict()
            for to_drop in ["parent_set"]:
                entry_dict.pop(to_drop)
            data_copy.append(entry_dict)
        attrs_copy = self.__dict__.copy()
        for k in ('_ehull_cache', '_pd_cache',
                  '_rdf_utility', '_uspex_entries'):
            attrs_copy.pop(k, None)
        if "tol_FP" not in attrs_copy:
            attrs_copy["tol_FP"] = DEFAULT_TOL_FP
        return data_copy, attrs_copy

    def __setstate__(self, state):
        data_copy, attrs_copy = state
        attrs_copy.pop('_ehull_cache', None)
        self.clear()
        for e in data_copy:
            if "structure_dict" in e:
                e["structure"] = Structure.from_dict(e["structure_dict"])
                e.pop("structure_dict")
            if "composition" in e:
                e["composition"] = Composition.from_dict(e["composition_dict"])
                e.pop("composition_dict")
            e["parent_set"] = self
        self.__dict__.update(attrs_copy)
        self.extend([CrystalEntry(**dct) for dct in data_copy])
        self.id = attrs_copy.get('id')

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def copy(self):
        """Create a deep copy of the dataset."""
        self_dict_copy = self.__dict__.copy()
        self_dict_copy['skip_dump'] = True
        new_dataset = CrystalDataset([CrystalEntry(**e.__dict__.copy()) for e in self], **self_dict_copy)
        return new_dataset

    def __copy__(self):
        return self.copy()

    def __add__(self, other):
        """Concatenate two datasets."""
        if not isinstance(other, CrystalDataset):
            raise TypeError(f"Cannot add {type(other)} to CrystalDataset")
        new_dataset = self.__copy__()
        new_dataset.extend(other, check_duplicates=False, reset_caches=True, reset_entry_caches=True)
        return new_dataset


if __name__ == "__main__":
    pass
