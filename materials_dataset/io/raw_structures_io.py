from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from zipfile import ZipFile
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import tempfile
from materials_dataset_v2.crystal_dataset import CrystalDataset, CrystalEntry
from abInitio_io_parse.ext_software_io import read_poscars

SEARCH_PATTERNS = {'poscar': '*POSCAR*', 'cif': '*.cif', 'zip': '*.zip'}

def _unzip_nested(folder: Path):
    """Extract every .zip inside *folder* into a sibling directory (name w/o .zip)."""
    stack = [folder]
    while stack:
        current = stack.pop()
        for z in current.rglob("*.zip"):
            target = z.with_suffix("")        # e.g. foo.zip → foo/
            target.mkdir(exist_ok=True)
            with ZipFile(z) as zf:
                zf.extractall(target)
            stack.append(target)              # might hold more .zip files

def process_zips(root: str | Path, f: Callable):
    """Find every .zip under *root*, extract (handling nested zips), call f(), then clean up."""
    for z in Path(root).rglob("*.zip"):
        with tempfile.TemporaryDirectory(dir=z.parent) as tmp:
            tmp_path = Path(tmp)
            with ZipFile(z) as zf:
                zf.extractall(tmp_path)
            _unzip_nested(tmp_path)           # recurse over nested archives
            f(tmp_path)                       # run your operation
            # tmp_path is deleted automatically on exit, leaving tree untouched

@dataclass
class RawStructureFilesIO:
    path: Path
    format: str = 'POSCAR'
    zipped_format: str = 'cif'
    pattern: str = None

    def __post_init__(self):
        assert self.format.casefold() in SEARCH_PATTERNS,\
            f"format {self.format} not supported, available options: {list(SEARCH_PATTERNS.keys())}"
        self.pattern = SEARCH_PATTERNS[self.format.casefold()]
        self.zipped_pattern =  SEARCH_PATTERNS[self.zipped_format.casefold()]

    def get_dataset_from_dir(self, gathered_entries=None, id_suffix: str= '', **kwargs):
        gathered_entries = gathered_entries or []
        if self.format == 'zip':
            process_zips(self.path, lambda pth: self.gather_entries_from_dir(pth, self.zipped_pattern, gathered_entries))
        else:
            self.gather_entries_from_dir(self.path, self.pattern, gathered_entries, id_suffix)
        if "message" not in kwargs:
            kwargs['message'] = f'Structures only, gathered from {self.format}-files in {self.path}'
        return CrystalDataset(gathered_entries, **kwargs)

    @staticmethod
    def gather_dataset_from_id_list(files_vs_ids_file: Path | str | None = None, **kwargs):
        base_path=files_vs_ids_file.parent
        entries = []
        with open(files_vs_ids_file, 'rt') as ids:
            for line in ids:
                poscar, sid = line.split(',')
                entries.append(CrystalEntry(id = sid, structure=Structure.from_file(base_path / poscar)))


    @staticmethod
    def dump_poscar_files(dataset, path, files_vs_ids_file: Path | str | None = None):
        """
        Dumps poscar files of all structures
        :param dataset:
        :param path:
        :param files_vs_ids_file:
        """
        path.mkdir(exist_ok=True)
        for e in dataset:
            e.structure.to(fmt="poscar", filename=path / e.poscarname, comment=e.id)
        if files_vs_ids_file:
            with open(files_vs_ids_file, 'wt') as e_ids:
                e_ids.writelines([f"{e.poscarname},{e.id}" for e in dataset])

    @staticmethod
    def dump_multiimage_poscar(dataset, filename: Path | str | None = None, sort_key=None,
                               entries_ids_file: Path | str | None = None):
        """
        Saves POSCARS file containing poscar-format description of structures of all entries

        :param dataset:
        :param filename:
        :param sort_key: allows to sort
        :param entries_ids_file:
        """
        if filename is None:
            filename = Path.cwd() / "POSCARS"
        with open(filename, 'wt') as fid:
            sequence = dataset if not sort_key else sorted(dataset, key=sort_key)
            for e in sequence:
                poscar_str = Poscar(e.structure.to, comment=e.id).get_str()
                fid.write(poscar_str)
        if entries_ids_file:
            with open(entries_ids_file, 'wt') as order_file:
                for e in sequence:
                    order_file.write(f"{e.id}\n")

    @staticmethod
    def get_dataset_from_multiimage_poscar(poscars: Path, entries_ids_file: Path | None = None,
                                           id_prefix='', **kwargs):
        """
        :param poscars: path to a poscar file containing multiple structures
        :param entries_ids_file: text file containing entry ids each on new line with orders corresponding to
                                 the structures in poscars. To be used to retreive entries from e.g. uspex calculations
        :param id_prefix:
        :param kwargs:
        :return:
        """
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms_list = read_poscars(poscars)
        if entries_ids_file:
            with open(entries_ids_file) as ids_h:
                ids = ids_h.read().split('\n')
        else:
            ids = [f"{id_prefix}{i}" for i in range(len(atoms_list))]
        entries = \
            [CrystalEntry(id=id, structure=AseAtomsAdaptor.get_structure(ats)) for id, ats in zip(ids, atoms_list)]
        if "message" not in kwargs:
            kwargs['message'] = f'Structures only, gathered from multientry poscar file {poscars.as_posix()}'
        return CrystalDataset(entries, **kwargs)

    @staticmethod
    def gather_entries_from_dir(struc_path, pattern=None, gathered_entries=None, id_prefix=''):
        gathered_entries = gathered_entries or []
        for i, struc_file in  enumerate(struc_path.rglob(pattern)):
            sid = f"{id_prefix}{i}"
            try:
                structure = Structure.from_file(struc_file, merge_tol=0.01)
                gathered_entries.append(CrystalEntry(id=sid, structure=structure))
            except ValueError:
                continue
        return gathered_entries
