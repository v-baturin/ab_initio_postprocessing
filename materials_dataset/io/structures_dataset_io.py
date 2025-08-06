from __future__ import annotations

"""StructureDatasetIO – read/write utilities for CrystalDataset

Features
--------
* **Load** structures from a directory tree that may contain nested zip archives.
  Archive exploration is optional via *expand_archives* (default **True**).
* **Load** structures from a multi‑image POSCARS file.
* **Dump** individual POSCAR files for each entry.
* **Dump** a combined multi‑image POSCARS file (plus optional plain‑text ID list).

Zip handling (copy‑then‑explode with directory‑hierarchy preservation, akin to
`find … -name '*.zip' -exec cp {} --parents …`) is isolated in
`zip_handling.py` so that archive specifics stay out of this module.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Collection

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from ..crystal_dataset import CrystalDataset, CrystalEntry
from .zip_handling import exploded_zip_tree         # archive helper
from materials_tools.abInitio_io_parse.ext_software_io import read_poscars               # existing helper

LOG = logging.getLogger(__name__)

###############################################################################
# Internal helpers                                                           #
###############################################################################

def _safe_structure_from_file(path: Path, merge_tol: float = 0.01) -> Structure | None:
    """Return Structure or *None* on parser failure; logs the error."""
    try:
        return Structure.from_file(path, merge_tol=merge_tol)
    except Exception as exc:  # ValueError, RuntimeError, etc.
        LOG.warning("Skipping %s – %s", path, exc)
        return None


def _entry_id_sequence(prefix: str) -> Iterator[str]:
    """Yield id strings prefix0, prefix1, … endlessly."""
    counter = 0
    while True:
        yield f"{prefix}{counter}"
        counter += 1


def _iter_structure_files(root: Path, patterns: Sequence[str]) -> Iterator[Path]:
    """Yield files in *root* matching any of *patterns* (via rglob)."""
    for pattern in patterns:
        yield from root.rglob(pattern)

###############################################################################
# Public façade                                                              #
###############################################################################

@dataclass(slots=True)
class StructureDatasetIO:
    """Read and write `CrystalDataset` objects.

    Parameters
    ----------
    root : Path
        Source directory that may contain structure files and/or zip archives.
    patterns : Sequence[str], default ("*POSCAR*", "*.cif")
        Glob patterns that identify structure files **both** outside and inside
        archives.  Provide as many patterns as needed for your workflow.
    id_prefix : str, default ""
        Prefix for auto‑generated entry IDs.
    expand_archives : bool, default True
        If *True* the loader copies every ``*.zip`` (preserving directory tree)
        to a temporary area, recursively explodes them, and reads structures
        from that expanded tree **in addition to** structures already present
        on disk.  If *False* only files on disk are inspected.
    """

    root: Path
    patterns: Sequence[str] = ("*POSCAR*", "*.cif")
    id_prefix: str = ""
    expand_archives: bool = True
    source_name: str = "NA"
    _id_iter: Iterator[str] = field(init=False, repr=False)

    # --------------------------------------------------------------------- #
    # Initialisation                                                        #
    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._id_iter = _entry_id_sequence(self.id_prefix)

    # --------------------------------------------------------------------- #
    # Reader methods                                                        #
    # --------------------------------------------------------------------- #
    def load_from_directory(
        self,
        *,
        elements: Collection[str] | None = None,
        message: str | None = None,
        expand_archives: bool = True
    ) -> CrystalDataset:
        """Return a dataset built from *root*.

        Parameters
        ----------
        message : str | None
            Optional dataset message.
        expand_archives : bool | None
            Overrides the instance‑level ``expand_archives`` flag.
        """
        use_archives = self.expand_archives if expand_archives is None else expand_archives

        entries: list[CrystalEntry] = []

        # 1) structures already on disk (outside archives)
        for file in _iter_structure_files(self.root, self.patterns):
            struct = _safe_structure_from_file(file)
            if struct:
                entries.append(CrystalEntry(id=next(self._id_iter), structure=struct,
                                            metadata={"source": self.source_name}))

        # 2) structures inside archives (if requested)
        if use_archives:
            with exploded_zip_tree(self.root) as tmp_root:
                for file in _iter_structure_files(tmp_root, self.patterns):
                    struct = _safe_structure_from_file(file)
                    if struct:
                        entries.append(CrystalEntry(id=next(self._id_iter), structure=struct,
                                                    metadata={"source": self.source_name}))
        if elements:
            entries = [e for e in entries if not (set(e.composition.as_data_dict()["elements"]) - set(elements))]
        msg = message or f"Structures collected from {self.root}"
        return CrystalDataset(entries, message=msg)

    def load_from_multiimage_poscar(
        self,
        poscars_file: Path,
        entries_ids_file: Path | None = None,
        *,
        message: str | None = None,
    ) -> CrystalDataset:
        """Return a dataset from a multi‑image POSCARS file."""
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms_list = read_poscars(poscars_file)
        if entries_ids_file:
            ids = Path(entries_ids_file).read_text().splitlines()
            if len(ids) != len(atoms_list):
                raise ValueError("Number of IDs does not match number of structures.")
        else:
            ids = [next(self._id_iter) for _ in range(len(atoms_list))]

        entries = [
            CrystalEntry(id=id_, structure=AseAtomsAdaptor.get_structure(ats))
            for id_, ats in zip(ids, atoms_list)
        ]
        msg = message or f"Structures loaded from {poscars_file.as_posix()}"
        return CrystalDataset(entries, message=msg)

    # --------------------------------------------------------------------- #
    # Writer methods                                                        #
    # --------------------------------------------------------------------- #
    @staticmethod
    def dump_poscar_files(
        dataset: Iterable[CrystalEntry],
        out_dir: Path,
    ) -> None:
        """Write each entry to its own POSCAR file in *out_dir*."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in dataset:
            filename = out_dir / entry.poscarname
            entry.structure.to(fmt="poscar", filename=filename, comment=entry.id)

    @staticmethod
    def dump_multiimage_poscar(
        dataset: Iterable[CrystalEntry],
        filename: Path,
        *,
        sort_key=None,
    ) -> None:
        """Write a POSCARS multi‑image file; optional plain‑text ID list."""
        sequence = sorted(dataset, key=sort_key) if sort_key else list(dataset)
        with open(filename, "wt") as fid:
            for entry in sequence:
                poscar_str = Poscar(entry.structure, comment=entry.id).get_str()
                fid.write(poscar_str)