from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Callable, NewType, Tuple, Set
from genutils.misc import describe_predicate
from .crystal_entry import CrystalEntry
from datetime import datetime

class CrystalDataset(Sequence[CrystalEntry]):

    def __init__(
            self,
            entries: Iterable[CrystalEntry],
            dataset_id: str | None = None,
            parent_ids: List[str, ...] | None = None,
            message='',
            base_path=None,
    ) -> None:
        self._entries = list(entries)
        self.metadata = {"message": message, "created_on": datetime.today().strftime('%Y-%m-%d %H:%M')}
        self.dataset_id = dataset_id or self._generate_id()
        self.parent_ids = parent_ids if parent_ids else []
        self._base_path = base_path or Path(os.getcwd())
        self._elements = None

    """Thin, read‑oriented container around CrystalEntry objects."""

    # --- Sequence protocol (read‑only operations) -----------------
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CrystalEntry]:
        return iter(self._entries)

    def __getitem__(self, idx) -> CrystalEntry:
        return self._entries[idx]

    def _generate_id(self) -> str:
        return hex(hash((id(self), datetime.today(), self.metadata["message"])))[2:]

    def override_base_path(self, new_path: str | Path):
        self._base_path = new_path

    @property
    def elements(self):
        """Set of elements in the dataset."""
        if not self._elements:
            self._elements = set(
                el for e in self if e.structure for el in e.composition.as_data_dict()['elements'])
        return self._elements

    @property
    def base_path(self):
        return self._base_path

    @classmethod
    def from_parents(cls, entries, parents, message, **kwargs):
        parent_ids = list(ds.dataset_id for ds in parents)
        if all([ds.base_path == parents[0].base_path for ds in parents]):
            base_path = parents[0].base_path
        else:
            base_path=kwargs.get("base_path", None)
        return cls(entries, parent_ids=parent_ids, message=message, base_path=base_path)

    # --- Minimal *pure* helpers that return new datasets ----------
    def merge(self, other: "CrystalDataset", message=None) -> "CrystalDataset":
        message = message or 'Simple merge'
        return CrystalDataset.from_parents(self._entries + list(other), parents=(self, other), message=message)

    def filter(self, predicate: Callable[[CrystalEntry], bool], message=None) -> "CrystalDataset":
        message = message or f"Filtered by {describe_predicate(predicate)}"
        return CrystalDataset.from_parents([e for e in self if predicate(e)], parents=(self,), message=message)

