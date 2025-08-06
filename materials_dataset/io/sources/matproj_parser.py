from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, List, Set, Iterable, Generator
import os

import pandas as pd
from pymatgen.core import Structure

try:
    from mp_api.client import MPRester as _BaseRester  # ≥0.9
except ImportError:                                     # older mp-api
    from mp_api.client import MPAPIRester as _BaseRester  # type: ignore

_DEFAULT_MP_KEY = "g1nTPIwz3KjPNNFt7lwkPdFammLCE66v"


@dataclass(slots=True)
class MPClient:
    """Materials Project v2 client (total-energy rows for ThermoDataset)."""

    api_key: str | None = None
    _mpr: Any = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _connect(self) -> _BaseRester:
        if self._mpr is None:
            key = self.api_key or os.getenv("MAPI_KEY") or _DEFAULT_MP_KEY
            self._mpr = _BaseRester(api_key=key)
        return self._mpr

    @staticmethod
    def _subspaces(elements: Set[str]) -> Generator[str, Any, None]:
        for r in range(1, len(elements) + 1):
            for combo in combinations(sorted(elements), r):
                yield "-".join(combo)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def query(self, elements: Iterable[str]) -> pd.DataFrame:
        """
        Return every calculation that contains *elements*.

        Columns: id, formula (full), e_total (eV), natoms, structure, source
        """
        rester = self._connect()

        thermo_docs = []
        for space in self._subspaces(set(elements)):
            thermo_docs.extend(
                rester.thermo.search(
                    chemsys=space,
                    thermo_types=['GGA_GGA+U'],
                    fields=["material_id", "formula_pretty", "energy_per_atom"],
                )
            )

        if not thermo_docs:
            return pd.DataFrame(
                columns=["id", "formula", "energy", "structure", "metadata"]
            )

        # One bulk structures call
        mids = {d.material_id for d in thermo_docs}
        struct_map = {
            s.material_id: s.structure
            for s in rester.summary.search(
                material_ids=list(mids), fields=["material_id", "structure"]
            )
        }

        rows = []
        for doc in thermo_docs:
            struct: Structure | None = struct_map.get(doc.material_id)
            if struct is None:
                continue
            natoms = len(struct)
            e_total = doc.energy_per_atom * natoms
            rows.append(
                {
                    "id": doc.material_id,
                    "formula": struct.composition.formula,
                    "energy": e_total,
                    "structure": struct,
                    "metadata": {"source": "MaterialsProject"},
                }
            )

        return pd.DataFrame(rows)

    # Legacy alias for old code paths
    def lowest_energy(self, elements: Set[str]) -> pd.DataFrame:
        return self.query(elements)
