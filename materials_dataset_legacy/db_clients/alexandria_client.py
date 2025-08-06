from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Iterable

import pandas as pd
from pymatgen.core import Structure

try:
    import ijson
except ImportError as err:
    raise ImportError("Install `ijson` to use AlexandriaClient.") from err


@dataclass(slots=True)
class AlexandriaClient:
    """Read *.json* snapshots of the Alexandria database."""

    root: Path | str = Path("~/work/Alexandria").expanduser()
    pattern: str = "alexandria*.json"

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _files(self) -> List[Path]:
        files = sorted(Path(self.root).glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No Alexandria files in {self.root}")
        return files

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def query(self, elements: Iterable[str]) -> pd.DataFrame:
        """Return rows that contain *only* the requested elements."""
        wanted = set(elements)
        rows: list[dict] = []

        for file in self._files():
            print(f"Parsing {file} ...")
            with file.open() as fh:
                for ent in ijson.items(fh, "entries.item"):
                    data = ent.get("data", {})
                    # try quick pre-filter on the raw string
                    raw_formula = data.get("formula", "")
                    if raw_formula and set(_split_formula(raw_formula)) - wanted:
                        continue

                    # build Structure
                    struct = Structure.from_dict(_convert_numbers(ent["structure"]))
                    struct_elems = {el.symbol for el in struct.composition.elements}
                    if struct_elems - wanted:
                        continue  # actual structure has unwanted elements

                    natoms = len(struct)
                    e_total = (
                        float(data["energy_total"])
                        if "energy_total" in data
                        else float(data.get("energy_pa") or data.get("energy_corrected")) * natoms
                    )

                    full_formula = struct.composition.formula
                    rows.append(
                        {
                            "id": data.get("mat_id"),
                            "formula": full_formula,
                            "e_total": e_total,
                            "natoms": natoms,
                            "structure": struct,
                            "source": "alex",
                        }
                    )

        return pd.DataFrame(rows)

    # legacy alias
    def lowest_energy(self, elements: Set[str]) -> pd.DataFrame:  # noqa: D401
        return self.query(elements)


# --------------------------- helpers ------------------------------------ #

def _split_formula(formula: str) -> List[str]:
    """Return list of element symbols in *formula* (e.g. 'B2Mo5Si1' → ['B', 'Mo', 'Si'])."""
    import re

    return re.findall(r"[A-Z][a-z]?", formula)


def _convert_numbers(obj):
    """Recursively cast `decimal.Decimal` values to float so pymatgen can read them."""
    from decimal import Decimal

    if isinstance(obj, dict):
        return {k: _convert_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numbers(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    return obj
