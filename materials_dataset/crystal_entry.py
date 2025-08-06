from typing import Any
from pymatgen.core import Structure, Composition
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CrystalEntry:

    id: str
    structure: Structure
    energy: float | None = None
    formula: str | None = None
    metadata: dict[str, Any]  = field(default_factory=dict)

    def __post_init__(self):
        if self.formula is None:
            derived = self.structure.composition.formula
            object.__setattr__(self, "formula", derived)

    def __hash__(self):
        return hash((self.id, (self.structure.to_json() if self.structure else None), self.formula))

    @property
    def natoms(self) -> int:
        if self.structure:
            return len(self.structure)
        else:
            return int(Composition(self.formula).num_atoms)

    @property
    def poscarname(self):
        return f"{self.id}POSCAR"

    @property
    def composition(self) -> Composition:
        if self.structure:
            return self.structure.composition
        else:
            return Composition(self.formula)

    def log_message(self, message):
        self.metadata["log"] = self.metadata.get("log", '') + message + '\n'

    def copy_with(self, **kw) -> "CrystalEntry":
        # start from existing field values …
        data = {
            "id": self.id,
            "structure": self.structure,
            "energy": self.energy,
            "metadata": self.metadata,
        }
        # overwrite whichever ones the caller supplied
        data.update(kw)
        return CrystalEntry(**data)


    # ------------------------------------------------------------------
    # Constructors from external sources
    # ------------------------------------------------------------------


