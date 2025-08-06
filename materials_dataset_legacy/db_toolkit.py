# phase_toolkit.py – unified access to OQMD, Alexandria & Materials Project
"""Materials‑database helper (Python ≥ 3.12)
====================================================
* **OQMD** – local MySQL
* **Alexandria** – JSON shards
* **Materials Project** – mp‑api (v2, R2SCAN‑aware)

All three back‑ends export the exact same columns:

``id, formula, e_pa, structure, source``

where **`e_pa` is the total energy per atom** at the highest available level
(R2SCAN → GGA + U → GGA).

Main helpers
------------
* `get_lowest_energy_set(elements, source)` — most stable polymorph of every
  formula containing *elements* (includes binaries & elements for a ternary).
* `build_summary(...)` — adds space group & non‑equivalent site counts and can
  optionally dump POSCARs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from pathlib import Path
import logging
import os
import re

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt, ternary


SYMPREC = 6.4e-4

# ---------------- optional back‑ends ---------------------------------------
try:
    import MySQLdb  # type: ignore
except ModuleNotFoundError:
    MySQLdb = None

try:
    from mp_api.client import MPRester as MPAPIRester  # type: ignore
except ModuleNotFoundError:
    MPAPIRester = None

try:
    import ijson  # type: ignore
except ModuleNotFoundError:
    ijson = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger(__name__)

# ---------------- helper utils ---------------------------------------------

def _convert_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, list):
        return [_convert_decimals(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _convert_decimals(v) for k, v in obj.items()}
    return obj


def _spacegroup(struct: Structure) -> tuple[str, int]:
    sga = SpacegroupAnalyzer(struct, symprec=SYMPREC)
    try:
        return sga.get_space_group_info()
    except AttributeError:  # very old pymatgen
        return sga.get_space_group_symbol(), sga.get_space_group_number()


def _neq_sites(struct: Structure) -> Counter[str]:
    symm = SpacegroupAnalyzer(struct, symprec=SYMPREC).get_symmetrized_structure()
    cnt: Counter[str] = Counter()
    for group in symm.equivalent_sites:
        cnt[group[0].specie.symbol] += 1
    return cnt

# ================= OQMD =====================================================
@dataclass(slots=True)
class OQMDClient:
    host: str = "localhost"
    user: str = "oqmd_user"
    passwd: str | None = "MaDnEsS!!"
    db: str = "oqmd"
    charset: str = "utf8mb4"
    _conn: any = field(init=False, default=None, repr=False)

    def _connect(self):
        if self._conn is None:
            if MySQLdb is None:
                raise ImportError("`mysqlclient` required for OQMD backend")
            self._conn = MySQLdb.connect(host=self.host, user=self.user, passwd=self.passwd, db=self.db, charset=self.charset)
            log.info("Connected to local OQMD")
        return self._conn

    def lowest_energy(self, elements: set[str]) -> pd.DataFrame:
        elem_rgx = "|".join(sorted(elements))
        list_rgx = rf"^({elem_rgx})_(?:({elem_rgx})_)*$"
        sql = f"""
        SELECT  e.id   AS id,
                c.formula,
                cal.energy_pa     AS e_pa,
                cal.output_id     AS structure_id,
                s.x1, s.y1, s.z1, s.x2, s.y2, s.z2, s.x3, s.y3, s.z3
        FROM    calculations cal
        JOIN    entries      e  ON e.id = cal.entry_id
        JOIN    compositions  c ON c.formula = cal.composition_id
        LEFT JOIN structures  s ON s.id = cal.output_id        -- ← exact lattice for that calc
        WHERE (
                 cal.configuration = 'static'
              OR (cal.configuration IS NULL
                  AND cal.settings LIKE  '%''xc'': ''PBE''%'
                  AND cal.settings RLIKE "'nsw'[[:space:]]*:[[:space:]]*0")
              )
          AND c.element_list REGEXP '{list_rgx}';
        """
        cur = self._connect().cursor()
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        cur.close()
        df = df.loc[df.groupby("formula").e_pa.idxmin()].reset_index(drop=True)
        df["source"] = "oqmd"
        return df

    def structure(self, row: pd.Series) -> Structure | None:
        if pd.isna(row.structure_id):
            return None
        cur = self._connect().cursor()
        cur.execute("SELECT element_id, x, y, z FROM atoms WHERE structure_id=%s ORDER BY id;", (int(row.structure_id),))
        atoms = cur.fetchall()
        cur.close()
        if not atoms:
            return None
        lattice = np.array([(row.x1, row.y1, row.z1), (row.x2, row.y2, row.z2), (row.x3, row.y3, row.z3)]).T.astype(str)
        species = [a[0] for a in atoms]
        coords = [[a[1], a[2], a[3]] for a in atoms]
        return Structure(lattice, species, coords, coords_are_cartesian=False)

# ================= Alexandria ==============================================
@dataclass(slots=True)
class AlexandriaClient:
    root: Path | str = Path("~/work/Alexandria").expanduser()
    pattern: str = "alexandria*.json"

    def _files(self):
        files = sorted(Path(self.root).glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"No Alexandria files in {self.root}")
        return files

    def lowest_energy(self, elements: set[str]) -> pd.DataFrame:
        if ijson is None:
            raise ImportError("Install `ijson` for Alexandria backend")
        el_pat = re.compile(r"[A-Z][a-z]?")
        rows: list[dict] = []
        for file in self._files():
            log.debug(f"Reading {file.name}")
            with file.open() as fh:
                for ent in ijson.items(fh, "entries.item"):
                    formula = ent.get("data", {}).get("formula", "")
                    if set(el_pat.findall(formula)) - elements:
                        continue
                    log.debug(f"Found {formula}")
                    data = ent.get("data", {})
                    e_pa = data.get("energy_pa") or data.get("energy_corrected") or float("nan")
                    structure = Structure.from_dict(_convert_decimals(ent["structure"]))
                    rows.append({
                        "id": data.get("mat_id"),
                        "formula": formula,
                        "e_pa": e_pa / len(structure),
                        "structure": structure,
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.loc[df.groupby("formula").e_pa.idxmin()].reset_index(drop=True)
        df["source"] = "alex"
        return df

# ================= Materials Project (mp‑api) ==============================
_DEFAULT_MP_KEY = "g1nTPIwz3KjPNNFt7lwkPdFammLCE66v"
@dataclass(slots=True)
class MPClient:
    """Materials Project v2 client using mp-api **thermo** endpoint for total energies.

    Prefers GGA_GGA+U_R2SCAN → GGA_GGA+U → R2SCAN.
    """
    api_key: str | None = None
    _rester: any = field(init=False, default=None, repr=False)

    def _mpr(self):
        if self._rester is None:
            if MPAPIRester is None:
                raise ImportError("Install mp-api: pip install mp-api")
            key = self.api_key or _DEFAULT_MP_KEY or os.getenv("MAPI_KEY", "")
            if not key:
                raise ValueError("Materials Project API key missing")
            self._rester = MPAPIRester(key)
            log.info("Connected to Materials Project via mp-api")
        return self._rester

    @staticmethod
    def _subspaces(elements: set[str]):
        for r in range(1, len(elements) + 1):
            for combo in combinations(sorted(elements), r):
                yield "-".join(combo)

    def lowest_energy(self, elements: set[str]) -> pd.DataFrame:
        prefer_order = ["R2SCAN"]
        mpr = self._mpr()
        thermo = mpr.materials.thermo

        # 1) fetch thermo docs for each sub-chemistry
        thermo_docs = []
        for cs in self._subspaces(elements):
            thermo_docs.extend(
                thermo.search(
                    chemsys=cs,
                    thermo_types=prefer_order,
                    fields=["material_id", "formula_pretty", "thermo_type", "energy_per_atom"],
                )
            )
        if not thermo_docs:
            return pd.DataFrame(columns=["id", "formula", "e_pa", "structure", "source"])

        # 2) fetch structures in one go
        mids = {d.material_id for d in thermo_docs}
        struct_map = {
            s.material_id: s.structure
            for s in mpr.summary.search(
                material_ids=list(mids),
                fields=["material_id", "structure"],
            )
        }

        # 3) pick best level per material
        ids, formulas, energies, structs = [], [], [], []
        seen = set()
        for level in prefer_order:
            for d in thermo_docs:
                if d.material_id in seen or d.thermo_type != level:
                    continue
                seen.add(d.material_id)
                ids.append(d.material_id)
                formulas.append(d.formula_pretty)
                energies.append(d.energy)
                structs.append(struct_map.get(d.material_id))

        # 4) build DataFrame and choose lowest per formula
        df = pd.DataFrame({
            "id": ids,
            "formula": formulas,
            "e_pa": energies,
            "structure": structs,
        })
        df = df.loc[df.groupby("formula").e_pa.idxmin()].reset_index(drop=True)
        df["source"] = "mp"
        return df

# ================= Facade ==================================================

class MaterialsToolkit:
    """High‑level wrapper selecting a single backend on each call."""

    def __init__(self, *, oqmd: dict | None = None, alex: dict | None = None, mp: dict | None = None):
        self.oqmd = OQMDClient(**oqmd) if oqmd is not None else None
        self.alex = AlexandriaClient(**alex) if alex is not None else None
        self.mp   = MPClient(**mp)   if mp   is not None else None
        if not any((self.oqmd, self.alex, self.mp)):
            raise ValueError("At least one backend must be configured")

    # ---------- convex‑hull set ----------
    def get_lowest_energy_set(self, elements: set[str], *, source: str = "oqmd") -> pd.DataFrame:
        match source:
            case "oqmd" if self.oqmd:
                return self.oqmd.lowest_energy(elements)
            case "alex" if self.alex:
                return self.alex.lowest_energy(elements)
            case "mp" if self.mp:
                return self.mp.lowest_energy(elements)
            case _:
                raise ValueError(f"Backend '{source}' not configured")

    # ---------- summary + POSCARs ----------
    def build_summary(
        self,
        elements: set[str],
        *,
        source: str = "oqmd",
        write_poscars: bool = False,
        poscar_dir: Path | str = "poscars",
    ) -> pd.DataFrame:
        df = self.get_lowest_energy_set(elements, source=source)
        pdir = Path(poscar_dir)
        recs: list[dict] = []

        for _, row in df.iterrows():
            # 1. retrieve Structure
            if source == "oqmd":
                struct = self.oqmd.structure(row)  # type: ignore[arg-type]
            else:
                struct = row.structure
            if struct is None:
                log.warning("No structure for %s (%s) — skipped", row.formula, row.id)
                continue

            # 2. crystal features
            sg_sym, sg_num = _spacegroup(struct)
            neq = _neq_sites(struct)
            n_atoms = len(struct)

            # 3. optional POSCAR dump
            if write_poscars:
                pdir.mkdir(parents=True, exist_ok=True)
                (pdir / f"{row.formula.replace(' ', '')}_{row.id}.POSCAR").write_text(struct.to(fmt="poscar"))

            # 4. collect record — only once!
            rec = {
                "id": row.id,
                "formula": row.formula,
                "e_pa": float(row.e_pa),
                "N": n_atoms,
                "sg_symbol": sg_sym,
                "sg_number": sg_num,
                **neq,
            }
            recs.append(rec)

        out = pd.DataFrame(recs)
        out["source"] = source

        # keep only per‑element columns that intersect requested elements
        elem_cols = [c for c in out.columns if len(c) <= 2 and c[0].isupper() and c != "N" ]
        for col in elem_cols:
            if col not in elements:
                out.drop(columns=col, inplace=True)

        # cast element count columns to int
        elem_cols = [c for c in out.columns if c in elements]
        for col in elem_cols:
            out[col] = out[col].fillna(0).astype(int)

        out.dropna(axis=1, how="all", inplace=True)  # remove empty cols
        return out



# import pandas as pd, numpy as np, matplotlib.pyplot as plt, ternary
# from pymatgen.core import Composition
# from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

def ternary_hull(df: pd.DataFrame, elems: list[str]) -> None:
    """Plot ternary convex hull outline and labels (python-ternary)."""
    # from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    #

    # --- compositions, triplets, total energies --------------------
    comps, triplets, entries = [], [], []
    for f, epa in zip(df.formula, df.e_pa):
        comp = Composition(f.replace(" ", ""))
        n    = sum(comp.values())
        comps.append(comp)
        triplets.append(tuple(comp[el] / n for el in elems))
        entries.append(PDEntry(comp, epa * n))        # total E

    pdg = PhaseDiagram(entries)

    # --- edges from triangular facets ------------------------------
    edges = {tuple(sorted((int(a), int(b))))
             for tri in pdg.facets
             for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    hull_idx = sorted({i for e in edges for i in e})

    # --- plot -------------------------------------------------------
    fig, tax = ternary.figure(scale=1); fig.set_size_inches(6, 6)
    norm, cmap = plt.Normalize(df.e_pa.min(), df.e_pa.max()), plt.cm.viridis

    for trip, e in zip(triplets, df.e_pa):                     # points
        tax.scatter([trip], color=[cmap(norm(e))], s=25)

    for i, j in edges:                                         # edges
        tax.line(triplets[i], triplets[j], lw=2, color='k')

    for idx in hull_idx:                                       # vertices
        tax.scatter([triplets[idx]], color='k', s=70)
        tax.annotate(comps[idx].reduced_formula, triplets[idx],
                     fontsize=7, ha='center', va='center')

    tax.left_corner_label(elems[0]); tax.right_corner_label(elems[1])
    tax.top_corner_label(elems[2]);  tax.boundary()
    tax.clear_matplotlib_ticks()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=tax.get_axes(), pad=0.05, label='Energy per atom (eV)')
    plt.show()




# ================= demo ====================================================

if __name__ == "__main__":
    SOURCE = "mp"            # "oqmd" | "alex" | "mp"
    ELEMENTS = {"Al", "Fe", "Ni"}
    WRITE_POSCARS = True

    tk = MaterialsToolkit(**{SOURCE: {}})
    df = tk.build_summary(ELEMENTS, source=SOURCE, write_poscars=WRITE_POSCARS)
    pd.set_option("display.max_columns", None)
    # format floats to 6 decimal places
    pd.set_option("display.float_format", "{:.6f}".format)
    # print with formatting
    print(df.to_string(index=False))
    ternary_hull(df, list(ELEMENTS))
