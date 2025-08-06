from dataclasses import dataclass, field
from typing import Any, Set, Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Structure

try:
    import MySQLdb
except ImportError as err:
    raise ImportError("Install `mysqlclient` to use OQMDClient.") from err


@dataclass(slots=True)
class OQMDClient:
    """Query a local OQMD MySQL dump and return rows ready for ThermoDataset."""

    host: str = "localhost"
    user: str = "oqmd_user"
    passwd: str | None = "MaDnEsS!!"
    db: str = "oqmd"
    charset: str = "utf8mb4"

    _conn: Any = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _connect(self):
        if self._conn is None:
            self._conn = MySQLdb.connect(
                host=self.host,
                user=self.user,
                passwd=self.passwd,
                db=self.db,
                charset=self.charset,
            )
        return self._conn

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def query(self, elements: Iterable[str]) -> pd.DataFrame:
        elem_rgx = "|".join(sorted(set(elements)))
        list_rgx = rf"^({elem_rgx})_(?:({elem_rgx})_)*$"

        sql = f'''
        SELECT
            e.id            AS entry_id,
            c.formula       AS formula,
            cal.id          AS calc_id,
            cal.energy_pa   AS e_pa,
            cal.configuration,
            cal.output_id   AS structure_id,
            cal.settings    AS calc_settings,
            s.x1, s.y1, s.z1,
            s.x2, s.y2, s.z2,
            s.x3, s.y3, s.z3
        FROM entries e
        JOIN compositions c ON c.formula = e.composition_id
        JOIN calculations  cal ON cal.entry_id = e.id
        LEFT JOIN structures  s ON s.id = cal.output_id
        WHERE c.element_list REGEXP '{list_rgx}'
          AND (
                 cal.configuration = 'static'
              OR (cal.configuration IS NULL
                  AND cal.settings LIKE '%"xc": "PBE"%'
                  AND cal.settings RLIKE "'nsw'[[:space:]]*:[[:space:]]*0")
              );
        '''

        cur = self._connect().cursor()
        cur.execute(sql)
        df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        cur.close()
        if df.empty:
            return df

        # prefer static; keep one calc per entry
        df["pref"] = (df["configuration"] != "static").astype(int)
        df = (
            df.sort_values(["entry_id", "pref"])
            .groupby("entry_id", as_index=False)
            .first()
        )

        # build Structure objects
        df["structure"] = [self._build_structure(r) for _, r in df.iterrows()]
        df = df[df["structure"].notnull()]

        # energies and sizes
        df["natoms"] = [len(s) for s in df["structure"]]
        df["energy"] = df["e_pa"] * df["natoms"]

        # final column set expected by ThermoEntry.from_row
        df = df.rename(columns={"entry_id": "id"})
        df["metadata"] = [{"source": "OQMD"}] * len(df)
        df["id"] = "oqmd_" + df["id"].astype(str)
        return df[["id", "formula", "energy", "structure", "metadata"]]

    # Legacy alias kept for old scripts
    def lowest_energy(self, elements: Set[str]) -> pd.DataFrame:
        return self.query(elements)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_structure(self, row: pd.Series) -> Structure | None:
        if pd.isna(row.structure_id):
            return None

        cur = self._connect().cursor()
        cur.execute(
            "SELECT element_id, x, y, z FROM atoms "
            "WHERE structure_id=%s ORDER BY id;",
            (int(row.structure_id),),
        )
        atoms = cur.fetchall()
        cur.close()

        if not atoms:
            return None

        lattice = np.array(
            [
                (row.x1, row.y1, row.z1),
                (row.x2, row.y2, row.z2),
                (row.x3, row.y3, row.z3),
            ],
            dtype=float,
        ).T
        species = [a[0] for a in atoms]
        coords = [[a[1], a[2], a[3]] for a in atoms]
        return Structure(lattice, species, coords, coords_are_cartesian=False)
