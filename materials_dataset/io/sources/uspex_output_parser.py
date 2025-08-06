from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Iterable

import re
import pandas as pd
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Poscar

from ..uspex_bridge import USPEXBridge

@dataclass
class USPEXOutputClient:

    path: str | Path
    mode: str = 'goodStructures' # or 'calcFolds'
    stage: int | None = None

    def query(self, elements: Iterable[str]| None = None) -> pd.DataFrame:
        if self.mode == 'goodStructures':
            df = self.parse_goodStructures()
        elif self.mode == 'calcFolds':
            df = self.parse_calcFolders(self.stage)
        else:
            raise NotImplemented("Unknown mode of parsing")
        if elements is not None:
            allowed = set(elements)
            mask = df["structure"].apply(
                lambda s: set(s.composition.get_el_amt_dict().keys()).issubset(allowed)
            )
            df = df[mask].reset_index(drop=True)
        return df

    def parse_table(self, file_path: Path) -> pd.DataFrame:
        """
        Parse an ASCII pipe-delimited table into a DataFrame.
        Ignores decorative separator lines starting with '+'.
        """
        df = pd.read_csv(
            file_path,
            sep=r"\|",
            engine="python",
            comment="+",
            header=0
        )
        df.columns = df.columns.str.strip()
        df = df.loc[:, df.columns != ""]

        # Convert numeric columns
        for col in ["ID", "Rank", "Enthalpy (eV)",
                    "Enthalpy above CH (eV/block)", "Volume (A^3)"]:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def parse_goodStructures(self) -> pd.DataFrame:
        """
        Walk `folder`, parse each X_Y_Z table + X_Y_Z_POSCARS,
        and return a DataFrame with columns:
        ['id', 'formula', 'e_total', 'natoms', 'structure', 'source']
        """
        all_parts = []

        for table_fp in self.path.iterdir():
            if not table_fp.is_file():
                continue
            if table_fp.name.endswith("_POSCARS"):
                continue

            base = table_fp.stem
            df = self.parse_table(table_fp)

            # 1) rename and pick only the needed columns
            df = df.rename(columns={
                "ID":            "id",
                "Enthalpy (eV)": "energy"
            })[["id", "Composition", "energy"]]

            # 2) parse Composition → formula & natoms
            def _parse_comp(comp: str):
                parts = re.findall(r"([A-Za-z]+):\s*(\d+)", comp)
                formula = "".join(f"{el}{cnt}" for el, cnt in parts)
                natoms  = sum(int(cnt) for _, cnt in parts)
                return formula, natoms

            parsed = df["Composition"].apply(_parse_comp)
            df["formula"] = parsed.apply(lambda x: x[0])
            df["natoms"]  = parsed.apply(lambda x: x[1])
            df = df.drop(columns=["Composition"])

            pos_fp = self.path / f"{base}_POSCARS"
            structs = {}
            text = pos_fp.read_text()
            for blk in re.split(r'(?=^EA\d+)', text, flags=re.MULTILINE):
                blk = blk.strip()
                if not blk:
                    continue
                sid = int(re.match(r"EA(\d+)", blk).group(1))
                structs[sid] = Poscar.from_str(blk).structure

            df["structure"] = df["id"].map(structs)
            df["metadata"]    = {"source": base}

            all_parts.append(df[["id","formula","energy", "structure", "metadata"]])

        return pd.concat(all_parts, ignore_index=True)

    def parse_calcFolders(
            self,
            stage: Optional[int] = None,
            id_list_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Scan `base_folder` for subdirs named CalcFold<X>_<Y>.  For each unique X:
          – if y_choice is given, pick the folder with Y == y_choice (skip X if absent)
          – otherwise pick the folder with the maximum Y
        Any folder where CONTCAR or vasprun.xml fails to load is skipped.
        Returns a DataFrame with columns ["id","formula","energy","natoms","structure","source"].
        """
        ids_order = None
        if id_list_file:
            # If entries_order_file is provided, read it to get the order of entries
            ids_order = USPEXBridge.read_idlist(id_list_file)
        pattern = re.compile(r'^CalcFold(\d+)_([0-9]+)$')
        oszicar_F_pattern = re.compile(r"F=\s*([-\d.]+)")

        grouping: dict[int, list[tuple[int, Path]]] = {}
        for sub in self.path.iterdir():
            if not sub.is_dir():
                continue
            m = pattern.match(sub.name)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            grouping.setdefault(x, []).append((y, sub))

        records = []
        for x, y_folders in grouping.items():
            # choose folder for this X
            if stage is not None:
                matches = [p for (y, p) in y_folders if y == stage]
                if not matches:
                    continue
                folder = matches[0]
            else:
                _, folder = max(y_folders, key=lambda yp: yp[0])

            # try to load CONTCAR
            try:
                contcar = Poscar.from_file(folder / "CONTCAR")
                struct = contcar.structure
            except Exception as e:
                print(f"Skipping {folder.name}: failed to read CONTCAR ({e})")
                continue

            # extract formula & natoms from structure
            natoms = struct.num_sites
            comp_dict = struct.composition.get_el_amt_dict()
            formula = "".join(f"{el}{int(cnt)}" for el, cnt in comp_dict.items())

            # try to load vasprun.xml
            try:
                vr = Vasprun(str(folder / "vasprun.xml"), parse_potcar_file=False)
                energy = vr.final_energy
                print(f"{folder.name}: Energy read from vasprun.xml")
            except Exception as e:
                oszicar = folder / "OSZICAR"
                if not oszicar.exists():
                    print(f"{folder.name}: Couldn't read energy from Vasprun.xml. OSZICAR is missing. Skipping")
                    continue
                with open(oszicar, "r") as oszicar_fid:
                    for line in reversed(oszicar_fid.readlines()):
                        m = oszicar_F_pattern.search(line)
                        if m:
                            print(f"{folder.name}: Last energy read from OSZICAR. Failed to read vasprun.xml ({e}) ")
                            energy = m[1]
                            break
                    else:
                        print(f"{folder.name}: Couldn't read energy from either Vasprun.xml and OSZICAR. Skipping")
                        continue
            e_id = ids_order[x+1] if ids_order else x  # calcfolds are 1-based hence x + 1 !!!
            records.append({
                "id": e_id,
                "formula": formula,
                "energy": energy,
                "natoms": natoms,
                "structure": struct,
                "metadata": {"source": folder.name}
            })

        return pd.DataFrame(records, columns=["id", "formula", "energy", "structure", "metadata"])

# Usage:
# from pathlib import Path
# df = parse_folder(Path("goodStructures"))
# print(df)
