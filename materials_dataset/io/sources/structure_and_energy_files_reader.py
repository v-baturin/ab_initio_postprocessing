import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from pymatgen.core import Structure


@dataclass
class CSV_and_POSCARS_client:
    results_dir_path: str | Path | None = None  # csv file or folder containing (only) alignn format csv
    results_csv: str | Path | None = None
    poscars_parent_path: str | Path | None = None
    source: str = "ALIGNN"
    poscars_parent_path = poscars_parent_path or Path(os.getcwd())
    per_atom_energy = ["ALIGNN"]

    def append_info(self, df, csv):
        df_temp = pd.read_csv(csv, dtype={'Energy': float})
        current_path = csv.parent
        df_temp["id"] = df_temp["File"].apply(lambda x: f"{current_path.relative_to(csv.parent)}/{x}")
        df_temp["structure"] = df_temp["File"].apply(lambda x: Structure.from_file(current_path /
                                                                                   self.poscars_parent_path / x))
        df_temp["formula"] = df_temp["structure"].apply(lambda x: x.composition.formula)
        df_temp["natoms"] = df_temp["structure"].apply(lambda x: len(x))
        df_temp["energy"] = df_temp["Energy"] * (df_temp["natoms"] if self.source in self.per_atom_energy else 1)
        df_temp["metadata"] = {"source": self.source}
        df = pd.concat([df, df_temp], ignore_index=True)
        return df


    def query(self) -> pd.DataFrame:
        assert self.results_csv or self.results_dir_path
        df = pd.DataFrame(columns=['id', 'energy', 'structure', 'metadata'])
        if self.results_csv:
            df = self.append_info(df, self.results_csv)
        elif self.results_dir_path:
            for current_csv in self.results_dir_path.rglob('*.csv'):
                df = self.append_info(df, current_csv)
        df.drop(columns=["Energy", "File", "natoms"], inplace=True)
        return df
