from pathlib import Path
from ..converters import df2ds
from .sources.matproj_parser import MPClient
from .sources.alexandria_parser import AlexandriaClient
from .sources.uspex_output_parser import USPEXOutputClient
from .sources.oqmd_parser import OQMDClient
from .sources.structure_and_energy_files_reader import CSV_and_POSCARS_client

mp_client = MPClient()
oqmd_client = OQMDClient()

def load_from_materials_project(elements, message=None, **kwargs):
    df = mp_client.query(elements)
    message = message or f"Full {elements} system from Materials Project"
    return df2ds(df, message=message)

def load_from_alexandria(elements, message=None, **kwargs):
    alex_client = AlexandriaClient(**kwargs)
    df = alex_client.query(elements)
    message = message or f"Full {elements} system from Alexandria database"
    return df2ds(df, message=message)

def load_from_oqmd(elements, message=None, **kwargs):
    df = oqmd_client.query(elements)
    message = message or f"Full {elements} system from OQMD"
    return df2ds(df, message=message)

def load_from_uspex_calc_folders(calcfolds_path: Path, stage=None, message=None, elements=None):
    uspex_client = USPEXOutputClient(calcfolds_path, stage=stage, mode='calcFolds')
    df = uspex_client.query(elements=elements)
    message = message or f"CalcFolds" +  (f' with elements={elements}' if elements else '.')
    return df2ds(df, message=message)

def load_from_uspex_goodstructures(goodstructures_path: Path, message=None, elements=None):
    uspex_client = USPEXOutputClient(goodstructures_path, mode='goodStructures')
    df = uspex_client.query(elements=elements)
    message = message or f"goodStructures" +  (f' with elements={elements}' if elements else '.')
    return df2ds(df, message=message)

def load_mattersim_estimated_set(csv_path: Path, poscars_base_path: Path, message=None):
    energy_struct_client = CSV_and_POSCARS_client(results_csv=csv_path, poscars_parent_path=poscars_base_path)
    df = energy_struct_client.query()
    message = message or f"Mattersim-estimated structures from {poscars_base_path.resolve()}"
    return df2ds(df, message=message)