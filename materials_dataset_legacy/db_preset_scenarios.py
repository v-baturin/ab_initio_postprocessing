from datetime import datetime
import numpy as np
from materials_tools.materials_dataset.crystalDataset import CrystalDataset
from materials_tools.materials_dataset.db_clients.alexandria_client import AlexandriaClient
from materials_tools.materials_dataset.db_clients.oqmd_client import OQMDClient
from materials_tools.materials_dataset.db_clients.mp_client import MPClient

elements = ["Mo", "Si", "B", "P"]
do_ehull_filtering = True  # Whether to filter entries based on energy above hull
do_deduplication = True  # Whether to deduplicate entries based on composition and structure
MAX_EHULL = 0.02  # Maximum energy above hull in eV/atom for filtering
CLIENTS = {"al": AlexandriaClient(), "oq": OQMDClient(), "mp": MPClient(), "ma": MPClient()}  # List of database clients to fetch data from
# ------------------------------------------------------------------ #

def gather_entries_from_databases(elements,
                                  database_names=None,
                                  do_ehull_filtering=True,
                                  do_deduplication=True,
                                  max_ehull=None, **kwargs):
    """Fetch data from Alexandria, OQMD, and Materials Project databases."""
    database_names = database_names or ['alexandria', 'oqmd', 'MatProj']  # Default databases to fetch data from
    if 'pattern' in kwargs:
        CLIENTS['al'] = AlexandriaClient(pattern=kwargs['pattern'])
    for db in database_names:
        assert isinstance(db, str), "Database names must be strings."
        assert db[:2].lower() in CLIENTS, f"Database '{db}' is not recognized. Available databases: {list(CLIENTS.keys())}."

    clients = [CLIENTS[db[:2].lower()] for db in database_names]
    if not do_ehull_filtering:
        max_ehull = np.inf
    else:
        max_ehull = max_ehull or MAX_EHULL
    reference_client = clients[0]
    reference_data = CrystalDataset.from_client(reference_client, elements, skip_dump=True, **kwargs)
    if do_ehull_filtering:
        reference_data = reference_data.filter(
        lambda e: e.e_above_hull < max_ehull, reset_entries=False, skip_dump=True)

    for client in clients[1:]: # Skip the first client as it's already processed
        print(f"Fetching data from {client.__class__.__name__}...")
        filtered_ds = CrystalDataset.from_client(client, elements, skip_dump=True)
        if do_ehull_filtering:
            filtered_ds = filtered_ds.filter(
                lambda e: e.e_above_hull < max_ehull, reset_entries=False, skip_dump=True)
        reference_data.extend(filtered_ds,
                              reset_entry_caches=False, reset_caches=True, check_duplicates=do_deduplication,
                              skip_dump=True,
                              **kwargs)
    reference_data.parents = None
    reference_data.metadata["message"] = (f"{datetime.today().strftime('%Y-%m-%d %H:%M')}"
                                          f" - Gathered from {', '.join(database_names)} databases with elements: {', '.join(elements)}")
    return reference_data
#
# dump_basename = f"allDB_{"".join(elements)}_data_{'filtered' if do_ehull_filtering else 'RAW'}"  # Name of the pickle file to save the dataset
# # dump_basename = "test"
#
# ## Fetch data from Alexandria, OQMD, and Materials Project databases
# db_entries = []
# reference_client = AlexandriaClient()  #pattern='alexandria_*.json')
# reference_data = CrystalDataset.from_client(reference_client, elements).filter(lambda e: e.e_above_hull < 0.01,
#                                                                                reset_entries=False)#
# for client in [OQMDClient(), MPClient()]:
#     print(f"Fetching data from {client.__class__.__name__}...")
#     filtered_ds = CrystalDataset.from_client(client, elements).filter(lambda e: e.e_above_hull < 0.01, reset_entries=False)
#     reference_data.extend(filtered_ds,
#                           reset_entry_caches=False, reset_caches=True, check_duplicates=True, tol_FP=0.07)
#
# for e in reference_data:
#     e.energy_total = None  # Set energy to None to remove it from the dataset
#     print(f"{e.origin_id}: energy cleared")
#
# db_data, _, _ = reference_data.deduplicated(enforce_compositions_separation=True,
#                                             fitness_list=[e.e_above_hull for e in reference_data],
#                                             reset_entries=False)
#
# db_data.present_as_table(f"{dump_basename}_table.txt",
#                          sort_by="Energy above hull (eV/atom)")
#
# with open(f"{dump_basename}.pkl", "wb") as f:
#     db_data._reset_entry_caches()
#     db_data._reset_caches()
#     pkl.dump(db_data, f)
#     print(f"Data of {len(db_data)} entries saved to {dump_basename}.pkl")
#
# with open(f"{dump_basename}.pkl", "rb") as f:
#     db_data = pkl.load(f)
#     print(f"Data loaded from {dump_basename}.pkl")
#
# db_data.write_poscar_files(f"POSCAR_{dump_basename}")