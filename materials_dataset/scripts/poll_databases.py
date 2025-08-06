from typing import Dict
from collections import defaultdict
from ..crystal_dataset import CrystalDataset
from ..io.preset_loaders import load_from_alexandria, load_from_oqmd, load_from_materials_project
from ..analysis.phase_diagram_tools import PhaseDiagramTools


MAX_EHULL = 0.02  # Maximum energy above hull in eV/atom for filtering
LOADERS = {"al": load_from_alexandria, "oq": load_from_oqmd, "mp": load_from_materials_project,
           "ma": load_from_materials_project}  # List of database clients to fetch data from
# ------------------------------------------------------------------ #

def poll_databases(elements,
                   database_names=None,
                   ref_db: str = 'al',
                   do_ehull_filtering=True,
                   do_deduplication=True,
                   max_ehull=None,
                   loader_kwargs: Dict | None =None):
    """
    Fetch data from Alexandria, OQMD, and Materials Project databases.
    Data is first taken from the reference database (Alexandria) then only the structures unseen in the reference DB
    are added.
    """
    if loader_kwargs is None: loader_kwargs = dict()

    if do_deduplication:
        from ..analysis.similarity_tools import SimilarityTools
        from ..io.uspex_bridge import USPEXBridge
        ub = USPEXBridge(elements)
        similarity_tk = SimilarityTools(ub.fp_dist)

    ref_db = ref_db[:2].casefold()
    database_names = database_names or ['alexandria', 'oqmd', 'MatProj']  # Default databases to fetch data from
    short_names_dict = {name[:2].casefold(): name for name in database_names}
    loader_kw = defaultdict(dict, {k.casefold()[:2]: v for k, v in loader_kwargs.items()})
    for db in short_names_dict:
        assert isinstance(db, str), "Database names must be strings."
        assert db in LOADERS, f"Database '{short_names_dict[db]}' is not recognized. Available databases: {list(LOADERS.keys())}."
    # other_loaders = [LOADERS[short_name] for short_name in short_names_dict if not short_name.startswith(ref_db)]
    short_names_dict.pop(ref_db)
    reference_loader = LOADERS[ref_db]

    reference_data = reference_loader(elements, **loader_kw[ref_db])

    if do_ehull_filtering:
        max_ehull = max_ehull or MAX_EHULL
        pd_tools = PhaseDiagramTools(reference_data)
        reference_data = reference_data.filter(lambda e: pd_tools.height_above_hull_pa(e) < max_ehull)

    for loader in short_names_dict: # Skip the first loader as it's already processed
        print(f"Fetching data from {short_names_dict[loader]}...")
        loaded_ds = LOADERS[loader](elements, **loader_kw[loader])
        pd_tools = PhaseDiagramTools(loaded_ds)
        if do_ehull_filtering:
            loaded_ds = loaded_ds.filter(lambda e: pd_tools.height_above_hull_pa(e) < max_ehull)
        if do_deduplication:
            unseen = similarity_tk.get_unseen_in_ref(loaded_ds, reference_data)
            loaded_ds = unseen
        reference_data = reference_data.merge(loaded_ds)
    msg = (f"Gathered from {', '.join(database_names)} databases "
           f"with elements: {', '.join(elements)}")
    return CrystalDataset([e.copy_with(**{'energy': None}) for e in reference_data], message=msg)
