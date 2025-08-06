import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ..crystal_dataset import CrystalDataset, CrystalEntry
from ..converters import cell_pos_atomtypes_from_pmg_structure, pmg_structure_to_torch_cell_frac_atomnumbers
# from my_packages.materials_tools.materials_dataset.db_clients
from materials_tools.geom_utils.coordination_tools import compute_species_pair_atoms, compute_species_pair
import pickle as pkl

matplotlib.use('TkAgg')

# with open("/home/vsbat/SYNC/00__WORK/2025-2026_MOLTEN_SALTS/Reference_data/alexandria_data/alexLiCoO_whole.pkl", 'br') as pkldump_fid:
#     alex_data_LiCoO = pkl.load(pkldump_fid)
# alex_data_LiCoO.filter(lambda e: e.e_above_hull < 0.1, reset_entries=False)
# atom_A, atom_B = "Co", "O"
# dataset = alex_data_LiCoO

# dataset = CrystalDataset.from_struct_folder("BaCuSiP", search_pattern='*.cif')
# atom_A, atom_B = "Cu", "P"
#
# dataset = CrystalDataset.from_struct_folder("/home/vsbat/SYNC/00__WORK/2025-2026_MOLTEN_SALTS/STRUCTURE_GENERATION/"
#                                         "Mattergen_Auguste/2025_07_BATCH_CuSiBP/all_extracted", search_pattern='*.cif')



def analyze_environment(dataset, atom_A, atom_B, csv_output, **kwargs):
    coordinations = dict()
    coordinations_data = {"ID":[], "Coordination":[]}
    for e in dataset:
        ats = e.structure.to_ase_atoms()
        if not (atom_A in ats.get_chemical_symbols() and atom_B in ats.get_chemical_symbols()):
            continue
        n_neighbours = compute_species_pair_atoms(ats, atom_A, atom_B, kernel="sigmoid", alpha=100, **kwargs)
        coordinations[n_neighbours.round().int().item()] = coordinations.get(n_neighbours.round().int().item(), 0) + 1
        coordinations_data["ID"].append(e.id)
        coordinations_data["Coordination"].append(n_neighbours.round().int().item())
    pd.DataFrame(coordinations_data).to_csv(csv_output, index=False)
    n_systems = np.sum(list(coordinations.values()))
    plt.figure()
    plt.bar(coordinations.keys(), [i / n_systems for i in coordinations.values()])
    plt.xlabel('# of neighbors')
    plt.title(f"{atom_A} - {atom_B} statistics")
    plt.ylabel('Count')
    plt.show()

def analyze_environment_in_entry(entry: CrystalEntry, type_A, type_B, **kwargs):
    cell, frac, at_numbers = pmg_structure_to_torch_cell_frac_atomnumbers(entry.structure)
    return compute_species_pair(cell, frac, types=at_numbers, type_A=type_A, type_B=type_B, **kwargs)