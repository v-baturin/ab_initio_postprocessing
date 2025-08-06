import warnings
import numpy as np
import os
from pathlib import Path
import pickle as pkl
from typing import Union, Any, Callable
from materials_tools.uspex_toolkit.IO import readAtomicStructuresToPoolEntries, \
    read_Individuals_uspexPY, atomsListToPoolEntries
from genutils.duplicate_analysis import remove_duplicates
from genutils.clustering_chunk import make_dist_matrix, clusterize_dist_matrix, select_best_representatives
from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import Atomistic
from USPEX.DataModel import Entry
from datetime import datetime

tdy = datetime.today().strftime('%Y%m%d')
from hydrures_tools.various_tools import fix_wrap_and_check_tetrahedra_in_res_dict

atomistic = Atomistic()



def remove_duplicate_entries(entries: list, fitness_list=None, threshold=np.inf, data_path: Path | None = None,
                             out_path: Path| None = None,
                             clusters_file=None, dist_matrix_file=None,
                             check_clusters_file=False, check_dist_matrix_file=False,
                             legacy=True, tol_Fp=0.08, enforce_compositions_separation=False,
                             compositions_list=None,
                             **kwargs) -> tuple[Any, Any, Any]:

    if not (check_clusters_file and Path(clusters_file).is_file()) or \
            not (check_dist_matrix_file and Path(dist_matrix_file).is_file()):
        rho = prepare_dist_function(entries, legacy=legacy, **kwargs)
    else:
        rho = lambda x:None

    best_representatives, clusters, best_idc = remove_duplicates(entries, dist_fn=rho,
                                                                 intercluster_mindistance=tol_Fp,
                                                                 fitness_list=fitness_list,
                                                                 threshold=threshold, data_path=data_path,
                                                                 clusters_file=clusters_file,
                                                                 dist_matrix_file=dist_matrix_file,
                                                                 check_clusters_file=check_clusters_file,
                                                                 check_dist_matrix_file=check_dist_matrix_file,
                                                                 do_split_clusters_by_labels=enforce_compositions_separation,
                                                                 labels_list=compositions_list)


    out_path = out_path or Path(os.getcwd()) / "deduplicated_POSCARS"

    atomistic.writeAtomicStructures(Path(out_path), [sb.getFlavour('origin') for sb in best_representatives])
    return best_representatives, clusters, best_idc


def remove_duplicates_from_atoms_list(atoms_list, fitness_list, threshold=np.inf,
                                      data_path: Path|None = None,
                                      out_path: Path| None = None,
                                      clusters_file=None, dist_matrix_file=None,
                                      check_clusters_file=True, check_dist_matrix_file=True,
                                      legacy=True, tolFp=0.08, **kwargs):
    """
    :param atoms_list:
    :param fitness_list:
    :param threshold:
    :param data_path:
    :param clusters_file:
    :param dist_matrix_file:
    :param check_clusters_file:
    :param check_dist_matrix_file:
    :param legacy:
    :param tolFp:
    :return:
    """
    data_path = data_path or Path(os.getcwd())
    elements =  set(sym for atoms in atoms_list for sym in atoms.get_chemical_symbols())
    systems = atomsListToPoolEntries(atoms_list)
    clusters_file = clusters_file or data_path / f"clusters_{tdy}.pkl"
    dist_matrix_file = dist_matrix_file or data_path / f"dist_mat_{tdy}.pkl"
    out_path = out_path or data_path / f"POSCARS_filtered"

    if check_clusters_file and Path(clusters_file).is_file():
        with open(clusters_file, 'rb') as clusters_fid:
            clusters = pkl.load(clusters_fid)
    else:
        if check_dist_matrix_file and Path(dist_matrix_file).is_file():
            with open(dist_matrix_file, 'rb') as dist_mat_fid:
                dist_matrix = pkl.load(dist_mat_fid)
        else:
            rho = prepare_dist_function(systems, legacy=legacy, elements=elements, **kwargs)
            dist_matrix = make_dist_matrix(systems, rho, dist_matrix_file)
        clusters = clusterize_dist_matrix(dist_matrix, clusters_out_file=clusters_file, tolFP=tolFp)
    best_representatives, _ = select_best_representatives(clusters, systems, fitness_list, threshold)
    atomistic.writeAtomicStructures(Path(out_path), [sb.getFlavour('origin') for sb in best_representatives])



def remove_duplicates_from_files(struct_file: Path,
                                 individuals_file: Path | None =None,
                                 out_path=None,
                                 individuals_header='', threshold=np.inf,
                                 clusters_file=None, dist_matrix_file=None,
                                 check_clusters_file=True, check_dist_matrix_file=True,
                                 legacy=True, tolFp=0.08,
                                 **kwargs):
    """
    @param struct_file: .uspex file with structures 
    @param individuals_file: 'Individuals' output file of USPEX 
    @param out_path: where to write filtered structures
    @param individuals_header: header of fitness (Enthalpy (eV) or '' for height above CH (older versions))
    @param threshold: maximum height above minimum fitness for a cluster to be considered
    @param kwargs: 
    @return: None
    """
    systems = readAtomicStructuresToPoolEntries(struct_file)
    if individuals_file is not None:
        fitness_list = np.array(read_Individuals_uspexPY(individuals_file)[individuals_header])
        fitness_list = fitness_list[fitness_list != 'None'].astype(float).flatten()
        systems = systems[:len(fitness_list)]
    else:
        fitness_list = None

    data_path = struct_file.parent
    clusters_file = clusters_file or data_path / f"clusters_{tdy}.pkl"
    dist_matrix_file = dist_matrix_file or data_path / f"dist_mat_{tdy}.pkl"
    out_path = out_path or data_path / f"{struct_file.name.replace('.uspex', '')}_filtered"

    if check_clusters_file and Path(clusters_file).is_file():
        with open(clusters_file, 'rb') as clusters_fid:
            clusters = pkl.load(clusters_fid)
    else:
        if check_dist_matrix_file and Path(dist_matrix_file).is_file():
            with open(dist_matrix_file, 'rb') as dist_mat_fid:
                dist_matrix = pkl.load(dist_mat_fid)
        else:
            rho = prepare_dist_function(systems, legacy=legacy)
            dist_matrix = make_dist_matrix(systems, rho, dist_matrix_file)
        clusters = clusterize_dist_matrix(dist_matrix, clusters_out_file=clusters_file, tolFP=tolFp)
    best_representatives, _ = select_best_representatives(clusters, systems, fitness_list, threshold)
    atomistic.writeAtomicStructures(Path(out_path), [sb.getFlavour('origin') for sb in best_representatives])
        
    #     # clusterizePoolEntries(dist_matrix=None, clusters_out_file=None, tolFP=0.16, **kwargs):
    #         
    #     if dist_matrix_file is not None:
    #         
    #     else:
    #         rho = prepare_dist_function(systems)
    #         dist_matrix = make_dist_matrix(systems, rho)
    #     
    #     clusters = clusterizePoolEntries(systems, rho, dist_matrix=dist_matrix, dist_matrix_out_file=None, clusters_out_file=None, tolFP=0.16, **kwargs)
    #         # elements = get_all_chem_symbols_in_entries(systems)
    #         # rdu = RadialDistributionUtility(symbols=elements, suffix='origin')
    #         # rho = rdu.dist
    #     
    # clu_matrix_file = struct_file.parent / ''
    # all_chem_elements = set()
    # if out_path is None:
    #     out_path = struct_file.parent / 'clearedPOSCAR'
    # remove_duplicates(systems, fitnesses, outfile_root=out_path, **kwargs)

def prepare_dist_function(systems, legacy=True, elements=None, **kwargs):
    if elements is None:
        elements = set()
        for system in systems:
            elements |= set(system['atomistic.structure.origin'].getAtomTypes().astype(str))
    rdu = RadialDistributionUtility(symbols=elements, suffix='origin', legacy=legacy, **kwargs)
    for system in systems:
        system.flavours['origin'].extensions.update({'radialDistributionUtility': (rdu, rdu.propertyExtension.propertyTable)})
        # system.flavours['origin'].extensions.update({'radialDistributionUtility': rdu.propertyExtension()})
    rho = rdu.dist
    return rho




# mode = "create_clusterisation"
# # mode = "analyse_clusterisation"
# utility = RadialDistributionUtility(symbols=['H', 'B', 'Ca'])
#
#
#
# if mode == "create_clusterisation":
#     res_path = Path(__file__).parent / 'resCa'
#     # res_path = Path('/home/vsbat/SYNC/00__WORK/20230414_Borohydrures/fix_indices_in_SEEDS/GroundStatePOSCARS/fictive_results_primitive')
#     rho = utility.dist
#     res_dict = readResFolders_uspexPY(res_path, individuals_kind='goodStructures')
#     res_dict = fix_wrap_and_check_tetrahedra_in_res_dict(res_dict)
#     structures = res_dict['structures']
#     for s in structures:
#         utility.calcFingerprint(s)
#     with open('dictionary.pkl', 'wb') as file:
#         print('dict saved')
#         pickle.dump(res_dict, file)
#
#
# elif mode == "analyse_clusterisation":
#     with open('dictionary.pkl', 'rb') as file:
#         res_dict = pickle.load(file)
#     # with open('clusters.pkl', 'rb') as file:
#     #     clusters = pickle.load(file)
#     # print(len(clusters), " clusters")
#     with open('dist_mat.pkl', 'rb') as file:
#         dist_marix = pickle.load(file)
#     # dist_marix += dist_marix.T
#     adj_mat = dist_marix <= 0.35
#     clusters = get_clusters_from_adj_mat(adj_mat)


