import numpy as np
import os
from pathlib import Path
import pickle as pkl
from typing import Union
from my_packages.ab_initio_postprocessing.uspex_toolkit.IO import readAtomicStructuresToPoolEntries, \
    read_Individuals_uspexPY
from my_packages.genutils.clustering_chunk import clustering_by_dist, make_dist_matrix, get_clusters_from_adj_mat
from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation, Atomistic
from datetime import datetime

tdy = datetime.today().strftime('%Y%m%d')
from my_packages.hydrures_tools.various_tools import fix_wrap_and_check_tetrahedra_in_res_dict

atomistic = Atomistic()


def remove_duplicates_from_files(struct_file: os.PathLike, individuals_file: os.PathLike|None =None, out_path=None,
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
        fitnesses = np.array(read_Individuals_uspexPY(individuals_file)[individuals_header])
        fitnesses = fitnesses[fitnesses != 'None'].astype(float).flatten()
        systems = systems[:len(fitnesses)]
    else:
        fitnesses = None

    data_path = struct_file.parent
    clusters_file = data_path / f"clusters_{tdy}.pkl" if clusters_file is None else clusters_file
    dist_matrix_file = data_path / f"dist_mat_{tdy}.pkl" if dist_matrix_file is None else dist_matrix_file
    out_path = data_path / f"{struct_file.name.replace('.uspex', '')}_filtered" if out_path is None else out_path

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
        clusters = clusterizePoolEntries(dist_matrix, clusters_out_file=clusters_file, tolFP=tolFp)
    best_representatives = select_best_representatives(clusters, systems, fitnesses, threshold)

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

def prepare_dist_function(systems, legacy=False):
    elements = set()
    for system in systems:
        elements |= set(system['atomistic.structure.origin'].getAtomTypes().astype(str))
    rdu = RadialDistributionUtility(symbols=elements, suffix='origin', legacy=legacy)
    for system in systems:
        system.flavours['origin'].extensions.update({'radialDistributionUtility': rdu.propertyExtension()})
    rho = rdu.dist
    return rho


def select_best_representatives(clusters, systems, fitnesses = None, threshold = 0.1, **kwargs):
    print(f"Processing {len(clusters)} clusters")
    if fitnesses is not None:
        ref_fitness = np.min(fitnesses)
    else:
        fitnesses = np.zeros(len(systems), dtype=float)
        ref_fitness = 0.
    good_fitnesses = []
    i = 0
    best_of_each = []
    for cluster in clusters:
        cl_en = fitnesses[cluster]
        min_en = np.min(cl_en)
        idx_min = cluster[np.where(cl_en == np.min(cl_en))[0][0]]
        struct = systems[idx_min]
        if min_en is not None and ref_fitness is not None and (min_en - ref_fitness > threshold):
            continue
        struct.setProperty("ID", i)
        good_fitnesses.append(min_en)
        best_of_each.append(struct)
        i += 1
    print(f"{len(best_of_each)} good ones. Writing")

    sorting_idx = np.argsort(good_fitnesses)
    sorted_best = []
    for i in sorting_idx:
        sorted_best.append(best_of_each[i])
    return sorted_best


def clusterizePoolEntries(dist_matrix=None, clusters_out_file=None, tolFP=0.16, **kwargs):
    adj_mat = dist_matrix <= tolFP
    clusters = get_clusters_from_adj_mat(adj_mat)
    if clusters_out_file is None:
        clusters_out_file = f'{os.getcwd()}/clusters_{tdy}.pkl'
    with open(clusters_out_file, 'wb') as cl_file:
        pkl.dump(clusters, cl_file)
        print(f'clustering saved in {clusters_out_file}')
    return clusters

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
