import unittest
from pathlib import Path
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
# from ..rdf_fingerprint import RadialFingerprint, compute_fingerprint
# from ....io.structures_dataset_io import  StructureDatasetIO
# from ....io.uspex_bridge import USPEXBridge
# from ....converters import cell_pos_atomtypes_from_pmg_structure

PATH_WITH_TESTS = Path(__file__).parent
PATH_WITH_DATASETS = PATH_WITH_TESTS / "../../../unittests_datasets"
TWO_BORONS = PATH_WITH_DATASETS / "two_borons"
TWO_MgAlO = PATH_WITH_DATASETS /  "two_systems_MgAlO"

# class USPEXBridge_Test(unittest.TestCase):
#
#     def setUp(self):
#
#         self.borons = TWO_BORONS
#         self.boron1, self.boron2 = StructureDatasetIO(self.borons).load_from_directory()
#         self.mgalo = TWO_MgAlO
#         self.mgalo1, self.mgalo2 = StructureDatasetIO(self.mgalo).load_from_directory()
#         self.ub = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True)
#
#
#     def test_rdf_fingerprint_mgalo(self):
#         cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(self.mgalo1.structure)
#         cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(self.mgalo2.structure)
#         fp1 = compute_fingerprint(cell1, pos1, types1, [True] * 3)
#         fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
#         dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
#         dist_uspex = self.ub.fp_dist(self.mgalo1, self.mgalo2)
#         fp1u = self.ub.uspex_entry_from_de(self.mgalo1)["radialDistributionUtility.structureFingerprint.origin"]
#         print(np.linalg.norm(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy()))
#         plt.plot(fp1u[('Mg', 'O')])
#         plt.figure()
#         plt.plot(fp1.values[('Mg', 'O')].cpu().numpy())
#         plt.figure()
#         plt.plot(fp1u[('Mg', 'O')], fp1.values[('Mg', 'O')].cpu().numpy())
#         plt.show()
#         # self.assertAlmostEqual(dist_torch.item(), dist_uspex, places=5)
#
#
#     def test_rdf_fingerprint_boron(self):
#         cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(self.boron1.structure)
#         cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(self.boron2.structure)
#         fp1 = compute_fingerprint(cell1, pos1, types1, [True] * 3)
#         fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
#         dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
#         dist_uspex = self.ub.fp_dist(self.boron1, self.boron2)
#         fp1u = self.ub.uspex_entry_from_de(self.boron1)["radialDistributionUtility.structureFingerprint.origin"]
#         print(np.linalg.norm(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy()))
#         self.assertAlmostEqual(dist_torch.item(), dist_uspex, places=5)


#PLOTTING
if __name__ == '__main__':

    from materials_dataset.analysis.structural_distance.rdf_fingerprint import RadialFingerprint, compute_fingerprint
    from materials_dataset.io.structures_dataset_io import  StructureDatasetIO
    from materials_dataset.io.uspex_bridge import USPEXBridge
    from materials_dataset.converters import cell_pos_atomtypes_from_pmg_structure
    matplotlib.use('TkAgg')
    borons = TWO_BORONS
    boron1, boron2 = StructureDatasetIO(borons).load_from_directory()
    mgalo = TWO_MgAlO
    mgalo1, mgalo2 = StructureDatasetIO(mgalo).load_from_directory()
    ub = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True)

    cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(mgalo1.structure)
    cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(mgalo2.structure)
    fp1 = compute_fingerprint(cell1, pos1, types1, [True] * 3)
    fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
    dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
    dist_uspex = ub.fp_dist(mgalo1, mgalo2)
    fp1u = ub.uspex_entry_from_de(mgalo1)["radialDistributionUtility.structureFingerprint.origin"]
    print(np.linalg.norm(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy()))
    # plt.plot()
    # plt.plot(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy(), 'r')
    # plt.figure()
    # plt.plot(fp1u[('Mg', 'O')], fp1.values[('Mg', 'O')].cpu().numpy())
    # plt.show()

    cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(boron1.structure)
    cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(boron2.structure)
    fp1 = compute_fingerprint(cell1, pos1, types1, [True] * 3)
    fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
    # dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
    # dist_uspex = ub.fp_dist(mgalo1, mgalo2)
    fp1u = ub.uspex_entry_from_de(boron1)["radialDistributionUtility.structureFingerprint.origin"]
    print(np.linalg.norm(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy()))
    plt.figure()
    plt.plot(fp1.values[('B', 'B')].cpu().numpy(), 'r')
    plt.title("B-B ours")
    plt.figure()
    plt.plot(fp1u[('B', 'B')], 'r')
    plt.title("B-B uspex")
    plt.figure()
    plt.plot(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy(), 'g')
    plt.title("B-B difference")
    plt.figure()
    plt.plot(fp1u[('B', 'B')], fp1.values[('B', 'B')].cpu().numpy(), 'k.')
    plt.title("B-B correl")
    plt.show()