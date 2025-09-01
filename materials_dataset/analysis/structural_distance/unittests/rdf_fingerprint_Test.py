import unittest
from pathlib import Path
import numpy as np
# import matplotlib
#
# import matplotlib.pyplot as plt
from ..rdf_fingerprint import RadialFingerprint, compute_fingerprint
from ....io.structures_dataset_io import  StructureDatasetIO
from ....io.uspex_bridge import USPEXBridge
from ....converters import cell_pos_atomtypes_from_pmg_structure

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
#         self.ub_mgalo = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True, new_fp=True)
#         self.ub_boron = USPEXBridge(elements={'B'}, legacy=True, new_fp=True)
#
#
#     def test_rdf_fingerprint_mgalo(self):
#         cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(self.mgalo1.structure)
#         cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(self.mgalo2.structure)
#         fp1, _ = compute_fingerprint(cell1, pos1, types1, [True] * 3)
#         fp2, _ = compute_fingerprint(cell2, pos2, types2, [True] * 3)
#         dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
#         dist_uspex = self.ub_mgalo.fp_dist(self.mgalo1, self.mgalo2)
#         fp1u = self.ub_mgalo.uspex_entry_from_de(self.mgalo1)["radialDistributionUtility.structureFingerprint.origin"]
#         print(f"|FPold-FPnew| = {np.linalg.norm(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy())}")
#         self.assertAlmostEqual(dist_torch.item(), dist_uspex, places=5)
#
#
#     def test_rdf_fingerprint_boron(self):
#         cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(self.boron1.structure)
#         cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(self.boron2.structure)
#         fp1, _ = compute_fingerprint(cell1, pos1, types1, [True] * 3)
#         fp2, _ = compute_fingerprint(cell2, pos2, types2, [True] * 3)
#         dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
#         dist_uspex = self.ub_boron.fp_dist(self.boron1, self.boron2)
#         fp1u = self.ub_boron.uspex_entry_from_de(self.boron1)["radialDistributionUtility.structureFingerprint.origin"]
#         print(f"|FPold-FPnew| = {np.linalg.norm(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy())}")
#         self.assertAlmostEqual(dist_torch.item(), dist_uspex, places=5)


# PLOTTING
if __name__ == '__main__':

    from materials_dataset.analysis.structural_distance.rdf_fingerprint import RadialFingerprint, compute_fingerprint
    from materials_dataset.io.structures_dataset_io import  StructureDatasetIO
    from materials_dataset.io.uspex_bridge import USPEXBridge
    from materials_dataset.converters import cell_pos_atomtypes_from_pmg_structure
    matplotlib.use('TkAgg')
    borons = TWO_BORONS
    boron1, boron2 = StructureDatasetIO(borons).load_from_directory()
    # mgalo = TWO_MgAlO
    # mgalo1, mgalo2 = StructureDatasetIO(mgalo).load_from_directory()
    # ub_mgalo = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True, new_fp=True)
    # ub_mgalo2 = USPEXBridge(elements={'Mg', 'Al', 'O'}, legacy=True, new_fp=True)
    ub_boron = USPEXBridge(elements={'B'}, legacy=True, new_fp=True)
    # ub_boron2 = USPEXBridge(elements={'B'}, legacy=True, new_fp=True)

    # cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(mgalo1.structure)
    # cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(mgalo2.structure)
    # fp1 = compute_fingerprint(cell1, pos1, types1, [True] * 3)
    # fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
    # dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
    # dist_uspex = ub_mgalo.fp_dist(mgalo1, mgalo2)
    # fp1u = ub_mgalo.uspex_entry_from_de(mgalo1)["radialDistributionUtility.structureFingerprint.origin"]
    # print(np.linalg.norm(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy()))
    # plt.plot()
    # plt.plot(fp1u[('Mg', 'O')] - fp1.values[('Mg', 'O')].cpu().numpy(), 'r')
    # plt.figure()
    # plt.plot(fp1u[('Mg', 'O')], fp1.values[('Mg', 'O')].cpu().numpy())
    # plt.show()

    cell1, pos1, types1 = cell_pos_atomtypes_from_pmg_structure(boron1.structure)
    print(f"file: {boron1.metadata["file"]}")
    # cell2, pos2, types2 = cell_pos_atomtypes_from_pmg_structure(boron2.structure)
    fp1, r = compute_fingerprint(cell1, pos1, types1, [True] * 3)
    r = r.cpu().numpy()
    # fp2 = compute_fingerprint(cell2, pos2, types2, [True] * 3)
    # dist_torch = RadialFingerprint.cosine_distance(fp1, fp2)
    # dist_uspex = ub.fp_dist(mgalo1, mgalo2)
    fp1u = ub_boron.uspex_entry_from_de(boron1)["radialDistributionUtility.structureFingerprint.origin"]
    print(np.linalg.norm(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy()))
    plt.figure()
    plt.plot(r, fp1.values[('B', 'B')].cpu().numpy(), 'r')
    plt.title("B-B ours")
    plt.figure()
    plt.plot(r, fp1u[('B', 'B')], 'r')
    plt.title("B-B uspex")
    plt.figure()
    plt.plot(fp1u[('B', 'B')] / abs(fp1u[('B', 'B')][0]) -
             fp1.values[('B', 'B')].cpu().numpy() / abs(fp1.values[('B', 'B')].cpu().numpy()[0]), 'g')
    plt.title("normalized B-B difference")
    plt.figure()
    plt.plot(fp1u[('B', 'B')], fp1.values[('B', 'B')].cpu().numpy(), 'k.')
    plt.title("B-B correl")
    plt.figure()
    plt.plot(r, fp1u[('B', 'B')]  - fp1.values[('B', 'B')].cpu().numpy(), 'go-')
    plt.title("B-B difference")

    import numpy as np
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure


    def bond_length_distribution_continuous(structure: Structure, Rmax: float, sigma: float = 0.05, dr: float = 0.01):
        """
        Вычисляет f(r) = sum_{i<j} exp(-(r - r_ij)^2 / 2sigma^2) / sqrt(2pi sigma^2)
        без биннинга, непрерывная функция.

        Parameters:
            structure (Structure): структура pymatgen
            Rmax (float): радиус отсечки (Å)
            sigma (float): ширина гауссианы (Å)
            dr (float): шаг по r (Å)

        Returns:
            r (np.ndarray): значения r
            f (np.ndarray): значения f(r)
        """
        # Построим суперъячейку, достаточную для покрытия Rmax
        lengths = structure.lattice.abc
        n_repeat = [int(np.ceil(2 * Rmax / l)) for l in lengths]
        supercell = structure.copy()
        supercell.make_supercell(n_repeat)

        # Считаем все r_ij < Rmax
        coords = supercell.cart_coords
        num_atoms = len(coords)
        distances = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                d = np.linalg.norm(coords[i] - coords[j])
                if d < Rmax:
                    distances.append(d)
        distances = np.array(distances)

        # Сетка по r
        r = np.arange(0, Rmax, dr)
        f_r = np.zeros_like(r)

        # Сумма гауссов
        prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        for d in distances:
            f_r += prefactor * np.exp(-0.5 * ((r - d) / sigma) ** 2)

        return r, f_r


    def plot_bond_distribution(structure: Structure, Rmax: float, sigma: float = 0.05, dr: float = 0.01):
        r, f_r = bond_length_distribution_continuous(structure, Rmax, sigma=sigma, dr=dr)
        plt.figure(figsize=(6, 4))
        plt.plot(r, f_r / r ** 2, lw=2)
        plt.xlabel("r (Å)")
        plt.ylabel("f(r)")
        plt.title("Bond Length Distribution")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()

    y = np.loadtxt(PATH_WITH_TESTS / 'uspexdist')
    # plt.figure()
    # plt.hist(y, bins = 400)
    plot_bond_distribution(boron2.structure, Rmax=10, sigma=0.03)
    plt.plot(r, 10000000*(fp1u[('B', 'B')] - fp1.values[('B', 'B')].cpu().numpy()), 'go-')
    plt.show()

