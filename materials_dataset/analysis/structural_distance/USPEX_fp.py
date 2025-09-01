"""
USPEX.Atomistic.RadialDistributionUtility
=========================================
    Fingerprint calculation function.
    Reference: A.R. Oganov, M. Valle. How to quantify energy landscapes. J. Chem. Phys, 104504, 2009.
"""

import logging
import numpy as np
from typing import Dict, Tuple
from collections.abc import Mapping

from scipy.special import erf
from scipy.spatial.distance import cdist
from itertools import combinations

from USPEX.Expressions import PropertyExtension
from USPEX.DataModel.Flavour import Flavour

logger = logging.getLogger(__name__)


RMAX_DEFAULT = 10.0
SIGMA_DEFAULT = 0.03
DELTA_DEFAULT = 0.08

TOLERANCE_DEFAULT = 0.012


class Fingerprint(Mapping):
    """
    Class representing radial distribution fingerprint.
    """
    def __init__(self, value: dict, weights, delta):
        sizes = [len(v) for v in value.values()]
        assert len(sizes) > 0
        self._value = value
        self._weights = weights
        self.delta = delta
        self._size = sizes[0]
        super().__init__()

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return self._size

    @property
    def weights(self):
        return self._weights

    @property
    def order(self):
        # eq. 5 in CPC-2010
        return np.sqrt(np.sum(np.fromiter((self._weights[key] * np.sum(self._value[key] ** 2)
                                           for key in self._value.keys()), dtype=float)) * self.delta)

    def __repr__(self):
        return self._value.__repr__()

    def __len__(self):
        return self._value.__len__()

    def __iter__(self):
        return self._value.__iter__()

    def __getitem__(self, item):
        if item in self._value:
            return self._value.__getitem__(item)
        else:
            return -np.ones(self._size)

    @staticmethod
    def cosine_distance(fingerprint1, fingerprint2):
        """
        Calculation of cosine distances using eq.(6b) from JCP-2009.
        """
        value1 = fingerprint1.value
        weights1 = fingerprint1.weights
        value2 = fingerprint2.value
        weights2 = fingerprint2.weights
        coef1 = 0
        coef2 = 0
        coef3 = 0
        for key in set(value1.keys()) | set(value2.keys()):
            fing1 = value1[key] if key in value1 else np.zeros((1), dtype=float)
            fing2 = value2[key] if key in value2 else np.zeros((1), dtype=float)
            weight1 = weights1[key] if key in weights1 else 0
            weight2 = weights2[key] if key in weights2 else 0
            coef1 += np.sqrt(weight1 * weight2) * np.sum(fing1 * fing2)
            coef2 += weight1 * np.sum(fing1 * fing1)
            coef3 += weight2 * np.sum(fing2 * fing2)
        dist = (1 - coef1 / (coef2 * coef3) ** 0.5) / 2
        return dist


    @staticmethod
    def cosineDistance(fing1, fing2):
        """
        Calculation of cosine distances using eq.(6b) from JCP-2009.
        """
        norm1 = np.linalg.norm(fing1, axis=1).reshape((-1, 1))
        norm2 = np.linalg.norm(fing2, axis=1).reshape((1, -1))
        return (1 - np.dot(fing1, fing2.T) / (norm1 * norm2)) / 2


# noinspection PyPep8Naming
class RadialDistributionUtility(object):
    """
    Utility for working with radial distribution related properties of systems.
    """

    propertyExtension = PropertyExtension()

    def __init__(self, symbols, suffix, Rmax=RMAX_DEFAULT, sigma=SIGMA_DEFAULT, delta=DELTA_DEFAULT,
                 tolerance=TOLERANCE_DEFAULT, storeDistances=True,
                 legacy=False, **kwargs):
        """
        :type Rmax: float
        :param Rmax: threshold distance between i-th anf j-th atom.
        :type sigma: float
        :param sigma: smearing parameter.
        :type delta: float
        :param delta: bin width.
        :type tolerance: float
        :param tolerance: tolerance within which systems considered the same.
        """
        self.symbols=symbols
        self.suffix = suffix
        self.Rmax = Rmax
        self.sigma = sigma
        self.delta = delta
        self.tolerance = tolerance
        self.legacy = legacy
        self.distances = {}
        self.storeDistances = storeDistances
        self.legacy_distances = {}

    def clean(self, system):
        """
        Removes all stored **RadialDistributionUtility** related propertiies from *system* dictionary.

        :param system: dictionary describing system.

        """
        systemFlavour = system.getFlavour(self.suffix)
        systemFlavour.delProperty('structureFingerprint', extension='radialDistributionUtility')
        systemFlavour.delProperty('complexFingerprint', extension='radialDistributionUtility')

    def calcFingerprint(self, system):
        """
        Calculates fingerprint and related things.
        """
        structure = system['atomistic.structure']
        disassembler = system['atomistic.disassembler']
        atomTypes = structure.getAtomTypes()
        uniqueSymbols, inverse, numIons = np.unique(atomTypes, return_inverse=True, return_counts=True)
        indices = np.argsort(inverse)
        revertIndices = np.argsort(indices)
        cartesian = structure.getCartesianCoordinates()
        cell = structure.getCell().getEnvelopeCell(cartesian, 1)
        cartesian = cell.center(cartesian)
        coordinates = cell.cartesianToFractional(cartesian)[indices]
        molIndices = [revertIndices[inds] for inds in disassembler.indices]
        envIndices = revertIndices[disassembler.envIndices]
        fp_pbc = structure.getCell().getPBC()
        lat = cell.getCellVectors()
        dist_matrix = _make_matrices(coordinates, molIndices, envIndices, lat, numIons, pbc=fp_pbc, Rmax=self.Rmax)

        N_type = numIons.shape[0]
        N_pair = dist_matrix.shape[0]  # the number of atomic pairs being considered
        #normalizer = numIons / np.linalg.det(lat) if sum(fp_pbc) == 3 else np.zeros((1,), dtype=float)
        normalizer = 1 / np.linalg.det(lat) if sum(fp_pbc) == 3 else np.zeros((1,), dtype=float)

        N_Bins = int(round(self.Rmax / float(self.delta)))
        fing = np.zeros((N_type, N_type, N_Bins))
        # sigma = self.sigma / (2.0 * np.log(2.0)) ** 0.5
        sigma = self.sigma
        sqrt2_sigm = sigma * 2.0 ** 0.5

        # Vectorization:
        N_bins = int(np.ceil(8 * sigma / self.delta))  # how many bins each dist can contribute

        # The following variables are vectors:
        interval = np.zeros((N_pair, 2))
        atom1 = dist_matrix[:, 0]
        type1 = dist_matrix[:, 1]
        type2 = dist_matrix[:, 2]
        btype1 = type1 * N_type + type2  #bond type
        R0 = dist_matrix[:, 3]
        R02 = R0 ** 2.0
        min_bin = np.floor((-4.0 * sigma + R0) / float(self.delta) + 0.5) + 1.0

        for i in range(N_bins):
            # Obtain the interval vectors:
            interval[:, 0] = 5.0 * np.sign(self.delta * (min_bin - 1.0) - R0)
            ID = np.where(abs(self.delta * (min_bin - 1.0) - R0) <= 5.0 * sqrt2_sigm)
            interval[ID[0], 0] = (self.delta * (min_bin[ID] - 1.0) - R0[ID]) / sqrt2_sigm
            interval[:, 1] = 5.0 * np.sign(self.delta * min_bin - R0)
            ID = np.where(abs(self.delta * min_bin - R0) <= 5.0 * sqrt2_sigm)
            interval[ID[0], 1] = (self.delta * min_bin[ID] - R0[ID]) / sqrt2_sigm
            my_erf = erf(interval)  # directly use erf function
            delt = 0.5 * (my_erf[:, 1] - my_erf[:, 0]) / R02  # used for fing


            # fing has 2 dimensions: we need to categorize by the following:
            for j in range(N_type * N_type):
                ID1 = np.where(btype1 == j)  # 1st filter by bond type
                delt1 = delt[ID1]
                for k in range(N_Bins):
                    ID = np.where(min_bin[ID1] - 1 == k)[0]  # 2nd filter by dist (bin)
                    fing[j // N_type, j % N_type, k] += sum(delt1[ID])
            min_bin += 1  # move to the next neighboring bin
        fing /= (4.0 * np.pi * numIons.reshape((-1, 1, 1)) * numIons.reshape((1, -1, 1)) * self.delta)
        fing -= normalizer.reshape((1, -1, 1))
        fing = {(s1.short_name, s2.short_name): fing[i, j] for i, s1 in enumerate(uniqueSymbols) for j, s2 in enumerate(uniqueSymbols)}
        fingerprint = Fingerprint(value=fing, weights=self._fingerprintWeights(structure),
                                  delta=self.delta)
        system.setProperty('structureFingerprint', fingerprint, extension='radialDistributionUtility')

    def dist(self, system1, system2, legacy=None):
        """
        Calculated distance between two systems. First it retrieves structure fingerprints of systems.
        Then calculates cosine distance between them.

        :param system1: dictionary describing first system.
        :param system2: dictionary describing second system.

        :return: distance between systems.
        """
        legacy = self.legacy if legacy is None else legacy

        expr = f'radialDistributionUtility.structureFingerprint.{self.suffix}'
        dist_fun = Fingerprint.cosine_distance
        distances_cache = self.legacy_distances
        if self.storeDistances:
            pair = (system1.ID, system2.ID) if system1.ID < system2.ID else (system2.ID, system1.ID)
            if pair not in distances_cache:
                distances_cache[pair] = dist_fun(system1[expr], system2[expr])
            distance = distances_cache[pair]
        else:
            distance = dist_fun(system1[expr], system2[expr])
        return distance

    def equal(self, system1, system2, tolerance=None):
        """
        Checks if systems coincide. It calculates distance between systems using **dist** method.
        If such distance is less then set up tolerance, then systems coincide.

        :param system1: dictionary describing first system.
        :param system2: dictionary describing second system.
        :param tolerance: threshold for systems to be considered equivalent.

        :return: if systems coincide or not.
        """
        tolerance = self.tolerance if tolerance is None else tolerance
        return self.dist(system1, system2) < tolerance

    @staticmethod
    def _fingerprintWeights(structure):
        """
        :rtype: Dict[Tuple[str,str], float]
        :return: weights of fingerprints of each atom type pair to be used in cosine distance calculation.
        """
        comp = structure.getComposition()
        # TODO Whether we really need to duplicate weights Like Fe-C and C-Fe
        weights = {(s1.short_name, s2.short_name): am1*am2 for s1, am1 in comp.items() for s2, am2 in comp.items()}
        weightSum = np.sum(list(weights.values()))
        for key in weights:
            weights[key] /= weightSum
        return weights

    @propertyExtension
    def structureFingerprint(self, system: Flavour):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculate or retrieve structure fingerprint of a system.
        """
        self.calcFingerprint(system)
        return system['radialDistributionUtility.structureFingerprint']



def _super_matrix(xmin: int, xmax: int, ymin: int, ymax: int, zmin: int, zmax: int):
    """
    This is a small utility to quickly generate a 3d matrix series of the type [x1 y1 z1; x2 y2 z2; ......]
    with x, y and z in the input ranges [x_min, x_max], [y_min, y_max] and [z_min, z_max].
    The functional is usually used to create supercells.
    Example:

        INPUT:  [0 1 0 1 0 1]
        OUTPUT: [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]

    :type xmin: int
    :param xmin: min x value.
    :type xmax: int
    :param xmax: max x value.
    :type ymin: int
    :param ymin: min y value.
    :type ymax: int
    :param ymax: max y value.
    :type zmin: int
    :param zmin: min z value.
    :type zmax: int
    :param zmax: max z value.

    :rtype: list
    :return matrix: resulted matrix.
    """

    # matrix = []
    # for i in range(xmin, xmax + 1):
    #     for j in range(ymin, ymax + 1):
    #         for k in range(zmin, zmax + 1):
    #             matrix.append([i, j, k])
    # return matrix
    return [[i, j, k] for i in range(xmin, xmax + 1) for j in range(ymin, ymax + 1) for k in range(zmin, zmax + 1)]


def _make_matrices(coor: np.ndarray, molIndices: list, envIndices,
                   lat: np.ndarray, numIons: np.ndarray, pbc=(1, 1, 1), Rmax=10.0):
    """
    The function prepares matrices for fingerprint calculation.

    :type coor: numpy array
    :param coor: fractional coordinates of atoms of system for which we want to calculate the distance matrix,
        groupped according atom types.
    :type molIndices: list of numpy arrays
    :param molIndices: specification of the molecules. A list of numpy arrays, i-th array contains indices of
        atoms that make up i-th molecule.
    :type envIndices: numpy array
    :param envIndices: specification of the environment, array of atom indices
    :type lat: numpy array
    :param lat: cell of system for which we want to calculate the distance matrix.
    :type numIons: numpy array
    :param numIons: number atoms of each type in system for which we want to calculate the distance matrix.
    :type Rmax: float
    :param Rmax: distance cutoff.

    :rtype: numpy array
    :return: distance matrix of the form [atom_i, atomi_type, atomj_type, dist]
    """

    assert isinstance(Rmax, float) and Rmax >= 0

    # coor = coor - np.floor(coor)  # scale it the [0 1]
    N_atom = np.sum(numIons)

    types = []
    for i in range(len(numIons)):
        types += [i] * numIons[i]
    types = np.asarray(types)

    dist_matrix = np.zeros((0, 4), dtype=float)

    # Leftmost, rightmost, uppermost, lowermost, "frontmost" and "backmost" points of
    # the origin-centered Rmax-sphere IN THE FRACTIONAL COORDINATES, (where this sphere is actually an ellipsoid),
    sphere_fract_margins = np.zeros((6, 3))
    inv_lat = np.linalg.inv(lat)
    for i in range(3):
        sphere_fract_margins[i] = np.cross(lat[i-2], lat[i-1])
        sphere_fract_margins[i] *= Rmax / np.linalg.norm(sphere_fract_margins[i])
    sphere_fract_margins = sphere_fract_margins @ inv_lat
    sphere_fract_margins[3:, :] = -sphere_fract_margins[:3, :]

    for i in range(N_atom):
        # target = np.zeros(matrix.shape)
        # Build the super cell containing Rmax sphere around i-th atom:
        shifted_sphere_margins = coor[i, :] + sphere_fract_margins
        mins = np.floor(np.min(shifted_sphere_margins, axis=0)) * pbc
        maxs = np.floor(np.max(shifted_sphere_margins, axis=0)) * pbc
        matrix_tmp = np.asarray(_super_matrix(*[int(x) for x in np.dstack((mins, maxs)).flatten()]))

        # Obtain the distances by vectorization, cdist is used here:
        N_sublattices = matrix_tmp.shape[0]
        S_coor = np.tile(coor, (N_sublattices, 1))
        S_matrix = np.zeros((N_atom * N_sublattices, 3))
        for i_sublat in range(N_sublattices):
            S_matrix[(i_sublat * N_atom):((i_sublat + 1) * N_atom), :] = matrix_tmp[i_sublat]
        tmp_dist = cdist(np.reshape(np.dot(coor[i], lat), (1, 3)), np.dot(S_coor + S_matrix, lat))
        tmp_type = np.tile(types, N_sublattices)

        # Remove distances within a molecule containing i-th atom and
        # distances between the atoms of environment
        if i in envIndices:
            to_delete = (envIndices.reshape((-1, 1)) + N_atom * np.arange(N_sublattices)).flatten()
        else:
            ignoreDist = set()
            central_cell_shift = np.flatnonzero(np.all(matrix_tmp == 0, axis=1)) * N_atom
            for inds in molIndices:
                if i in inds:
                    ignoreDist.update(inds + central_cell_shift)
            to_delete = np.asarray(list(ignoreDist), dtype=int)

        tmp_dist = np.delete(tmp_dist, to_delete, axis=1)
        tmp_type = np.delete(tmp_type, to_delete, axis=0)

        # Remove distances exceeding Rmax and those smaller than 0.5 \AA
        to_delete = np.where((tmp_dist > Rmax) | (tmp_dist < 0.5))
        tmp_dist = np.delete(tmp_dist, to_delete[1], axis=1)
        tmp_type = np.delete(tmp_type, to_delete[1], axis=0)

        if tmp_dist.shape[1] > 0 and i not in envIndices:  # sometimes you can meet an isolated atom
            tmp2 = np.zeros((tmp_dist.shape[1], 4))
            tmp2[:, 0] = i
            tmp2[:, 1] = types[i]
            tmp2[:, 2] = tmp_type
            tmp2[:, 3] = tmp_dist
            dist_matrix = np.vstack((dist_matrix, tmp2))

    return dist_matrix
