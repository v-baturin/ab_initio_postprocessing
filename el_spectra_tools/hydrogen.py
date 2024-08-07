import numpy as np
import scipy.special as spe


def HFunc(r, theta, phi, n, l, m):
    """
    Hydrogen wavefunction // a_0 = 1

    INPUT
        r: Radial coordinate
        theta: Polar coordinate
        phi: Azimuthal coordinate
        n: Principle quantum number
        l: Angular momentum quantum number
        m: Magnetic quantum number

    OUTPUT
        Value of wavefunction
    """

    coeff = np.sqrt((2.0 / n) ** 3 * spe.factorial(n - l - 1) / (2.0 * n * spe.factorial(n + l)))
    laguerre = spe.assoc_laguerre(2.0 * r / n, n - l - 1, 2 * l + 1)
    sphHarm = spe.sph_harm(m, l, phi, theta)  # Note the different convention from doc

    return coeff * np.exp(-r / n) * (2.0 * r / n) ** l * laguerre * sphHarm


def H_real_p(r, r0, orb='px'):
    dr = r - r0
    x = dr[0]
    y = dr[1]
    z = dr[2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan(y / x)
    if orb == 0 or orb == 'px':
        ans = (HFunc(r, theta, phi, 2, 1, 1) + HFunc(r, theta, phi, 2, 1, -1)) / np.sqrt(2)
    elif orb == 1 or orb == 'py':
        ans = (HFunc(r, theta, phi, 2, 1, 1) - HFunc(r, theta, phi, 2, 1, -1)) / (1j * np.sqrt(2))
    elif orb == 2 or orb == 'pz':
        ans = HFunc(r, theta, phi, 2, 1, 0)
    else:
        raise RuntimeError("Invalid value of xyz_arg")
    return ans
