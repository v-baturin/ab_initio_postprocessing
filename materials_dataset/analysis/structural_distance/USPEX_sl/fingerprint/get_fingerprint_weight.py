import numpy as np


def get_fingerprint_weight(numIons):
    """
    Replicates the Matlab logic:

        L = length(numIons);
        S = 0;
        weight = zeros(L*L,1);
        for i = 1:L
            for j = 1:L
                ind = (i-1)*L + j;
                weight(ind) = numIons(i)*numIons(j);
                S = S + numIons(i)*numIons(j);
            end
        end
        weight = weight / S;

    Parameters
    ----------
    numIons : 1D array-like of length L
        Number of atoms (or molecules) for each species.

    Returns
    -------
    weight : 1D numpy array of length L*L
        Normalized so that the sum of all elements is 1.
    """
    # Convert numIons to a numpy array (float or int)
    numIons = np.asarray(numIons, dtype=float)
    L = len(numIons)

    # Initialize weight array
    weight = np.zeros(L * L, dtype=float)

    # Fill and sum
    total = 0.0
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            weight[idx] = numIons[i] * numIons[j]
            total += weight[idx]

    # Normalize
    weight /= total

    return weight

#w = get_fingerprint_weight([1,1])
#print(w)
