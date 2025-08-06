import numpy as np

def cosineDistance(matrA, matrB, weight):
    """
    Python equivalent of the Matlab function cosineDistance.

    Parameters
    ----------
    matrA : array-like of shape (r, c)
        First matrix.
    matrB : array-like of shape (r, c)
        Second matrix, same shape as matrA.
    weight : array-like
        Either a 1D array of length r (weights for each row),
        or a 2D array of shape (r, r). If 1D, we do elementwise
        multiplication with broadcasting. If 2D, we use matrix
        multiplication.

    Returns
    -------
    dist : float
        A "cosine distance" measure, weighted by `weight`,
        then scaled by 1/2. If the raw value is extremely close
        to 1, it is clamped to 0.99999.
    """

    A = np.asarray(matrA, dtype=float)
    B = np.asarray(matrB, dtype=float)
    w = np.asarray(weight, dtype=float)

    # Handle the 1D weight case via broadcasting instead of
    # constructing a diagonal matrix
    if w.ndim == 1:
        # shape of A and B: (r, c)
        # shape of w: (r,)
        # shape of w[:, None]: (r, 1), which broadcasts over columns
        num = np.sum((w[:, None] * A) * B)
        den = np.sqrt(np.sum(w[:, None] * (A * A)) *
                      np.sum(w[:, None] * (B * B)))
    else:
        # 2D weight: use matrix multiplication
        AB = A * B
        A2 = A * A
        B2 = B * B
        num = np.sum(w @ AB)
        den = np.sqrt(np.sum(w @ A2) * np.sum(w @ B2))

    # Avoid division by zero in edge cases (e.g. all-zero rows)
    if den == 0.0:
        dist = 0.0
    else:
        dist = (1.0 - num / den) / 2.0

    # If the value is extremely close to 1, clamp it slightly below 1
    if dist > 1.0 - 1e-6:
        dist = 0.99999
        print("Fingerprint might be wrong !!!")

    return dist
