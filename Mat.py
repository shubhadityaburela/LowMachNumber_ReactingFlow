import numpy as np
from scipy import sparse

"""
This class creates coefficient matrices based on Finite Difference approach with given:
'derivative',
'size of the matrix (N)', 
'grid spacing (dx)'
'Periodicity of the grid' 
'Explicit friction argument'
"""


class Mat:
    def __init__(self) -> None:
        pass

    @staticmethod
    def D_periodic(Coeffs, N, h):
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D = sparse.csc_matrix(np.zeros((N, N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D = D + Coeffs[k - diagonalLow] * sparse.csc_matrix(np.diag(np.ones(N - abs(k)), k))
            if k < 0:
                D = D + Coeffs[k - diagonalLow] * sparse.csc_matrix(
                    np.diag(np.ones(abs(k)), N + k))
            if k > 0:
                D = D + Coeffs[k - diagonalLow] * sparse.csc_matrix(
                    np.diag(np.ones(abs(k)), -N + k))

        return D / h

    # In non-periodic cases we need to adjust for the boundary nodes. Therefore we apply BlockUL and BlockBR matrices
    # for that purpose
    @staticmethod
    def D_nonperiodic(Coeffs, N, h, BlockUL, BlockBR):
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D = sparse.csc_matrix(np.zeros((N, N), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D = D + Coeffs[k - diagonalLow] * sparse.csc_matrix(np.diag(np.ones(N - abs(k)), k))

        a = BlockUL.shape[0]
        b = BlockUL.shape[1]
        D[0:a, 0:b] = BlockUL

        a = BlockBR.shape[0]
        b = BlockBR.shape[1]
        D[D.shape[0] - a:D.shape[0], D.shape[1] - b:D.shape[1]] = BlockBR

        return D / h
