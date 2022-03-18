
from scipy.sparse import linalg


def directLU_Init(A):
    lu = linalg.splu(A)

    return lu

