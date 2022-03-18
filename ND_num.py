import numpy as np


def num_ND(refVar):
    # Non dimensional numbers
    M = refVar['u_ref'] / np.sqrt(refVar['gamma'] * refVar['p_ref'] / refVar['rho_ref'])
    Re = refVar['rho_ref'] * refVar['u_ref'] * refVar['X_ref'] / refVar['mu']
    Fr = refVar['u_ref'] / np.sqrt(refVar['g'] * refVar['X_ref'])
    Pr = refVar['gamma'] * refVar['Cv'] * refVar['mu'] / refVar['K']

    params = {
        'M': M,
        'Re': Re,
        'Fr': Fr,
        'Pr': Pr
    }

    return params
