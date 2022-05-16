import numpy as np


def CalcReactionRates(prim_var, refVar):
    # First re-dimensionalize the variables
    rho = prim_var[:, :, 0] * refVar['rho_ref']
    T = prim_var[:, :, 4] * refVar['T_ref']
    Y = (prim_var[:, :, 5] / prim_var[:, :, 0]) * refVar['mass_frac_ref']

    # There is currently 1 species CH4. This is a one-step Arrhenius reaction mechanism
    # omegaDot = A * rho * Y * exp (-Ea / RT)
    A = 0  # 1.64e10
    Ea = 67.55 * refVar['R'] * refVar['T_ref']

    # Reaction rate
    omegaDot = A * rho * Y * np.exp(- Ea / (refVar['R'] * T))

    # Non-dimensionalize the final reaction rate
    omegaDot = omegaDot / refVar['omegaDot_ref']  # omegaDot

    prim_var[:, :, 6] = omegaDot





