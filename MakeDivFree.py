import scipy
import numpy as np
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse import linalg


def makeDivergenceFree(params, numbers_ND, coefficient_mats, prim_var):
    rho = prim_var[:, :, 0]
    u_aux = prim_var[:, :, 1]
    v_aux = prim_var[:, :, 2]

    NN = params['Geometry Parameters']['Nxi'] * params['Geometry Parameters']['Neta']
    rho_sub = np.reshape(rho, newshape=NN, order="F")
    Q = (1 / (numbers_ND['Pr'] * numbers_ND['Re'])) * coefficient_mats.LaplaceEnergy.dot(np.reciprocal(rho_sub))
    p_x, p_y, divUOld, p_project = calcGradP(params, coefficient_mats, u_aux, v_aux, rho, Q)

    if not params['Geometry Parameters']['XI_Periodic']:
        p_x[0, :] = 0
        p_x[-1, :] = 0
        p_y[0, :] = 0
        p_y[-1, :] = 0

    if not params['Geometry Parameters']['ETA_Periodic']:
        p_x[:, 0] = 0
        p_x[:, -1] = 0
        p_y[:, 0] = 0
        p_y[:, -1] = 0

    u = u_aux - params['Time Parameters']['dt'] * p_x / rho
    v = v_aux - params['Time Parameters']['dt'] * p_y / rho

    prim_var[:, :, 1] = u
    prim_var[:, :, 2] = v

    divUNew = coefficient_mats.Div_Xi * u + v * coefficient_mats.Div_Eta.transpose()

    return prim_var, divUOld, divUNew, p_project, p_x, p_y


def makeDivergenceFreeRHS(params, numbers_ND, coefficient_mats, prim_var, rhs_u_wop, rhs_v_wop):
    rho = prim_var[:, :, 0]

    NN = params['Geometry Parameters']['Nxi'] * params['Geometry Parameters']['Neta']
    rho_sub = np.reshape(rho, newshape=NN, order="F")
    Q = (1 / (numbers_ND['Pr'] * numbers_ND['Re'])) * coefficient_mats.LaplaceEnergy.dot(np.reciprocal(rho_sub))
    p_x, p_y, divRHSOld, p_project = calcGradPRHS(params, coefficient_mats, rhs_u_wop, rhs_v_wop, rho, Q)

    if not params['Geometry Parameters']['XI_Periodic']:
        p_x[0, :] = 0
        p_x[-1, :] = 0
        p_y[0, :] = 0
        p_y[-1, :] = 0

    if not params['Geometry Parameters']['ETA_Periodic']:
        p_x[:, 0] = 0
        p_x[:, -1] = 0
        p_y[:, 0] = 0
        p_y[:, -1] = 0

    rhs_u = rhs_u_wop - p_x / rho
    rhs_v = rhs_v_wop - p_y / rho

    divRHSNew = coefficient_mats.Div_Xi * rhs_u + rhs_v * coefficient_mats.Div_Eta.transpose()

    return rhs_u, rhs_v, divRHSOld, divRHSNew


def calcGradP(params, coefficient_mats, u_aux, v_aux, rho, Q):
    if params['Geometry Parameters']['TransformedGrid']:
        raise NotImplemented("Transformed grids not implemented")
    else:
        dt = params['Time Parameters']['dt']
        Q = np.reshape(Q, newshape=[params['Geometry Parameters']['Nxi'], params['Geometry Parameters']['Neta']],
                       order="F")
        divU = coefficient_mats.Div_Xi * u_aux + v_aux * coefficient_mats.Div_Eta.transpose()
        Source = (divU - Q) / dt

    NN = params['Geometry Parameters']['Nxi'] * params['Geometry Parameters']['Neta']
    rho_sub = np.reshape(rho, newshape=NN, order="F")
    RHO_inv = diags(np.reciprocal(rho_sub), format="csc")

    Lp = coefficient_mats.Div_Xi_kron * RHO_inv * coefficient_mats.Grad_Xi_kron \
        + coefficient_mats.Div_Eta_kron * RHO_inv * coefficient_mats.Grad_Eta_kron

    LU = linalg.splu(Lp)

    if params['Equation Parameters']['PressureCalcMethod'] == 'backslashLaplace':
        p = scipy.sparse.linalg.inv(Lp) * np.squeeze(np.reshape(Source, newshape=[-1, 1], order="F"))
    elif params['Equation Parameters']['PressureCalcMethod'] == 'backslashLaplaceNeumann':
        raise NotImplemented("backslashLaplaceNeumann not implemented as a Pressure method")
    elif params['Equation Parameters']['PressureCalcMethod'] == 'LULaplace':
        p = LU.solve(np.squeeze(np.reshape(Source, newshape=[-1, 1], order="F")))
    elif params['Equation Parameters']['PressureCalcMethod'] == 'FFT':
        raise NotImplemented("FFT is not implemented as a Pressure method")
    else:
        raise NotImplemented("Please select from the following options given in parameters for the Pressure method")

    p = np.reshape(p, newshape=[params['Geometry Parameters']['Nxi'], params['Geometry Parameters']['Neta']], order="F")
    p_x = coefficient_mats.Grad_Xi * p
    p_y = p * coefficient_mats.Grad_Eta.transpose()

    return p_x, p_y, divU, p


def calcGradPRHS(params, coefficient_mats, rhs_u_wop, rhs_v_wop, rho, Q):
    if params['Geometry Parameters']['TransformedGrid']:
        raise NotImplemented("Transformed grids not implemented")
    else:
        dt = params['Time Parameters']['dt']
        Q = np.reshape(Q, newshape=[params['Geometry Parameters']['Nxi'], params['Geometry Parameters']['Neta']],
                       order="F")
        divRHS = coefficient_mats.Div_Xi * rhs_u_wop + rhs_v_wop * coefficient_mats.Div_Eta.transpose()
        Source = divRHS - Q / dt

    NN = params['Geometry Parameters']['Nxi'] * params['Geometry Parameters']['Neta']
    rho_sub = np.reshape(rho, newshape=NN, order="F")
    RHO_inv = diags(np.reciprocal(rho_sub), format="csc")

    Lp = coefficient_mats.Div_Xi_kron * RHO_inv * coefficient_mats.Grad_Xi_kron \
        + coefficient_mats.Div_Eta_kron * RHO_inv * coefficient_mats.Grad_Eta_kron

    LU = linalg.splu(Lp)

    if params['Equation Parameters']['PressureCalcMethod'] == 'backslashLaplace':
        p = scipy.sparse.linalg.inv(Lp) * np.squeeze(np.reshape(Source, newshape=[-1, 1], order="F"))
    elif params['Equation Parameters']['PressureCalcMethod'] == 'backslashLaplaceNeumann':
        raise NotImplemented("backslashLaplaceNeumann not implemented as a Pressure method")
    elif params['Equation Parameters']['PressureCalcMethod'] == 'LULaplace':
        p = LU.solve(np.squeeze(np.reshape(Source, newshape=[-1, 1], order="F")))
    elif params['Equation Parameters']['PressureCalcMethod'] == 'FFT':
        raise NotImplemented("FFT is not implemented as a Pressure method")
    else:
        raise NotImplemented("Please select from the following options given in parameters for the Pressure method")

    p = np.reshape(p, newshape=[params['Geometry Parameters']['Nxi'], params['Geometry Parameters']['Neta']], order="F")
    p_x = coefficient_mats.Grad_Xi * p
    p_y = p * coefficient_mats.Grad_Eta.transpose()

    return p_x, p_y, divRHS, p
