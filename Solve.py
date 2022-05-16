import RungeKutta
import MakeDivFree
import Chemistry

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal


def heat_source(grid):
    mu_x = 0.5
    variance_x = 0.008
    mu_y = 0.5
    variance_y = 0.008
    pos = np.empty(grid.X.shape + (2,))
    pos[:, :, 0] = grid.X
    pos[:, :, 1] = grid.Y
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    profile = 1 + (np.max(rv.pdf(pos)) - rv.pdf(pos)) / (np.max(rv.pdf(pos)) - np.min(rv.pdf(pos)))

    return 1 / profile


class CalcSolver:
    def __init__(self, params, refVar, numbers_ND, grid, coefficient_mats, prim_var) -> None:
        self.ep = params['Equation Parameters']
        self.gp = params['Geometry Parameters']
        self.dp = params['Derivative Parameters']
        self.tp = params['Time Parameters']
        self.fp = params['Filter Parameters']

        self.output_var = []
        ########################################################
        print("############Solver calculation start#############")
        print("\n")

        values_n = prim_var
        for ts in range(self.tp['TimeSteps']):
            dt = self.tp['dt']
            print(f'Current time step: {ts}; Time step size: {dt}; Current time: {0 + ts * dt}')

            values_1, fluxes, rhs, errs = self.timeStepGeneral(values_n, params, refVar, numbers_ND, grid,
                                                               coefficient_mats)
            values_n = values_1
            self.output_var.append(values_1)

        self.output_var = ReDimensionalize(self.output_var, refVar, self.tp['TimeSteps'])

        print("\n")
        print("############Solver calculation end###############")

    def timeStepGeneral(self, values_n, params, refVar, numbers_ND, grid, coefficient_mats):
        scheme = RungeKutta.RungeKutta(self.tp['TimeScheme'])
        stages = len(scheme.b)

        rhs = np.zeros((self.gp['Nxi'], self.gp['Neta'], self.ep['number'], stages), dtype=float)
        fluxes = 'dummy'

        # To check convergence if implicit scheme
        values_1_old = values_n
        values_1 = values_n  # Just to allocate

        # Fix point iteration
        for j in range(self.tp['FixPointIter']):
            for kStep in range(stages):
                rhs, fluxes = self.__iterStep(values_n, rhs, fluxes, kStep, scheme, params, numbers_ND,
                                              coefficient_mats, grid, refVar)

            values_1 = values_n
            for kStep in range(stages):
                values_1 = values_1 + self.tp['dt'] * scheme.b[kStep] * rhs[:, :, :, kStep]

            # divUNew = coefficient_mats.Div_Xi * values_1[:, :, 1] + values_1[:, :, 2] * coefficient_mats.Div_Eta.transpose()
            # print(np.linalg.norm(divUNew))
            # Correct the velocities
            values_1, divUOld, divUNew, p_project, _, _ = MakeDivFree.makeDivergenceFree(params, numbers_ND,
                                                                                         coefficient_mats, values_1,
                                                                                         refVar, grid)
            values_1[:, :, 3] = p_project
            values_1[:, :, 4] = (refVar['P_th'] / refVar['p_ref']) / values_1[:, :, 0]

            if not scheme.implicit:
                print('explicit scheme do not require iterations')
                break

            # Calculate the error of iteration for implicit schemes
            errAll = values_1_old[:, :, 1:3] - values_1[:, :, 1:3]
            errTotal = np.linalg.norm(np.squeeze(np.reshape(errAll, newshape=[-1, 1], order="F")))

            # Test convergence in a different manner, in case slow convergence is expected
            converged = errTotal < self.tp['FixPointIterErr']

            if converged:
                print('solution converged')
                break

            print('error per iteration : ', j, 'is : ', errTotal)

            values_1_old = values_1

        if not scheme.implicit:
            converged = True
            errTotal = 0
        if not converged:
            print('Warning not converged')

        return values_1, fluxes, rhs, errTotal

    def __iterStep(self, values_n, rhs, fluxes, kStep, scheme, params, numbers_ND, coefficient_mats, grid, refVar):

        values_sub = self.__get_values_substep(values_n[:, :, 0:7], rhs[:, :, 0:7, :], kStep, scheme)[0]

        # Calculate the new reaction rates
        Chemistry.CalcReactionRates(values_sub, refVar)

        NN = self.gp['Nxi'] * self.gp['Neta']
        rho_sub = np.reshape(values_sub[:, :, 0], newshape=NN, order="F")
        u_sub = np.reshape(values_sub[:, :, 1], newshape=NN, order="F")
        v_sub = np.reshape(values_sub[:, :, 2], newshape=NN, order="F")
        T_sub = np.reshape(values_sub[:, :, 4], newshape=NN, order="F")
        Y_sub = np.reshape(values_sub[:, :, 5] / values_sub[:, :, 0], newshape=NN, order="F")
        omegaDot_sub = np.reshape(values_sub[:, :, 6], newshape=NN, order="F")

        U = diags(u_sub, format="csc")
        V = diags(v_sub, format="csc")

        Du = (U * coefficient_mats.Grad_Xi_kron + V * coefficient_mats.Grad_Eta_kron)
        Q_th = (1 / (numbers_ND['Pr'] * numbers_ND['Re'])) * coefficient_mats.LaplaceEnergy.dot(np.reciprocal(rho_sub))
        Q_r = refVar['h_0'] * omegaDot_sub / (refVar['gamma'] * refVar['P_th'] / refVar['p_ref'])
        Q_h = 0.0005 / (refVar['gamma'] * refVar['P_th'] / refVar['p_ref']) * heat_source(grid)
        Q = Q_th + Q_r + np.reshape(Q_h, newshape=NN, order="F")

        fric_U = (1 / (numbers_ND['Re'] * rho_sub)) * (coefficient_mats.LaplaceFric.dot(u_sub) +
                                                       (1 / 3) * coefficient_mats.Grad_Xi_kron.dot(Q))
        fric_V = (1 / (numbers_ND['Re'] * rho_sub)) * (coefficient_mats.LaplaceFric.dot(v_sub) +
                                                       (1 / 3) * coefficient_mats.Grad_Eta_kron.dot(Q))
        rhs_rho = - coefficient_mats.Div_Xi_kron.dot(rho_sub * u_sub) - coefficient_mats.Div_Eta_kron.dot(
            rho_sub * v_sub)
        rhs_u_woP = - (Du.dot(u_sub) - fric_U) #+ 1 / numbers_ND['Fr'] ** 2
        rhs_v_woP = - (Du.dot(v_sub) - fric_V) #+ 1 / numbers_ND['Fr'] ** 2

        # Reactive part
        D = refVar['D_0'] * (T_sub ** refVar['n']) / rho_sub
        rhs_Y = - coefficient_mats.Div_Xi_kron.dot(rho_sub * Y_sub * u_sub) - \
                coefficient_mats.Div_Eta_kron.dot(rho_sub * Y_sub * v_sub) + (
                        1 / (numbers_ND['Re'] * numbers_ND['Sc'])) * (coefficient_mats.Div_Xi_kron.dot(
            rho_sub * D * coefficient_mats.Grad_Xi_kron.dot(Y_sub)) + coefficient_mats.Div_Eta_kron.dot(
            rho_sub * D * coefficient_mats.Grad_Eta_kron.dot(Y_sub))) - numbers_ND['Da'] * omegaDot_sub

        rhs_rho = np.reshape(rhs_rho, newshape=[self.gp['Nxi'], self.gp['Neta']], order="F")
        rhs_u_woP = np.reshape(rhs_u_woP, newshape=[self.gp['Nxi'], self.gp['Neta']], order="F")
        rhs_v_woP = np.reshape(rhs_v_woP, newshape=[self.gp['Nxi'], self.gp['Neta']], order="F")
        rhs_Y = np.reshape(rhs_Y, newshape=[self.gp['Nxi'], self.gp['Neta']], order="F")

        rhs_u, rhs_v, divRHSOld, divRHSNew = MakeDivFree.makeDivergenceFreeRHS(params, numbers_ND,
                                                                               coefficient_mats,
                                                                               values_sub, rhs_u_woP,
                                                                               rhs_v_woP, refVar, grid)
        rhs[:, :, 0, kStep] = rhs_rho
        rhs[:, :, 1, kStep] = rhs_u
        rhs[:, :, 2, kStep] = rhs_v
        rhs[:, :, 5, kStep] = rhs_Y

        return rhs, fluxes

    def __get_values_substep(self, values_n, rhs, kStep, scheme):
        a = scheme.a
        c = scheme.c
        stageNo = int(np.size(a, 0))
        values_sub = values_n

        for j in range(stageNo):
            values_sub = values_sub + (self.tp['dt'] * a[kStep, j] if a.ndim > 1 else a[j]) * rhs[:, :, :, j]

        t_sub = c[kStep] * self.tp['dt']

        return values_sub, t_sub


def ReDimensionalize(output_var, refVar, TimeSteps):
    for ts in range(TimeSteps):
        rho_ND = output_var[ts][:, :, 0]
        u_ND = output_var[ts][:, :, 1]
        v_ND = output_var[ts][:, :, 2]
        p_ND = output_var[ts][:, :, 3]
        T_ND = output_var[ts][:, :, 4]
        rhoY_ND = output_var[ts][:, :, 5]
        omegaDot_ND = output_var[ts][:, :, 6]

        output_var[ts][:, :, 0] = rho_ND * refVar['rho_ref']
        output_var[ts][:, :, 1] = u_ND * refVar['u_ref']
        output_var[ts][:, :, 2] = v_ND * refVar['v_ref']
        output_var[ts][:, :, 3] = p_ND * refVar['p_ref']
        output_var[ts][:, :, 4] = T_ND * refVar['T_ref']
        rhoY = rhoY_ND * refVar['mass_frac_ref'] * refVar['rho_ref']
        output_var[ts][:, :, 5] = rhoY / output_var[ts][:, :, 0]
        output_var[ts][:, :, 6] = omegaDot_ND * refVar['omegaDot_ref']

    return output_var
