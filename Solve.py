import RungeKutta
import MakeDivFree

import numpy as np
from scipy.sparse import diags


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
                                              coefficient_mats, grid)

            values_1 = values_n
            for kStep in range(stages):
                values_1 = values_1 + self.tp['dt'] * scheme.b[kStep] * rhs[:, :, :, kStep]

            # Correct the velocities
            values_1, divUOld, divUNew, p_project, _, _ = MakeDivFree.makeDivergenceFree(params, numbers_ND,
                                                                                         coefficient_mats, values_1)
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

    def __iterStep(self, values_n, rhs, fluxes, kStep, scheme, params, numbers_ND, coefficient_mats, grid):

        values_sub = self.__get_values_substep(values_n[:, :, 0:5], rhs[:, :, 0:5, :], kStep, scheme)[0]
        values_sub, divUOld, divUNew, p_project, p_x, p_y = MakeDivFree.makeDivergenceFree(params, numbers_ND,
                                                                                           coefficient_mats, values_sub)

        NN = self.gp['Nxi'] * self.gp['Neta']
        rho_sub = np.reshape(values_sub[:, :, 0], newshape=NN, order="F")
        u_sub = np.reshape(values_sub[:, :, 1], newshape=NN, order="F")
        v_sub = np.reshape(values_sub[:, :, 2], newshape=NN, order="F")

        U = diags(u_sub, format="csc")
        V = diags(v_sub, format="csc")

        Du = (U * coefficient_mats.Grad_Xi_kron + V * coefficient_mats.Grad_Eta_kron)
        Q = (1 / (numbers_ND['Pr'] * numbers_ND['Re'])) * coefficient_mats.LaplaceEnergy.dot(np.reciprocal(rho_sub))
        fric_U = (1 / (numbers_ND['Re'] * rho_sub)) * (coefficient_mats.LaplaceFric.dot(u_sub) +
                                                       (1 / 3) * coefficient_mats.Grad_Xi_kron.dot(Q))
        fric_V = (1 / (numbers_ND['Re'] * rho_sub)) * (coefficient_mats.LaplaceFric.dot(v_sub) +
                                                       (1 / 3) * coefficient_mats.Grad_Eta_kron.dot(Q))

        rhs_rho = - coefficient_mats.Div_Xi_kron.dot(rho_sub * u_sub) - coefficient_mats.Div_Eta_kron.dot(
            rho_sub * v_sub)
        rhs_u_woP = - (Du.dot(u_sub) - fric_U) #+ 1 / numbers_ND['Fr'] ** 2
        rhs_v_woP = - (Du.dot(v_sub) - fric_V) #+ 1 / numbers_ND['Fr'] ** 2

        rhs_rho = np.reshape(rhs_rho, newshape=[self.gp['Nxi'], self.gp['Neta']], order="F")
        rhs_u = np.reshape(rhs_u_woP, newshape=[self.gp['Nxi'], self.gp['Neta']],
                           order="F") - p_x / values_sub[:, :, 0]
        rhs_v = np.reshape(rhs_v_woP, newshape=[self.gp['Nxi'], self.gp['Neta']],
                           order="F") - p_y / values_sub[:, :, 0]

        rhs[:, :, 0, kStep] = rhs_rho
        rhs[:, :, 1, kStep] = rhs_u
        rhs[:, :, 2, kStep] = rhs_v

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
        x = output_var[ts][:, :, 0] * refVar['rho_ref']
        output_var[ts][:, :, 0] = x
        x = output_var[ts][:, :, 1] * refVar['u_ref']
        output_var[ts][:, :, 1] = x
        x = output_var[ts][:, :, 2] * refVar['v_ref']
        output_var[ts][:, :, 2] = x
        x = output_var[ts][:, :, 3] * refVar['p_ref']
        output_var[ts][:, :, 3] = x
        x = output_var[ts][:, :, 4] * refVar['T_ref']
        output_var[ts][:, :, 4] = x

    return output_var
