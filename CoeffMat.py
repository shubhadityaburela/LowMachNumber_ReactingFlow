import numpy as np
from scipy import sparse
import warnings

import PressureInit
import Filter
from Mat import Mat


class CoeffMat:
    def __init__(self, params, grid) -> None:
        # Instantiate the parameters for initializing the coefficient matrices
        self.ep = params['Equation Parameters']
        self.gp = params['Geometry Parameters']
        self.dp = params['Derivative Parameters']
        self.tp = params['Time Parameters']
        self.fp = params['Filter Parameters']
        self.grid = grid

        # Coefficient array for the matrix creation. More details in self.stencil_selection
        self.__GradCoef = None
        self.__DivCoef = None
        self.__GradCoefUL = None
        self.__GradCoefBR = None
        self.__DivCoefUL = None
        self.__DivCoefBR = None

        self.__ExpFrictn = None
        self.__FricLapCoeff = None

        # Coefficient Matrices to be used in Solver
        self.Grad_Xi = None
        self.Div_Xi = None
        self.Grad_Eta = None
        self.Div_Eta = None
        self.Grad_Xi_kron = None  # Coefficient Matrices in the Kronecker form
        self.Div_Xi_kron = None
        self.Grad_Eta_kron = None
        self.Div_Eta_kron = None

        self.BarLaplace = None
        self.LaplaceFric = None
        self.LaplaceEnergy = None

        self.ExpFric_Xi = None  # Explicit friction coefficient matrices
        self.ExpFric_Eta = None

        # LU decomposition object stored here which is generated in 'init_pressure'
        self.LU = None

        # Filter matrices
        self.A_Xi = None
        self.B_Xi = None
        self.A_Eta = None
        self.B_Eta = None

        ########################################################
        self.initMat()

        self.initFilterMat()
        ########################################################

    def initMat(self):
        if self.dp['name'] == '1stOrder':
            self.__GradCoef = np.array([0, -1, 1, 0, 0])
            self.__DivCoef = np.array([0, 0, -1, 1, 0])
        elif self.dp['name'] == '2ndOrder':
            self.__GradCoef = np.array([1 / 2, -2, 3 / 2, 0, 0])
            self.__DivCoef = np.array([0, 0, -3 / 2, 2, -1 / 2])
        elif self.dp['name'] == '3rdOrder':
            self.__GradCoefUL = np.array([[-11 / 6, 3, -3 / 2, 1 / 3],
                                          [-1 / 3, -1 / 2, 1, -1 / 6]])  # Non periodic (left)
            self.__GradCoef = np.array([1 / 6, -1, 1 / 2, 1 / 3, 0])  # inner
            self.__GradCoefBR = np.array([-1 / 3, 3 / 2, -3, 11 / 6])

            self.__DivCoefUL = np.array([-11 / 6, 3, -3 / 2, 1 / 3])  # Non periodic (left)
            self.__DivCoef = np.array([0, -1 / 3, -1 / 2, 1, -1 / 6])  # inner
            self.__DivCoefBR = np.array([[1 / 6, -1, 1 / 2, 1 / 3],
                                         [-1 / 3, 3 / 2, -3, 11 / 6]])  # right
        elif self.dp['name'] == '3rdOrderMixed':
            alpha = self.dp['MixFactor']
            self.__GradCoef = alpha * np.array([1 / 6, -1, 1 / 2, 1 / 3, 0]) + (1 - alpha) * \
                              np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
            self.__DivCoef = alpha * np.array([0, -1 / 3, -1 / 2, 1, -1 / 6]) + (1 - alpha) * \
                             np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        elif self.dp['name'] == '5thOrder':
            self.__GradCoefUL = np.array([[-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
                                          [-1 / 5, -13 / 12, 2, -1, 1 / 3, -1 / 20],
                                          [1 / 20, -1 / 2, -1 / 3, 1, -1 / 4, 1 / 30]])
            self.__GradCoef = np.array([-2, 15, -60, 20, 30, -3, 0]) / 60
            self.__GradCoefBR = np.array([[1 / 20, -1 / 3, 1, -2, 13 / 12, 1 / 5],
                                          [-1 / 5, 5 / 4, -10 / 3, 5, -5, 137 / 60]])

            self.__DivCoefUL = np.array([[-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
                                         [-1 / 5, -13 / 12, 2, -1, 1 / 3, -1 / 20]])
            self.__DivCoef = np.array([0, 3, -30, -20, 60, -15, 2]) / 60
            self.__DivCoefBR = np.array([[-1 / 30, 1 / 4, -1, 1 / 3, 1 / 2, -1 / 20],
                                         [1 / 20, -1 / 3, 1, -2, 13 / 12, 1 / 5],
                                         [-1 / 5, 5 / 4, -10 / 3, 5, -5, 137 / 60]])
        elif self.dp['name'] == '7thOrder':
            self.__GradCoef = np.array([3, -28, 126, -420, 105, 252, -42, 4, 0]) / (140 * 3)
            self.__DivCoef = np.array([0, -4, 42, -252, -105, 420, -126, 28, -3]) / (140 * 3)
        elif self.dp['name'] == 'secondSymmetric':
            self.__GradCoef = np.array([-1 / 2, 0, 1 / 2])
            self.__DivCoef = self.__GradCoef
            self.__GradCoefUL = np.array([-1, 1, 0])
            self.__GradCoefBR = np.array([0, -1, 1])
            self.__DivCoefUL = self.__GradCoefUL
            self.__DivCoefBR = self.__GradCoefBR
        else:
            raise NotImplemented("Please select the derivative order from the list already implemented")

        Nxi = self.gp['Nxi']
        Neta = self.gp['Neta']
        dXi = self.grid.dXi
        dEta = self.grid.dEta

        M = Mat()
        if self.gp['XI_Periodic']:
            self.Grad_Xi = M.D_periodic(self.__GradCoef, Nxi, dXi)
            self.Div_Xi = M.D_periodic(self.__DivCoef, Nxi, dXi)
        else:
            self.Grad_Xi = M.D_nonperiodic(self.__GradCoef, Nxi, dXi,
                                           self.__GradCoefUL, self.__GradCoefBR)
            self.Div_Xi = M.D_nonperiodic(self.__DivCoef, Nxi, dXi,
                                          self.__DivCoefUL, self.__DivCoefBR)

        if self.gp['ETA_Periodic']:
            self.Grad_Eta = M.D_periodic(self.__GradCoef, Neta, dEta)
            self.Div_Eta = M.D_periodic(self.__DivCoef, Neta, dEta)
        else:
            self.Grad_Eta = M.D_nonperiodic(self.__GradCoef, Neta, dEta,
                                            self.__GradCoefUL, self.__GradCoefBR)
            self.Div_Eta = M.D_nonperiodic(self.__DivCoef, Neta, dEta,
                                           self.__DivCoefUL, self.__DivCoefBR)

        # Create the matrices in Kronecker form
        if self.dp['KroneckerForm']:
            self.Grad_Xi_kron = sparse.kron(sparse.eye(Neta, format="csc"), self.Grad_Xi, format="csc")
            self.Div_Xi_kron = sparse.kron(sparse.eye(Neta, format="csc"), self.Div_Xi, format="csc")
            self.Grad_Eta_kron = sparse.kron(self.Grad_Eta, sparse.eye(Nxi, format="csc"), format="csc")
            self.Div_Eta_kron = sparse.kron(self.Div_Eta, sparse.eye(Nxi, format="csc"), format="csc")

        # Explicit Friction matrix calculation
        if self.dp['ExplicitFriction'] == '8thOrder':
            self.__ExpFrictn = np.array([-1 / 560, 8 / 315, -1 / 5, 8 / 5,
                                         -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])
            self.ExpFric_Xi = M.D_periodic(self.__ExpFrictn, Nxi, dXi ** 2)
            self.ExpFric_Eta = M.D_periodic(self.__ExpFrictn, Neta, dEta ** 2)
        elif self.dp['ExplicitFriction'] is None:
            warnings.warn("Explicit Friction is set to None.")
            pass
        else:
            raise NotImplemented(
                "Please select one of the given options from the list of options for Explicit Friction")

        # Laplace matrices
        if self.gp['XI_Periodic']:
            P_Xi = 1
        else:
            p_Xi = sparse.eye(Nxi, format="csc")
            p_Xi[0, 0] = 0
            p_Xi[-1, -1] = 0

            P_Xi = sparse.kron(sparse.eye(Neta, format="csc"), p_Xi, format="csc")

        if self.gp['ETA_Periodic']:
            P_Eta = 1
        else:
            p_Eta = sparse.eye(Neta, format="csc")
            p_Eta[0, 0] = 0
            p_Eta[-1, -1] = 0

            P_Eta = sparse.kron(p_Eta, sparse.eye(Nxi, format="csc"), format="csc")

        P = P_Xi * P_Eta

        if self.gp['TransformedGrid']:
            raise NotImplemented("Transformed grids are not implemented at the moment")
        else:
            #  Tentative initiation
            if self.dp['KroneckerForm']:
                # This should only be used for the pressure and not for friction
                bar_laplace = self.Div_Xi_kron * P * self.Grad_Xi_kron + \
                              self.Div_Eta_kron * P * self.Grad_Eta_kron
                if not self.gp['XI_Periodic'] and not self.gp['ETA_Periodic']:
                    # for points (0, 0)
                    k = 0
                    bar_laplace[k, :] = 0
                    bar_laplace[k, k] = 1

                    # for points (Nxi - 1, 0)
                    k = Nxi - 1
                    bar_laplace[k, :] = 0
                    bar_laplace[k, k] = 1

                    # for points (0, Neta - 1)
                    k = (Neta - 2) * (Nxi - 1) + 1
                    bar_laplace[k, :] = 0
                    bar_laplace[k, k] = 1

                    # for points (Nxi - 1, Neta - 1)
                    k = (Neta - 1) * (Nxi - 1)
                    bar_laplace[k, :] = 0
                    bar_laplace[k, k] = 1

                self.BarLaplace = bar_laplace

                # Energy Laplace
                self.LaplaceEnergy = self.BarLaplace

        if self.ep['initPressureSolver'] == 'directLU_Init':
            self.LU = PressureInit.directLU_Init(self.BarLaplace)
        else:
            raise NotImplemented("Please select the appropriate method for initializing the pressure solver")

        if self.dp['KroneckerForm']:
            if self.dp['FrictionLaplace'] == 'DivGrad':
                warnings.warn("'DivGrad' Might not work for non-periodic cases and the friction laplace is anyway not "
                              "required here")
            elif self.dp['FrictionLaplace'] == 'explicit7Point':
                self.__FricLapCoeff = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
                D_Xi = M.D_periodic(self.__FricLapCoeff, Nxi, dXi)
                D_Eta = M.D_periodic(self.__FricLapCoeff, Neta, dEta)
                D_Xi_kron = sparse.kron(sparse.eye(Neta, format="csc"), D_Xi, format="csc")
                D_Eta_kron = sparse.kron(D_Eta, sparse.eye(Nxi, format="csc"), format="csc")

                self.LaplaceFric = D_Xi_kron + D_Eta_kron
            else:
                raise NotImplemented("Other methods for FrictionLaplace are not implemented. Please select from the "
                                     "pre-given list")

        pass

    def initFilterMat(self):
        if self.fp['filtering']:
            if self.fp['name'] == 'padeFilterConservative':
                f = Filter.Filter(self.fp, self.gp, self.ep)
                self.A_Xi, self.B_Xi, self.A_Eta, self.B_Eta = f.initPadeFilter()
            elif self.fp['name'] == 'padeFilter':
                f = Filter.Filter(self.fp, self.gp, self.ep)
                self.A_Xi, self.B_Xi, self.A_Eta, self.B_Eta = f.initPadeFilter()
            elif self.fp['name'] == 'padeFilterPrimitive':
                f = Filter.Filter(self.fp, self.gp, self.ep)
                self.A_Xi, self.B_Xi, self.A_Eta, self.B_Eta = f.initPadeFilter()
            else:
                raise NotImplemented("Other filters not implemented. Please select one from the list")

        pass
