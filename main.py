import sys
import warnings

from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal

import Grid
import ND_num
import CoeffMat
import plot
import MakeDivFree
from Solve import CalcSolver

from scipy.sparse import linalg
import scipy
import numpy as np

"""
In this code we solve the Bunsen burner flame model with the help of Finite differences.

A little theory :

The broad category of N-S flow problems with increasing physical richness:
- Incompressible Navier stokes equations
    These problems does not have explicit equation to calculate the pressure, so a pressure poisson equation is 
    formulated with the help of continuity equation and momentum equation and a pressure correction is done at each
    time step to satisfy the divergence free condition of the problem. The pressure calculation here does not hold 
    a particularly great significance as it is extremely small whereas the pressure gradients are an important 
    variable as these are used to make the solution divergence free at each time step. Thus only important variables  
    now are U, V, W (velocities).
     
- Incompressible Navier stokes equations with variable density
    These problems take into account the dependency of the density on the temperature (rho = rho(T)). Normally the 
    density is dependent upon both the temperature and pressure but as we are still in the incompressible regime 
    the pressure variation is negligible so the density only depends upon the temperature. Here we have an extra density 
    term in the continuity as well as the momentum equations. These are a special kind of problems as these can be 
    interpreted as "reduced physical richness" forms of the compressible N-S equations. More on this later...
      
- Compressible Navier stokes equations
    These problems include continuity, momentum and energy equations with density, pressure, (U, V, W), and temperature
    as unknowns. We thus have 6 unknowns with only 5 equations for which the equation of state acts as the 6th equation.
    Thus the problem is now solvable. There are a lot of different variations possible for this problem which can be 
    looked upon in the literature.
     
- Compressible reactive Navier stokes equations
    These problems are an extension of the compressible Navier stokes problem in a sense that we introduce chemical
    processes into the equation thus increasing the complexity of the problem in terms of physical richness.
    
    
For the Bunsen burner flame the flame speed is generally of the order of 5-10 m/s which is orders of magnitude smaller
than the acoustic wave speed in these regimes. We are already in a very high temperature regime in the bunsen flame 
case where the acoustic velocity in the gaseous medium is of the order of 1000 m/s. For the compressible flow solvers
this results into extremely slow solution convergence as it has to deal with multiple scales corresponding to the 
flow velocity and acoustic velocity in high temperatures. We therefore go for a low-mach number asymptotic which 
inherently "reduces" our compressible reacting Navier stokes equations into an incompressible type with variable 
density.

Our Model:
    We have a low-mach number incompressible flow with variable density containing continuity, momentum and species
    chemical reaction equations along with a pressure poisson equation to be solved for pressure correction.
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Input parameters
    params = {'Geometry Parameters':
                  {'Nxi': 48,  # X and Y grid points
                   'Neta': 49,
                   'Lxi': 1,  # Lengths of the domain in the X and Y direction
                   'Leta': 1,
                   'X_Xi': 1,  # Derivative of the cartesian coordinates with respect to computational coordinates
                   'X_Eta': 0,
                   'Y_Xi': 0,
                   'Y_Eta': 1,
                   'J': 1,  # Jacobian of the mesh
                   'XI_Periodic': True,  # Periodicity of the grid in X and Y direction
                   'ETA_Periodic': True,
                   'TransformedGrid': False,  # 'True' if non-cartesian grid is implemented
                   'X': None,  # Grid coordinates in matrix format for both cartesian and computational coordinates
                   'Y': None,
                   'XI': None,
                   'ETA': None
                   },
              'Equation Parameters':
                  {'iterStep': 'iterStepSkew',  # Type of solver for the iteration step
                   'number': 5,  # Number of individual equations to be solved in the 2d incompressible NS equations
                   'initPressureSolver': 'directLU_Init',  # Initialize the pressure solver with appropriate method
                   'onlyPressureGradient': False,  # Calculate the pressure gradient for adjoint calculations
                   'PressureCalcMethod': 'LULaplace',
                   # Options : 'backslashLaplace', 'backslashLaplaceNeumann', 'LULaplace', 'FFT'
                   'FractionalStep': False,  # Whether to perform fractional step for pressure calculation
                   'makeDivergenceFree': True  # Whether to make the problem divergence free
                   },
              'Derivative Parameters':
                  {'name': '5thOrder',
                   # Derivative order for equations. Options : '1stOrder', '2ndOrder', '3rdOrder', '3rdOrderMixed', '5thOrder', '7thOrder', 'secondSymmetric'
                   'FrictionLaplace': 'explicit7Point',
                   # Method to calculate the Friction laplace matrix. Options : 'DivGrad', 'explicit7Point'
                   'MixFactor': 0.5,  # Mix factor for the '3rdOrderMixed' derivative order case
                   'KroneckerForm': True,
                   # Variable to define whether the kronecker forms of the coefficient matrices are required
                   'ExplicitFriction': None
                   # This creates explicit friction matrices for the problem. Options : '8thOrder', None
                   },
              'Time Parameters':
                  {'TimeScheme': 'classicRK4',
                   # Scheme for the Time integration. Options : 'gauss2', 'gauss4', 'gauss6', 'classicRK4', 'EulerExplicit'
                   'TimeSteps': 100,  # Time steps for the Time integration
                   'dt': 0.005,  # Time step size
                   'FixPointIter': 60,  # Number of fixed point iterations for the implicit schemes
                   'FixPointIterErr': 1e-5,  # The convergence condition for the fixed point iteration
                   'filterOldStep': False,
                   'filterRHS': False
                   },
              'Filter Parameters':
                  {'filtering': False,  # Choose whether to apply the filtering
                   'name': 'padeFilter',
                   # Name of the filter. Options : 'padeFilterConservative', 'padeFilter', 'padeFilterPrimitive'
                   'Type': 'non-per',  # Type of the filter. Options : 'non-per', 'per'
                   'order': 2,  # Order of the filter. Options : 2, 4, 6
                   'alpha': 0.35  # Filter factor
                   }
              }

    # Introduce reference variables and problem constants
    refVar = {
        'X_ref': 1,
        'Y_ref': 1,
        'u_ref': 1,
        'v_ref': 1,
        't_ref': 1,
        'rho_ref': 1.2,
        'p_ref': 1e5,
        'T_ref': 300,
        'gamma': 1.399,
        'mu': 18e-6,
        'g': 9.81,
        'Cv': 718,
        'K': 500e-3,
        'P_th': 2e5
    }
    # Introduce dimensionless numbers
    numbers_ND = ND_num.num_ND(refVar)

    # Construct the grid (Cartesian or transformed) by non-dimensionalizing the dimensions
    grid = Grid.CartesianGrid(params, refVar)

    # Create Coefficient matrices
    coefficient_mats = CoeffMat.CoeffMat(params, grid)

    # Input model
    prim_var = np.zeros((params['Geometry Parameters']['Nxi'], params['Geometry Parameters']['Neta'],
                         params['Equation Parameters']['number']), dtype=float)
    case = 'vortices'
    if case == 'vortices':
        # Vortex positions (x) and (y)
        x0s = np.array([1 / 2, 1 / 2])
        y0s = np.array([0.4, 0.6])
        betas = np.array([1 / 14, 1 / 14])  # Vortex core size
        alphas = np.array([10, -10])  # Strength

        rho = 1.5  # Initial density
        omegaShift = True
        omegaShiftValue = None
        LaplaceInvert = 'LULaplace'
        fromOmegaDerivative = False
        preFactorType = '1'  # Options : '1', '1-R2'

        Nxi = params['Geometry Parameters']['Nxi']
        Neta = params['Geometry Parameters']['Neta']
        omega = np.zeros((Nxi, Neta), dtype=float)
        for i in range(len(x0s)):
            x0 = x0s[i]
            y0 = y0s[i]
            beta = betas[i]
            alpha = alphas[i]
            R2 = ((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) / beta ** 2

            if preFactorType == '1-R2':
                factor = 1 - R2
            elif preFactorType == '1':
                factor = 1
            else:
                raise NotImplemented("Please select proper preFactorType from the options given")

            omega_part = alpha * factor * np.exp(-R2)
            omega = omega + omega_part

        if omegaShift:
            omegaShiftValue = np.sum(np.sum(omega, axis=0)) / (Nxi * Neta)
            omega = omega - omegaShiftValue

        NN = Nxi * Neta
        J = params['Geometry Parameters']['J']
        Jomega = np.reshape((J * omega), newshape=NN, order="F")

        InvertMethod = LaplaceInvert
        if InvertMethod == 'FFT':
            raise NotImplemented("FFT Model not implemented yet")
        else:
            b = coefficient_mats.LU.solve(Jomega)
            Psi = np.reshape(b, newshape=[Nxi, Neta], order="F")
            Psi = scipy.sparse.csc_matrix(Psi)

        Psi_x = coefficient_mats.Grad_Xi * Psi
        Psi_y = Psi * coefficient_mats.Grad_Eta.transpose()

        if fromOmegaDerivative:
            sumU = np.sum(np.sum(-Psi_y, axis=0))
            sumV = np.sum(np.sum(Psi_x, axis=0))

            omega = np.reshape(omega, newshape=[Nxi, Neta], order="F")

            omega_x = coefficient_mats.Grad_Xi * omega
            omega_y = omega * coefficient_mats.Grad_Eta.transpose()

            if params['Geometry Parameters']['TransformedGrid']:
                raise NotImplemented("Transformed grid not implemented")

            Jomega_x = np.squeeze(np.reshape((J * omega_x), newshape=[-1, 1], order="F"))
            Jomega_y = np.squeeze(np.reshape((J * omega_y), newshape=[-1, 1], order="F"))

            Psi_x = coefficient_mats.LU.solve(Jomega_x)
            Psi_y = coefficient_mats.LU.solve(Jomega_y)
            Psi_x = np.reshape(Psi_x, newshape=[Nxi, Neta], order="F")
            Psi_y = np.reshape(Psi_y, newshape=[Nxi, Neta], order="F")

            sumUnew = np.sum(np.sum(-Psi_y))
            sumVnew = np.sum(np.sum(Psi_x))

            u = - np.reshape(Psi_y, newshape=[Nxi, Neta], order="F") + (-sumUnew + sumU) / NN
            v = np.reshape(Psi_x, newshape=[Nxi, Neta], order="F") + (-sumVnew + sumV) / NN
        else:
            u = - np.reshape(Psi_y, newshape=[Nxi, Neta], order="F")
            v = np.reshape(Psi_x, newshape=[Nxi, Neta], order="F")

        mu_x = 0.5
        variance_x = 0.008
        mu_y = 0.5
        variance_y = 0.008
        pos = np.empty(grid.X.shape + (2,))
        pos[:, :, 0] = grid.X
        pos[:, :, 1] = grid.Y
        rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

        prim_var[:, :, 0] = rho + (np.max(rv.pdf(pos)) - rv.pdf(pos)) / (np.max(rv.pdf(pos)) - np.min(rv.pdf(pos)))
        prim_var[:, :, 1] = u.todense()
        prim_var[:, :, 2] = v.todense()
        prim_var[:, :, 4] = ((refVar['P_th'] / refVar['p_ref']) / (prim_var[:, :, 0] / refVar['rho_ref'])) * refVar['T_ref']
    elif case == 'taylorGreen':
        raise NotImplemented("taylorGreen not implemented")
    elif case == 'shearLayer':
        raise NotImplemented("shearLayer not implemented")
    else:
        raise NotImplemented("Please select the appropriate Initial Model for the problem from the list")

    # Make the input values Numerically divergence free
    if params['Equation Parameters']['makeDivergenceFree']:
        prim_var, DivergenceOld, DivergenceNew, _, _, _ = MakeDivFree.makeDivergenceFree(params, numbers_ND,
                                                                                         coefficient_mats, prim_var)
        print(np.linalg.norm(DivergenceOld), np.linalg.norm(DivergenceNew))
        if np.linalg.norm(DivergenceNew) > 1e-12:
            print("The initial condition did not end up to be divergence free. "
                  "Please be aware that either there is some inconsistency or the initial density is non-uniform")
    else:
        warnings.warn("You have selected the inputs to be not divergence free. Please re-think")
        exit()

    # make the input values non-dimensional
    prim_var[:, :, 0] = prim_var[:, :, 0] / refVar['rho_ref']  # density
    prim_var[:, :, 1] = prim_var[:, :, 1] / refVar['u_ref']  # u
    prim_var[:, :, 2] = prim_var[:, :, 2] / refVar['v_ref']  # v
    prim_var[:, :, 3] = prim_var[:, :, 3] / refVar['p_ref']  # p
    prim_var[:, :, 4] = prim_var[:, :, 4] / refVar['T_ref']  # T

    # plt.contourf(grid.X, grid.Y, prim_var[:, :, 3])
    # plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.contour3D(grid.X, grid.Y, prim_var[:, :, 1], 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('rho')
    # ax.set_title('3D contour')
    # plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(grid.X, grid.Y, prim_var[:, :, 0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    #
    # sys.exit()

    # Run the solver
    switch = True
    if switch:
        result = CalcSolver(params, refVar, numbers_ND, grid, coefficient_mats, prim_var)
        np.save('prim_var.npy', result.output_var)
    else:
        # Plot the results
        result = np.load('prim_var.npy')
        if result is None:
            print('Solution of the primary variables not computed. Please compute it first')
            exit()
        plot.plot(params, grid, result, var_name='rho', type_plot='Quiver')
