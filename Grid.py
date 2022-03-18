import numpy as np


class GridPar:
    def __init__(self) -> None:
        self.Lxi = None
        self.Leta = None
        self.dXi = None
        self.dEta = None
        self.XI = None
        self.ETA = None
        self.X = None
        self.Y = None


def CartesianGrid(params, refVar):
    grid_par = GridPar()

    grid_par.Lxi = params['Geometry Parameters']['Lxi'] / refVar['X_ref']
    grid_par.Leta = params['Geometry Parameters']['Leta'] / refVar['Y_ref']

    # Create the grid for individual directions
    if params['Geometry Parameters']['XI_Periodic']:
        Xi = np.arange(1, params['Geometry Parameters']['Nxi'] + 1) * \
             params['Geometry Parameters']['Lxi'] / params['Geometry Parameters']['Nxi']
    else:
        Xi = np.linspace(0, params['Geometry Parameters']['Lxi'], params['Geometry Parameters']['Nxi'])
    dXi = Xi[1] - Xi[0]

    dEta = None
    if params['Geometry Parameters']['Neta'] == 1:
        Eta = 0
        dEta = 1
    else:
        if params['Geometry Parameters']['ETA_Periodic']:
            Eta = np.arange(1, params['Geometry Parameters']['Neta'] + 1) * \
                  params['Geometry Parameters']['Leta'] / params['Geometry Parameters']['Neta']
        else:
            Eta = np.linspace(0, params['Geometry Parameters']['Leta'], params['Geometry Parameters']['Neta'])
        dEta = Eta[1] - Eta[0]

    grid_par.dXi = dXi
    grid_par.dEta = dEta

    # Create the mesh grid for future
    XI, ETA = np.meshgrid(Xi, Eta)
    # We want to have Xi changing in the first index and Eta changing in the second index
    grid_par.XI = np.transpose(XI)
    grid_par.ETA = np.transpose(ETA)

    grid_par.X = grid_par.XI
    grid_par.Y = grid_par.ETA

    return grid_par
