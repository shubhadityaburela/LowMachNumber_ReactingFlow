import numpy as np


class Scheme:
    def __init__(self):
        self.name = None
        self.a = None
        self.b = None
        self.c = None
        self.implicit = None


def RungeKutta(scheme_name):
    scheme = Scheme()

    if scheme_name == 'gauss2':
        scheme.name = 'gauss2'
        scheme.a = np.array([1/2])
        scheme.b = np.array([1])
        scheme.c = np.array([1/2])
        scheme.implicit = True
    elif scheme_name == 'gauss4':
        scheme.name = 'gauss4'
        scheme.a = np.array([[1/4, 1/4-np.sqrt(3)/6],
                             [1/4+np.sqrt(3)/6, 1/4]])
        scheme.b = np.array([1/2, 1/2])
        scheme.c = np.array([1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6])
        scheme.implicit = True
    elif scheme_name == 'gauss6':
        scheme.name = 'gauss6'
        scheme.a = np.array([[5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
                             [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
                             [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]])
        scheme.b = np.array([5 / 18, 4 / 9, 5 / 18])
        scheme.c = np.array([1 / 2 - np.sqrt(15) / 10, 1 / 2, 1 / 2 + np.sqrt(15) / 10])
        scheme.implicit = True
    elif scheme_name == 'classicRK4':
        scheme.name = 'classicRK4'
        scheme.a = np.array([[0, 0, 0, 0],
                             [1 / 2, 0, 0, 0],
                             [0, 1 / 2, 0, 0],
                             [0, 0, 1, 0]])
        scheme.b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        scheme.c = np.array([0, 1 / 2, 1 / 2, 1])
        scheme.implicit = False
    elif scheme_name == 'EulerExplicit':
        scheme.name = 'EulerExplicit'
        scheme.a = np.array([0])
        scheme.b = np.array([1])
        scheme.c = np.array([0])
        scheme.implicit = False
    else:
        raise NotImplemented("Please select one of the options from the list for the Runge-Kutta time integration")

    return scheme




