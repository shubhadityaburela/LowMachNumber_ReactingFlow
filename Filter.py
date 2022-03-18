import numpy as np
from scipy.sparse import diags
from scipy import sparse

from Mat import Mat


class Filter:
    def __init__(self, fp, gp, ep) -> None:
        self.__Type = fp['Type']
        self.__order = fp['order']
        self.__alpha = fp['alpha']
        self.__name = fp['name']

        self.__number = ep['number']

        self.__Nxi = gp['Nxi']
        self.__Neta = gp['Neta']

    def initPadeFilter(self):
        A_Xi, B_Xi = self.padeFilterMatrices(self.__Nxi, self.__Type, self.__order, self.__alpha)
        A_Eta, B_Eta = self.padeFilterMatrices(self.__Neta, self.__Type, self.__order, self.__alpha)

        return A_Xi, B_Xi, A_Eta, B_Eta

    @staticmethod
    def padeFilterMatrices(N, Type='non-per', order=4, alpha=0.35):
        # Choice of alpha : alpha = 0 --> explicit filter
        #                   alpha = 0.5 --> no filtering
        # Bu' = Au --> u' = B\Au
        # Check the alpha values
        if alpha < -0.5 or alpha > 0.5:
            print("error in Pade filter: non suitable choice of alpha")
            exit()

        # Determine size and reshape to vector
        # B == matrix
        if Type == 'non-per':
            d = np.ones(N)
            s = np.ones(N - 1) * alpha
            # Build matrix (sparse)
            offset = [-1, 0, 1]
            B = diags([s, d, s], offset).toarray()
            B[0, 1] = 0
            B[-1, -2] = 0
        elif Type == 'per':
            d = np.ones(N)
            s = np.ones(N - 1) * alpha
            # Build matrix sparse
            offset = [-1, 0, 1]
            B = diags([s, d, s], offset).toarray()
            B[-1, 0] = alpha
            B[0, -1] = alpha
        else:
            raise NotImplemented("Other filter types are not implemented. Please select the one from the list")

        # A == matrix
        # coefficients integer
        a = np.zeros(6)
        b = np.zeros((5, 11))
        if order == 2:
            a[1] = +1 / 2 + alpha
            a[2] = +1 / 2 + alpha
            a_stencil = 1  # stencil width
        if order == 4:
            a[1] = +5 / 8 + 3 * alpha / 4
            a[2] = +1 / 2 + alpha
            a[3] = -1 / 8 + alpha / 4
            a_stencil = 2  # stencil width

            # point 2('non-per' case)
            b[1, 1] = +1 / 16 + 7 * alpha / 8
            b[1, 2] = +3 / 4 + alpha / 2
            b[1, 3] = +3 / 8 + alpha / 4
            b[1, 4] = -1 / 4 + alpha / 2
            b[1, 5] = +1 / 16 - alpha / 8
        if order == 6:
            a[1] = +11 / 16 + 5 * alpha / 8
            a[2] = +15 / 32 + 17 * alpha / 16
            a[3] = -3 / 16 + 3 * alpha / 8
            a[4] = +1 / 32 - alpha / 16
            a_stencil = 3  # stencil width

            # point 2('non-per' case)
            b[1, 1] = 1 / 64 + 31 * alpha / 32
            b[1, 2] = 29 / 32 + 3 * alpha / 16
            b[1, 3] = 15 / 64 + 17 * alpha / 32
            b[1, 4] = -5 / 16 + 5 * alpha / 8
            b[1, 5] = 15 / 64 - 15 * alpha / 32
            b[1, 6] = -3 / 32 + 3 * alpha / 16
            b[1, 7] = 1 / 64 - 1 * alpha / 32

            # point 3('non-per' case)
            b[2, 1] = -1 / 64 + 1 * alpha / 32
            b[2, 2] = 3 / 32 + 13 * alpha / 16
            b[2, 3] = 49 / 64 + 15 * alpha / 32
            b[2, 4] = 5 / 16 + 3 * alpha / 8
            b[2, 5] = -15 / 64 + 15 * alpha / 32
            b[2, 6] = 3 / 32 - 3 * alpha / 16
            b[2, 7] = -1 / 64 + 1 * alpha / 32
        else:
            raise NotImplemented("Please specify the proper order of the filter from the list")

        # A -- matrix
        a = 0.5 * a
        if Type == 'non-per':
            # build sparse matrix
            A = sparse.csc_matrix(np.zeros((N, N), dtype=float))
            for i in range(a.size()):
                A = A + a[i] * sparse.csc_matrix(np.diag(np.ones(N - abs(i)), -i))
            A = A + A.transpose()

            # Bounds not to be modified
            A[0, :] = 0
            A[N - 1, :] = 0
            A[0, 0] = 1
            A[N - 1, N - 1] = 1

            for i in range(a_stencil - 2):
                A[i + 1, 0:a_stencil * 2] = b[i, 0:a_stencil * 2]
                A[N - 1 - i, N - 1 - a_stencil * 2:N - 1] = np.fliplr(b[i, 0:a_stencil * 2])
        elif Type == 'per':
            e = np.ones(N)

            # build sparse matrix
            A = sparse.csc_matrix(np.zeros((N, N), dtype=float))
            for i in range(a.size() - 1):
                A = A + a[i + 1] * sparse.csc_matrix(np.diag(np.ones(N - abs(i + 1)), i + 1))
            for i in range(a.size() - 1):
                A = A + a[i + 1] * sparse.csc_matrix(np.diag(np.ones(i + 1), i - N))
            A = A + A.transpose() + 2 * a[0] * sparse.csc_matrix(np.diag(np.ones(N), 0))
        else:
            raise NotImplemented("Other filter types are not implemented. Please select the one from the list")

        return A, B

    def filterWrapper(self, values):
        values_tilde = values

        if self.__name == 'explicit':
            filterEqns = self.__number
            for eqnNo in range(filterEqns):
                values_tilde[:, :, eqnNo] = self.explicitFilter(values_tilde[:, :, eqnNo])
        else:
            raise NotImplemented("Other filters not implemented. Please select 'explicit'")

        return values_tilde

    def explicitFilter(self, u):
        if self.__order == 2:
            phis = np.array([-1/2, 1/4])
        elif self.__order == 4:
            phis = np.array([-3/8, 1/4, -1/16])
        elif self.__order == 6:
            phis = np.array([-5/16, 15/64, -3/32, 1/64])
        elif self.__order == 8:
            phis = np.array([-35/128, 7/32, -7/64, 1/32, -1/256])
        elif self.__order == 10:
            phis = np.array([-63/256, 105/512, -15/128, 45/1024, -5/512, 1/1024])
        elif self.__order == 12:
            phis = np.array([-231/1024, 99/512, -495/4096, 55/1024, -33/2048, 3/1024, -1/4096])
        else:
            raise NotImplemented("Please select the order of the filters from the list given")

        filterVals = [np.flipud(phis)[0:-2], phis[0:-1]]

        mid = len(phis)
        filterVals[mid] = filterVals[mid] + 1

        M = Mat()
        Fxi = M.D_periodic(filterVals, self.__Nxi, 1)
        Feta = M.D_periodic(filterVals, self.__Neta, 1)

        u = Fxi * u * np.transpose(Feta)

        return u
