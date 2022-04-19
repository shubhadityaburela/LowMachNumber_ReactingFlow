import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.gridspec as gridspec


def plot(params, grid, prim_var, var_name='u', type_plot='2D'):
    # Choose the variable to plot
    if var_name == 'rho':
        k = 0
    elif var_name == 'u':
        k = 1
    elif var_name == 'v':
        k = 2
    elif var_name == 'p':
        k = 3
    elif var_name == 'T':
        k = 4
    else:
        k = 1

    if type_plot == '2D':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '2%', '2%')

        cv0 = prim_var[0][:, :, k]
        cf = ax.contourf(grid.XI, grid.ETA, cv0)
        cb = fig.colorbar(cf, cax=cax)
        tx = ax.set_title('Time step: 0')
    elif type_plot == '3D':
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    elif type_plot == 'Streamline':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        cu0 = prim_var[0][:, :, 1]
        cv0 = prim_var[0][:, :, 2]
        cf = ax.streamplot(grid.ETA_Vec, grid.XI_Vec, cu0, cv0, color=cu0, linewidth=1, cmap='autumn')
        tx = ax.set_title('Time step: 0')
    elif type_plot == 'Quiver':
        fig = plt.figure()
        ax = fig.add_subplot(111)

        z = grid.XI * np.exp(-grid.XI ** 2 - grid.ETA ** 2)
        dx, dy = np.gradient(z)
        n = -2
        # Defining color
        color = np.sqrt(((dx - n) / 2) * 2 + ((dy - n) / 2) * 2)

        cu0 = prim_var[0][:, :, 1]
        cv0 = prim_var[0][:, :, 2]

        M = np.hypot(cu0, cv0)
        cf = ax.quiver(grid.XI, grid.ETA, cu0, cv0, color)
        tx = ax.set_title('Time step: 0')

    def animate(i):
        if type_plot == '2D':
            vmax = np.max(prim_var[i][:, :, k])
            vmin = np.min(prim_var[i][:, :, k])
            # levels = np.linspace(vmin, vmax, 100, endpoint=True)
            h = ax.contourf(grid.XI, grid.ETA, prim_var[i][:, :, k], vmax=vmax, vmin=vmin)  # , levels=levels)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            cax.cla()
            fig.colorbar(h, cax=cax)
            tx.set_text('Time step {0}'.format(i))
        elif type_plot == '3D':
            ax.cla()
            vmax = np.max(prim_var[i][:, :, k])
            vmin = np.min(prim_var[i][:, :, k])
            print(vmax, vmin)
            h = ax.plot_surface(grid.XI, grid.ETA, prim_var[i][:, :, k], cmap=cm.coolwarm, linewidth=0,
                                antialiased=False)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlim(vmin, vmax)
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.set_title(var_name)
        elif type_plot == 'Streamline':
            ax.cla()
            h = ax.streamplot(grid.ETA_Vec, grid.XI_Vec, prim_var[i][:, :, 1], prim_var[i][:, :, 2],
                              color=prim_var[i][:, :, 1], linewidth=1, cmap='autumn')
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            tx.set_text('Time step {0}'.format(i))
        elif type_plot == 'Quiver':
            ax.cla()
            h = ax.quiver(grid.XI, grid.ETA, prim_var[i][:, :, 1], prim_var[i][:, :, 2], color)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            tx.set_text('Time step {0}'.format(i))

    ani = animation.FuncAnimation(fig, animate, frames=params['Time Parameters']['TimeSteps'],
                                  interval=1, blit=False, repeat=False)
    plt.show()
