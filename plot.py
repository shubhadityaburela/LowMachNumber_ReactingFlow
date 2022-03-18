import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def show_animation(q, Xgrid=None, cycles=1, frequency = 1, figure_number = None, vmin = None,vmax=None,
                   save_path=None, use_html=True):

    ndim = np.size(Xgrid)
    ntime = q.shape[-1]
    if figure_number == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(num=figure_number)

    if Xgrid is not None:

        if vmin is None:
            vmin = np.min(q)
        if vmax is None:
            vmax = np.max(q)

        h = ax.pcolormesh(Xgrid[0], Xgrid[1], q[..., 0])
        h.set_clim(vmin, vmax)
        fig.colorbar(h)
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        def init():
            h.set_array(q[:-1, :-1, 0].ravel())
            return (h,)

        def animate(t):
            h.set_array(q[:-1,:-1,int(t*frequency)].ravel())
            if save_path is not None:
                fig.savefig(save_path+"/vid_%3.3d.png" % t)
            return (h,)

        if use_html:
            anim = animation.FuncAnimation(fig, animate, init_func=init,
                                           frames=ntime//frequency, repeat = cycles, interval=20, blit=True)
            return anim
        else:
            init()
            for t in range(0, cycles * ntime, frequency):
                animate(t)
                plt.draw()
                plt.pause(0.05)
    else:
        x = np.arange(0,np.size(q,0))
        h, = ax.plot(x,q[:, 0])
        ax.set_ylim(np.min(q),np.max(q))
        ax.set_xlabel(r"$x$")
        for t in range(0, cycles * ntime, frequency):
            h.set_data(x,q[:,t % ntime])
            if save_path is not None:
                fig.savefig(save_path+"/vid_%3.3d.png" % t)
            plt.draw()
            plt.pause(0.05)


def plot(params, grid, prim_var, var_name='u'):

    def animate(i):
        ax.clear()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(var_name)
        if var_name == 'rho':
            k = 0
        elif var_name == 'u':
            k = 1
        elif var_name == 'v':
            k = 2
        elif var_name == 'p':
            k = 3
        else:
            k = 1
        h = ax.contourf(grid.XI, grid.ETA, prim_var[i][:, :, k])

    fig = plt.figure()
    ax = fig.gca()
    interval = 1
    ani = animation.FuncAnimation(fig, animate, frames=params['Time Parameters']['TimeSteps'],
                                  interval=interval, blit=False, repeat=False)
    plt.show()
