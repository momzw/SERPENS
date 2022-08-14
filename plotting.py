import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from rebound.plotting import fading_line
matplotlib.use('TkAgg')



def plotting(sim, density=True, save=True, show=True, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    ps = sim.particles

    # ============================================================================

    # PLOT CENTER AT SUN
    # ------------------
    # a_jup = sim.particles[1].calculate_orbit(primary=sim.particles[0]).a
    # lim = a_jup * 1.5
    # ax.set_xlim([-lim, lim])
    # ax.set_ylim([-lim, lim])

    # PLOT CENTER AT JUPITER
    # ----------------------
    lim = 15 * ps[1].r

    ax.set_xlim([-lim + ps[1].x, lim + ps[1].x])
    ax.set_ylim([-lim + ps[1].y, lim + ps[1].y])

    xlocs = np.linspace(ps[1].x - 12 * ps[1].r, ps[1].x + 12 * ps[1].r, 13)
    ylocs = np.linspace(ps[1].y - 12 * ps[1].r, ps[1].y + 12 * ps[1].r, 13)
    xlabels = np.around((np.array(xlocs) - ps[1].x) / ps[1].r, 2)
    ylabels = np.around((np.array(ylocs) - ps[1].y) / ps[1].r, 2)

    ax.set_xticks(xlocs)
    ax.set_xticklabels([str(x) for x in xlabels])
    ax.set_yticks(ylocs)
    ax.set_yticklabels([str(y) for y in ylabels])

    fig.suptitle("Particle Simulation around Planetary Body", size='x-large', y=.95)
    ax.set_title(f"Number of Particles: {sim.N}", y=.90)
    ax.set_xlabel("x-distance in planetary radii")
    ax.set_ylabel("y-distance in planetary radii")

    # PLOT CENTER AT IO
    # -----------------
    # lim = 1e7
    # ax.set_xlim([-lim + ps[2].x, lim + ps[2].x])
    # ax.set_ylim([-lim + ps[2].y, lim + ps[2].y])

    # ============================================================================

    ax.plot([ps[0].x, ps[2].x], [ps[0].y, ps[2].y], color='k',
            linestyle='--', linewidth=1)

    Io_patch = plt.Circle((ps[2].x, ps[2].y), ps[2].r, fc='k', alpha=.7)
    Jup_patch = plt.Circle((ps[1].x, ps[1].y), ps[1].r, fc='orange')

    ax.add_patch(Io_patch)
    ax.add_patch(Jup_patch)
    ax.scatter(ps[0].x, ps[0].y, s=35, facecolor='yellow', zorder=3)  # Sun
    ax.scatter(ps[1].x, ps[1].y, s=35, facecolor='orange', zorder=3)  # Jupiter
    ax.scatter(ps[2].x, ps[2].y, s=10, facecolor='black', zorder=2)  # Io

    Io = ps[2]
    o = np.array(Io.sample_orbit(primary=sim.particles[1]))
    lc = fading_line(o[:, 0], o[:, 1], alpha=0.5)
    ax.add_collection(lc)

    # xp, yp = ps[1].x, ps[1].y
    # X = []
    # Y = []
    # for k in range(np.size(xedges) - 1):
    #    xval = (xedges[k] + xedges[k + 1]) / 2
    #    x_norm = (xval - xp)/ps[1].r
    #    X.append(xval)
    # for j in range(np.size(yedges) - 1):
    #    yval = (yedges[j] + yedges[j + 1]) / 2
    #    y_norm = (yval - yp)/ps[1].r
    #    Y.append(yval)

    if density:
        if "histogram" in kwargs and "xedges" in kwargs and "yedges" in kwargs:
            H = kwargs.get("histogram")
            xedges = kwargs.get("xedges")
            yedges = kwargs.get("yedges")
            ax.imshow(H, interpolation='gaussian', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Reds', norm=colors.LogNorm())
        else:
            print("Error: Trying to plot density without passing necessary kwargs \"histogram\", \"xedges\", \"yedges\"")
    else:
        for particle in ps[3:]:
            ax.scatter(particle.x, particle.y, s=.2, facecolor='red', alpha=.3)

    i = kwargs.get("iter", 0)
    if save: plt.savefig(f'plots/sim_{i}.png')
    if show: plt.show()

    plt.close()
