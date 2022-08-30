import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import rebound
from rebound.plotting import fading_line
matplotlib.use('TkAgg')



def plotting(sim, density=True, save=True, show=True, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    ps = sim.particles

    ax.scatter(ps[0].x, ps[0].y, s=35, facecolor='yellow', zorder=3)  # Star
    ax.scatter(ps["planet"].x, ps["planet"].y, s=35, facecolor='sandybrown', zorder=3)  # Planet

    try:
        sim.particles["moon"]
    except rebound.ParticleNotFound:
        moon_exists = False
    else:
        moon_exists = True

    # ============================================================================
    if moon_exists:
        # ORIGIN AT PLANET
        # ----------------------
        lim = 15 * ps["planet"].r

        ax.set_xlim([-lim + ps["planet"].x, lim + ps["planet"].x])
        ax.set_ylim([-lim + ps["planet"].y, lim + ps["planet"].y])

        xlocs = np.linspace(ps["planet"].x - 12 * ps["planet"].r, ps["planet"].x + 12 * ps["planet"].r, 13)
        ylocs = np.linspace(ps["planet"].y - 12 * ps["planet"].r, ps["planet"].y + 12 * ps["planet"].r, 13)
        xlabels = np.around((np.array(xlocs) - ps["planet"].x) / ps["planet"].r, 2)
        ylabels = np.around((np.array(ylocs) - ps["planet"].y) / ps["planet"].r, 2)

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        fig.suptitle("Particle Simulation around Planetary Body", size='x-large', y=.95)
        ax.set_title(f"Number of Particles: {sim.N}", y=.90, c='w')
        ax.set_xlabel("x-distance in planetary radii")
        ax.set_ylabel("y-distance in planetary radii")

        # Show direction to Sun:
        ax.plot([ps["sun"].x, ps["moon"].x], [ps["sun"].y, ps["moon"].y], color='bisque',
                linestyle=':', linewidth=1, zorder=1)

        moon_patch = plt.Circle((ps["moon"].x, ps["moon"].y), ps["moon"].r, fc='y', alpha=.7)
        planet_patch = plt.Circle((ps["planet"].x, ps["planet"].y), ps["planet"].r, fc='sandybrown')

        ax.add_patch(moon_patch)
        ax.add_patch(planet_patch)

        ax.scatter(ps["moon"].x, ps["moon"].y, s=10, facecolor='y', zorder=2)

        Io = ps["moon"]
        o = np.array(Io.sample_orbit(primary=sim.particles["planet"]))
        lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color='yellow')
        ax.add_collection(lc)

        for i in range(sim.N_active - 3):
            major_obj = plt.Circle((ps[3 + i].x, ps[3 + i].y), ps[3 + i].r, fc='r', alpha=0.7)
            ax.add_patch(major_obj)
            ax.scatter(ps[3 + i].x, ps[3 + i].y, s=10, fc='r', zorder=2)
            o_add = np.array(ps[3 + i].sample_orbit(primary=ps["planet"]))
            lc_add = fading_line(o_add[:, 0], o_add[:, 1], alpha=0.5, color='red')
            ax.add_collection(lc_add)

    else:
        # ORIGIN AT STAR
        # ------------------
        lim = 15 * ps[0].r

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])

        xlocs = np.linspace(-12 * ps[0].r, 12 * ps[0].r, 13)
        ylocs = np.linspace(-12 * ps[0].r, 12 * ps[0].r, 13)
        xlabels = np.around(np.array(xlocs) / ps[0].r, 2)
        ylabels = np.around(np.array(ylocs)  / ps[0].r, 2)

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        fig.suptitle("Particle Simulation around Stellar Body", size='x-large', y=.95)
        ax.set_title(f"Number of Particles: {sim.N}", y=.90, c='w')
        ax.set_xlabel("x-distance in stellar radii")
        ax.set_ylabel("y-distance in stellar radii")

        planet_patch = plt.Circle((ps["planet"].x, ps["planet"].y), ps["planet"].r, fc='sandybrown')
        star_patch = plt.Circle((ps[0].x, ps[0].y), ps[0].r, fc='y', zorder=4)
        ax.add_patch(planet_patch)
        ax.add_patch(star_patch)

        o = np.array(ps["planet"].sample_orbit(primary=sim.particles[0]))
        lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color='yellow')
        ax.add_collection(lc)

        for i in range(sim.N_active - 2):
            major_obj = plt.Circle((ps[2 + i].x, ps[2 + i].y), ps[2 + i].r, fc='r', alpha=0.7)
            ax.add_patch(major_obj)
            ax.scatter(ps[2 + i].x, ps[2 + i].y, s=10, fc='r', zorder=2)

    # PLOT CENTER AT MOON
    # -----------------
    # lim = 1e7
    # ax.set_xlim([-lim + ps[2].x, lim + ps[2].x])
    # ax.set_ylim([-lim + ps[2].y, lim + ps[2].y])

    # ============================================================================

    ax.set_facecolor('k')

    if density:
        if "histogram" in kwargs and "xedges" in kwargs and "yedges" in kwargs:
            H = kwargs.get("histogram")
            xedges = kwargs.get("xedges")
            yedges = kwargs.get("yedges")
            norm = colors.LogNorm() if not np.max(H) == 0 else colors.Normalize(vmin = 0, vmax = 0) # Not needed if first sim_instance is already with particles.
            cmap = matplotlib.cm.afmhot
            cmap.set_bad('k', 1.)
            ax.imshow(H, interpolation='gaussian', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)
        else:
            print("Error: Trying to plot density without passing necessary kwargs \"histogram\", \"xedges\", \"yedges\"")
    else:
        for particle in ps[sim.N_active:]:
            ax.scatter(particle.x, particle.y, s=.2, facecolor='red', alpha=.3)

    i = kwargs.get("iter", 0)
    if save: plt.savefig(f'plots/sim_{i}.png')
    if show: plt.show()

    plt.close()
