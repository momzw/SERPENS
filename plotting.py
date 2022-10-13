import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import rebound
from rebound.plotting import fading_line
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('TkAgg')


def plotting(fig, ax, sim, density=True, plane="xy", **kwargs):
    ax.set_aspect("equal")

    ps = sim.particles

    try:
        sim.particles["moon"]
    except rebound.ParticleNotFound:
        moon_exists = False
    else:
        moon_exists = True

    if plane == "xy":
        ps_star_coord1 = ps[0].x
        ps_star_coord2 = ps[0].y
        ps_planet_coord1 = ps["planet"].x
        ps_planet_coord2 = ps["planet"].y
        if moon_exists:
            ps_moon_coord1 = ps["moon"].x
            ps_moon_coord2 = ps["moon"].y
    elif plane == "yz":
        ps_star_coord1 = ps[0].y
        ps_star_coord2 = ps[0].z
        ps_planet_coord1 = ps["planet"].y
        ps_planet_coord2 = ps["planet"].z
        if moon_exists:
            ps_moon_coord1 = ps["moon"].y
            ps_moon_coord2 = ps["moon"].z
    else:
        raise ValueError('Invalid plane in plotting.')

    ax.scatter(ps_star_coord1, ps_star_coord2, s=35, facecolor='yellow', zorder=3)  # Star
    ax.scatter(ps_planet_coord1, ps_planet_coord2, s=35, facecolor='sandybrown', zorder=3)  # Planet

    # ============================================================================
    if moon_exists:
        # ORIGIN AT PLANET
        # ----------------------
        lim = 12 * ps["planet"].r

        ax.set_xlim([-lim + ps_planet_coord1, lim + ps_planet_coord1])
        ax.set_ylim([-lim + ps_planet_coord2, lim + ps_planet_coord2])

        xlocs = np.linspace(ps_planet_coord1 - lim, ps_planet_coord1 + lim, 13)
        ylocs = np.linspace(ps_planet_coord2 - lim, ps_planet_coord2 + lim, 13)
        xlabels = np.around((np.array(xlocs) - ps_planet_coord1) / ps["planet"].r, 2)
        ylabels = np.around((np.array(ylocs) - ps_planet_coord2) / ps["planet"].r, 2)

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        if plane=='xy':
            ax.set_xlabel("x-distance in planetary radii", fontsize='x-large')
            ax.set_ylabel("y-distance in planetary radii", fontsize='x-large')
        elif plane=='yz':
            ax.set_xlabel("y-distance in planetary radii", fontsize='x-large')
            ax.set_ylabel("z-distance in planetary radii", fontsize='x-large')
            ax.set_ylim([-lim/2 + ps_planet_coord2, lim/2 + ps_planet_coord2])

        moon_patch = plt.Circle((ps_moon_coord1, ps_moon_coord2), ps["moon"].r, fc='y', alpha=.7, label="moon")
        planet_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), ps["planet"].r, fc='sandybrown', label="planet")

        ax.add_patch(moon_patch)
        ax.add_patch(planet_patch)

        ax.scatter(ps_moon_coord1, ps_moon_coord2, s=10, facecolor='y', zorder=2)

        if plane=='xy':
            Io = ps["moon"]
            o = np.array(Io.sample_orbit(primary=sim.particles["planet"]))
            lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color='yellow')
            ax.add_collection(lc)

            # Show direction to Sun:
            ax.plot([ps_star_coord1, ps_moon_coord1], [ps_star_coord2, ps_moon_coord2], color='bisque',
                    linestyle=':', linewidth=1, zorder=1)

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
        lim = 5 * ps[0].r

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])

        xlocs = np.linspace(-lim, lim, 13)
        ylocs = np.linspace(-lim, lim, 13)
        xlabels = np.around(np.array(xlocs) / ps[0].r, 2)
        ylabels = np.around(np.array(ylocs)  / ps[0].r, 2)

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        if plane=='xy':
            ax.set_xlabel("x-distance in stellar radii", fontsize='x-large')
            ax.set_ylabel("y-distance in stellar radii", fontsize='x-large')
        elif plane=='yz':
            ax.set_xlabel("y-distance in stellar radii", fontsize='x-large')
            ax.set_ylabel("z-distance in stellar radii", fontsize='x-large')

        planet_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), ps["planet"].r, fc='sandybrown', label="planet")
        star_patch = plt.Circle((ps_star_coord1, ps_star_coord2), ps[0].r, fc='y', zorder=4, label="star")
        ax.add_patch(planet_patch)
        ax.add_patch(star_patch)

        if plane=='xy':
            o = np.array(ps["planet"].sample_orbit(primary=sim.particles[0]))
            lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color='yellow')
            ax.add_collection(lc)

        for i in range(sim.N_active - 2):
            if plane=="xy":
                major_obj = plt.Circle((ps[2 + i].x, ps[2 + i].y), ps[2 + i].r, fc='r', alpha=0.7)
                ax.scatter(ps[2 + i].x, ps[2 + i].y, s=10, fc='r', zorder=2)
            elif plane=="yz":
                major_obj = plt.Circle((ps[2 + i].y, ps[2 + i].z), ps[2 + i].r, fc='r', alpha=0.7)
                ax.scatter(ps[2 + i].y, ps[2 + i].z, s=10, fc='r', zorder=2)
            ax.add_patch(major_obj)

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
            im = ax.imshow(H, interpolation='gaussian', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)

            if not np.max(H) == 0:
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

        else:
            print("Error: Trying to plot density without passing necessary kwargs \"histogram\", \"xedges\", \"yedges\"")
    else:
        for particle in ps[sim.N_active:]:
            if plane=="xy":
                ax.scatter(particle.x, particle.y, s=.2, facecolor='white', alpha=.3)
            elif plane=="yz":
                ax.scatter(particle.y, particle.z, s=.2, facecolor='white', alpha=.3)

    return ax

