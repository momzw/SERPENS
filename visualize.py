import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rebound.plotting import fading_line
from init import Parameters

import scipy.ndimage

class Visualize:

    def __init__(self, rebsim):
        params = Parameters()
        self.ns = params.num_species
        self.sim = rebsim
        self.ps = rebsim.particles
        self.moon = params.int_spec["moon"]

        self.subplot_rows = int(np.ceil(self.ns / 3))
        self.subplot_columns = params.num_species if self.ns <= 3 else 3

        self.fig = plt.figure(figsize=(15, 15))
        gs1 = gridspec.GridSpec(self.subplot_rows, self.subplot_columns)
        gs1.update(wspace=0.12, hspace=0.1)

        self.axs = [plt.subplot(gs1[f]) for f in range(self.subplot_rows * self.subplot_columns)]
        for ax_num in range(len(self.axs)):
            if ax_num >= params.num_species:
                self.axs[ax_num].remove()
            else:
                species_name = params.get_species(ax_num+1).description
                self.axs[ax_num].set_facecolor('k')
                self.axs[ax_num].set_title(f"{species_name}", c='k', size='large')

        if self.moon:
            self.fig.suptitle(
                r"Particle Densities [cm$^{-3}$] around Planetary Body",
                size='xx-large')
            self.boundary = params.int_spec["r_max"] * rebsim.particles["moon"].calculate_orbit(primary=rebsim.particles["planet"]).a
        else:
            self.fig.suptitle(
                r"Particle Densities [cm$^{-3}$] around Stellar Body",
                size='xx-large')
            self.boundary = params.int_spec["r_max"] * rebsim.particles["planet"].a

    def __call__(self, save_path = None, show_bool=True, **kwargs):

        handles, labels = self.axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        self.fig.legend(by_label.values(), by_label.keys(), loc='upper right')
        self.fig.text(0.1, 0.5, "y-distance in primary radii", rotation="vertical", verticalalignment='center',horizontalalignment='right', fontsize='x-large')
        self.fig.text(0.5, 0.05, "x-distance in primary radii", horizontalalignment='center', fontsize='x-large')

        if save_path is not None:
            i = kwargs.get("i", -1)
            if self.moon:
                orbit_phase = np.around(self.sim.particles["moon"].calculate_orbit(
                    primary=self.sim.particles["planet"]).f * 180 / np.pi)
            else:
                orbit_phase = np.around(self.sim.particles["planet"].calculate_orbit(
                    primary=self.sim.particles[0]).f * 180 / np.pi)
            frame_identifier = f"ColumnDensity_TopDown_{i}_{orbit_phase}"
            plt.savefig(f'output/{save_path}/plots/{frame_identifier}.png')
        if show_bool:
            plt.show()
        plt.close()

    def __del__(self):
        plt.clf()

    def setup_ax(self, ax, perspective):
        #if lim is None:
        #    #lim = 35 * self.ps["planet"].r
        #    lim = self.ps[self.sim.N_active - 1].x - self.ps["planet"].x
        #else:
        #    lim *= self.ps["planet"].r

        lim = 35 * self.ps["planet"].r if self.moon else 10 * self.ps[0].r

        if perspective == "topdown":
            ps_star_coord1 = self.ps[0].x
            ps_star_coord2 = self.ps[0].y
            ps_planet_coord1 = self.ps["planet"].x
            ps_planet_coord2 = self.ps["planet"].y

            ax.set_xlabel("x-distance in planetary radii", fontsize='x-large')
            ax.set_ylabel("y-distance in planetary radii", fontsize='x-large')

        elif perspective == "los":
            ps_star_coord1 = self.ps[0].y
            ps_star_coord2 = self.ps[0].z
            ps_planet_coord1 = self.ps["planet"].y
            ps_planet_coord2 = self.ps["planet"].z

            ax.set_xlabel("y-distance in planetary radii", fontsize='x-large')
            ax.set_ylabel("z-distance in planetary radii", fontsize='x-large')

        else:
            raise ValueError("Invalid perspective in plotting.")

        ax.set_xlim([-lim + ps_planet_coord1, lim + ps_planet_coord1]) if self.moon else ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim + ps_planet_coord2, lim + ps_planet_coord2]) if self.moon else ax.set_ylim([-lim, lim])

        if self.moon:
            ps_moon_coord1 = self.ps["moon"].x
            ps_moon_coord2 = self.ps["moon"].y

            xlocs = np.linspace(ps_planet_coord1 - lim, ps_planet_coord1 + lim, 13)
            ylocs = np.linspace(ps_planet_coord2 - lim, ps_planet_coord2 + lim, 13)
            xlabels = np.around((np.array(xlocs) - ps_planet_coord1) / self.ps["planet"].r, 1)
            ylabels = np.around((np.array(ylocs) - ps_planet_coord2) / self.ps["planet"].r, 1)

            moon_patch = plt.Circle((ps_moon_coord1, ps_moon_coord2), self.ps["moon"].r, fc='y', alpha=.7, label="moon")
            ax.add_patch(moon_patch)

            o = np.array(self.ps["moon"].sample_orbit(primary=self.ps["planet"]))

            # Show direction to Sun:
            if perspective == "topdown":
                ax.plot([ps_star_coord1, ps_moon_coord1], [ps_star_coord2, ps_moon_coord2], color='bisque', linestyle=':',linewidth=1, zorder=1)

        else:
            xlocs = np.linspace(-lim, lim, 13)
            ylocs = np.linspace(-lim, lim, 13)
            xlabels = np.around(np.array(xlocs) / self.ps[0].r, 2)
            ylabels = np.around(np.array(ylocs) / self.ps[0].r, 2)

            star_patch = plt.Circle((ps_star_coord1, ps_star_coord2), self.ps[0].r, fc='y', zorder=10, label="star")
            ax.add_patch(star_patch)

            o = np.array(self.ps["planet"].sample_orbit(primary=self.ps[0]))

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        planet_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), self.ps["planet"].r, fc='sandybrown', label="planet", zorder=10)
        ax.add_patch(planet_patch)

        if perspective == "topdown":
            lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color='yellow')
            ax.add_collection(lc)

        if self.moon:
            for i in range(self.sim.N_active - 3):
                if perspective == "topdown":
                    major_obj = plt.Circle((self.ps[3 + i].x, self.ps[3 + i].y), self.ps[3 + i].r, alpha=0.7, fc='y')
                    ax.add_patch(major_obj)
                    o_add = np.array(self.ps[3 + i].sample_orbit(primary=self.ps["planet"]))
                    lc_add = fading_line(o_add[:, 0], o_add[:, 1], alpha=0.5, color='yellow')
                    ax.add_collection(lc_add)
                elif perspective == "los":
                    major_obj = plt.Circle((self.ps[3 + i].y, self.ps[3 + i].z), self.ps[3 + i].r, alpha=0.7, fc='y')
                    ax.add_patch(major_obj)
        else:
            for i in range(self.sim.N_active - 2):
                if perspective == "topdown":
                    major_obj = plt.Circle((self.ps[3 + i].x, self.ps[3 + i].y), self.ps[3 + i].r, alpha=0.7, fc='y')
                    ax.add_patch(major_obj)
                elif perspective == "los":
                    major_obj = plt.Circle((self.ps[3 + i].y, self.ps[3 + i].z), self.ps[3 + i].r, alpha=0.7, fc='y')
                    ax.add_patch(major_obj)

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', visible=True)
        #if k == 0:
        #    plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=True)
        #else:
        #    plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=False)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')

    def add_histogram(self, ax, H, xedges, yedges, perspective, **kwargs):
        ax = self.axs[ax]
        self.setup_ax(ax, perspective=perspective)
        norm_min = kwargs.get("norm_min", None)
        if norm_min is not None:
            norm = colors.LogNorm(vmin=norm_min) if not np.max(H) == 0 else colors.Normalize(vmin=0, vmax=0)  # Not needed if first sim_instance is already with particles.
        else:
            norm = colors.LogNorm() if not np.max(H) == 0 else colors.Normalize(vmin=0, vmax=0)
        cmap = matplotlib.cm.afmhot
        cmap.set_bad('k', 1.)
        im = ax.imshow(H, interpolation='gaussian', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, norm=norm)

    def add_dtfe(self, ax, x, y, dtfe, perspective, clim=None, cb_format='%.2f'):
        ax = self.axs[ax]
        self.setup_ax(ax, perspective=perspective)
        scatter = ax.scatter(x, y, c=dtfe, cmap=plt.cm.afmhot, marker='.', norm=colors.LogNorm(), s=1.5, zorder=5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #cax.tick_params(axis='both', which='major', labelsize=8)
        plt.colorbar(scatter, cax=cax, orientation='vertical', format=cb_format)
        if clim is not None:
            plt.clim(clim[0], clim[1])

    def add_triplot(self, ax, x, y, simplices, perspective):
        ax = self.axs[ax]
        self.setup_ax(ax, perspective=perspective)
        ax.triplot(x, y, simplices, linewidth=0.1, c='w')

    def add_contour(self, ax, x, y, z, perspective):
        ax = self.axs[ax]
        self.setup_ax(ax, perspective=perspective)
        #z = np.clip(z, a_min=1e-30, a_max=None)
        #z = scipy.ndimage.filters.gaussian_filter(z, sigma=z.std())
        ax.contour(x, y, z, colors='w', norm=colors.LogNorm(), zorder=6)
        #ax.contourf(x, y, z, cmap=matplotlib.cm.afmhot, norm=colors.LogNorm(), zorder=4)











