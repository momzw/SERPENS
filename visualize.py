import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rebound.plotting import fading_line
from init import Parameters
from scipy.ndimage.filters import gaussian_filter
from matplotlib.widgets import Slider, RangeSlider
matplotlib.use('TkAgg')

class Visualize:

    # TODO (FOR LATER): Check out VISPY for fast interactive plots running over GPU

    def __init__(self, rebsim, interactive=True, cmap=plt.cm.afmhot, lim=35):
        params = Parameters()
        self.ns = params.num_species
        self.sim = rebsim
        self.ps = rebsim.particles
        self.moon = params.int_spec["moon"]

        self.subplot_rows = int(np.ceil(self.ns / 3))
        self.subplot_columns = params.num_species if self.ns <= 3 else 3

        self.fig = plt.figure(figsize=(15, 15))
        gs1 = gridspec.GridSpec(self.subplot_rows, self.subplot_columns)
        gs1.update(wspace=0.2, hspace=0.1)

        self.axs = [plt.subplot(gs1[f]) for f in range(self.subplot_rows * self.subplot_columns)]
        for ax_num in range(len(self.axs)):
            if ax_num >= params.num_species:
                self.axs[ax_num].remove()
            else:
                species_name = params.get_species(num=ax_num + 1).description
                self.axs[ax_num].set_facecolor('k')
                self.axs[ax_num].set_title(f"{species_name}", c='k', size='large')

        if self.moon:
            self.fig.suptitle("Serpens Simulation around Planetary Body")
            self.boundary = params.int_spec["r_max"] * rebsim.particles["moon"].calculate_orbit(
                primary=rebsim.particles["planet"]).a
        else:
            self.fig.suptitle("Serpens Simulation around Stellar Body")
            self.boundary = params.int_spec["r_max"] * rebsim.particles["planet"].a

        self.cmap = cmap
        self.cmap.set_bad(color='k', alpha=1.)
        self.lim = lim

        self.cf = None
        self.c = None
        self.scatter = None
        self.cb_interact = None
        self.interactive = interactive

    def __call__(self, save_path=None, show_bool=True, **kwargs):

        handles, labels = self.axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        self.fig.legend(by_label.values(), by_label.keys())
        self.fig.text(0.08, 0.5, "y-distance in primary radii", rotation="vertical", verticalalignment='center',
                      horizontalalignment='right', fontsize='x-large')
        self.fig.text(0.5, 0.05, "x-distance in primary radii", horizontalalignment='center', fontsize='x-large')

        if save_path is not None:
            fn = kwargs.get("filename", -1)
            if self.moon:
                orbit_phase = np.around(self.sim.particles["moon"].calculate_orbit(
                    primary=self.sim.particles["planet"]).f * 180 / np.pi)
            else:
                orbit_phase = np.around(self.sim.particles["planet"].calculate_orbit(
                    primary=self.sim.particles[0]).f * 180 / np.pi)
            frame_identifier = f"ColumnDensity_TopDown_{fn}_{orbit_phase}"
            plt.savefig(f'output/{save_path}/plots/{frame_identifier}.png')

        if show_bool:

            if self.cf is None and self.c is None and self.scatter is None:
                self.interactive = False

            if self.interactive and len(self.axs) == 1:

                slider_ax = self.fig.add_axes([0.87, 0.1, 0.03, 0.6])

                if self.cf is not None:
                    slider = RangeSlider(slider_ax, "Threshold", self.cf.norm.vmin, self.cf.norm.vmax,
                                         orientation='vertical', facecolor='crimson')
                elif self.c is not None:
                    slider = RangeSlider(slider_ax, "Threshold", self.c.norm.vmin, self.c.norm.vmax,
                                         orientation='vertical', facecolor='crimson')
                else:
                    slider = RangeSlider(slider_ax, "Threshold", self.scatter.norm.vmin, self.scatter.norm.vmax,
                                         orientation='vertical', facecolor='crimson')

                if self.cf or self.c is not None:
                    axfreq = self.fig.add_axes([0.92, 0.1, 0.03, 0.6])
                    smoothing_slider = Slider(ax=axfreq, label='Smoothing', valmin=0.1, valmax=5, valinit=1,
                                              orientation='vertical', facecolor='crimson')
                    smoothing_slider.on_changed(
                        lambda update: self.__update_interactive(update, slider, smoothing_slider))
                    slider.on_changed(lambda update: self.__update_interactive(update, slider, smoothing_slider))
                else:
                    slider.on_changed(lambda update: self.__update_interactive(update, slider))

                plt.show()
            else:
                plt.show()
        plt.close()

    def __del__(self):
        plt.clf()
        plt.close()

    def set_title(self, title_string, size='xx-large'):
        self.fig.suptitle(title_string, size=size)

    def __setup_ax(self, ax, perspective, celest_colors='default'):
        # if lim is None:
        #    #lim = 35 * self.ps["planet"].r
        #    lim = self.ps[self.sim.N_active - 1].x - self.ps["planet"].x
        # else:
        #    lim *= self.ps["planet"].r

        ax.set_aspect("equal")
        lim = self.lim * self.ps["planet"].r if self.moon else self.lim * self.ps[0].r

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

        if celest_colors == 'default':
            fc = ['y', 'sandybrown', 'y']
        else:
            if isinstance(celest_colors, (list, np.ndarray)):
                fc = celest_colors
            else:
                fc = ['y', 'sandybrown', 'y']

        if self.moon:
            ps_moon_coord1 = self.ps["moon"].x
            ps_moon_coord2 = self.ps["moon"].y

            xlocs = np.linspace(ps_planet_coord1 - lim, ps_planet_coord1 + lim, 13)
            ylocs = np.linspace(ps_planet_coord2 - lim, ps_planet_coord2 + lim, 13)
            xlabels = np.around((np.array(xlocs) - ps_planet_coord1) / self.ps["planet"].r, 1)
            ylabels = np.around((np.array(ylocs) - ps_planet_coord2) / self.ps["planet"].r, 1)

            moon_patch = plt.Circle((ps_moon_coord1, ps_moon_coord2), self.ps["moon"].r, fc=fc[2], alpha=.7, label="moon", zorder=10)
            ax.add_patch(moon_patch)

            o = np.array(self.ps["moon"].sample_orbit(primary=self.ps["planet"]))

            # Show direction to Sun:
            if perspective == "topdown":
                ax.plot([ps_star_coord1, ps_moon_coord1], [ps_star_coord2, ps_moon_coord2], color='bisque',
                        linestyle=':', linewidth=1, zorder=10)

        else:
            xlocs = np.linspace(-lim, lim, 13)
            ylocs = np.linspace(-lim, lim, 13)
            xlabels = np.around(np.array(xlocs) / self.ps[0].r, 2)
            ylabels = np.around(np.array(ylocs) / self.ps[0].r, 2)

            star_patch = plt.Circle((ps_star_coord1, ps_star_coord2), self.ps[0].r, fc=fc[0], zorder=10, label="star")
            ax.add_patch(star_patch)

            o = np.array(self.ps["planet"].sample_orbit(primary=self.ps[0]))

        ax.set_xticks(xlocs)
        ax.set_xticklabels([str(x) for x in xlabels])
        ax.set_yticks(ylocs)
        ax.set_yticklabels([str(y) for y in ylabels])

        planet_patch = plt.Circle((ps_planet_coord1, ps_planet_coord2), self.ps["planet"].r, fc=fc[0],
                                  label="planet", zorder=10)
        ax.add_patch(planet_patch)

        if perspective == "topdown":
            line_color = fc[2] if self.moon else fc[1]
            line_color = colors.to_rgba(line_color)
            lc = fading_line(o[:, 0], o[:, 1], alpha=0.5, color=line_color[:3], zorder=10)
            ax.add_collection(lc)

        # Additional celestial objects
        if self.moon:
            for i in range(self.sim.N_active - 3):

                fc = 'y'
                try:
                    fc = fc[3+i]
                    if not isinstance(fc, str):
                        fc = 'y'
                except:
                    pass

                if perspective == "topdown":
                    major_obj = plt.Circle((self.ps[3 + i].x, self.ps[3 + i].y), self.ps[3 + i].r, alpha=0.7, fc=fc, zorder=10)
                    ax.add_patch(major_obj)
                    o_add = np.array(self.ps[3 + i].sample_orbit(primary=self.ps["planet"]))
                    color = colors.to_rgba(fc)
                    lc_add = fading_line(o_add[:, 0], o_add[:, 1], alpha=0.5, color=color[:3], zorder=10)
                    ax.add_collection(lc_add)
                elif perspective == "los":
                    major_obj = plt.Circle((self.ps[3 + i].y, self.ps[3 + i].z), self.ps[3 + i].r, alpha=0.7, fc=fc, zorder=10)
                    ax.add_patch(major_obj)
        else:
            for i in range(self.sim.N_active - 2):

                fc = 'y'
                try:
                    fc = celest_colors[2+i]
                    if not isinstance(fc, str):
                        fc = 'y'
                except:
                    pass

                if perspective == "topdown":
                    major_obj = plt.Circle((self.ps[3 + i].x, self.ps[3 + i].y), self.ps[3 + i].r, alpha=0.7, fc=fc, zorder=10)
                    ax.add_patch(major_obj)
                elif perspective == "los":
                    major_obj = plt.Circle((self.ps[3 + i].y, self.ps[3 + i].z), self.ps[3 + i].r, alpha=0.7, fc=fc, zorder=10)
                    ax.add_patch(major_obj)

        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', visible=True)
        if ax == self.axs[0]:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=True)
        else:
            plt.setp(ax.get_yticklabels(), horizontalalignment='right', visible=False)
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=False)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')

    def __update_interactive(self, val, slider=None, smoothing_slider=None):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        lvls = np.linspace(slider.val[0], slider.val[1], 25)

        if self.cf is not None:
            for tpcf in self.cf.collections:
                tpcf.remove()
            # Update the image's colormap
            np.seterr(divide='ignore')
            logdens = np.where(self.dens > 0,
                               np.log(gaussian_filter(self.dens, smoothing_slider.val, mode='constant')), 0)
            np.seterr(divide='warn')
            self.cf = self.axs[0].contourf(self.X, self.Y, logdens / np.log(10), levels=lvls, cmap=self.cmap,
                                           vmin=slider.val[0], vmax=slider.val[1], zorder=2)

        if self.c is not None:
            for tpc in self.c.collections:
                tpc.remove()
            np.seterr(divide='ignore')
            logdens = np.where(self.dens > 0,
                               np.log(gaussian_filter(self.dens, smoothing_slider.val, mode='constant')), 0)
            np.seterr(divide='warn')
            self.c = self.axs[0].contour(self.X, self.Y, logdens / np.log(10), levels=lvls, cmap=self.cmap,
                                         vmin=slider.val[0], vmax=slider.val[1], zorder=3)

        if self.cb_interact is not None:
            self.cb_interact.norm.vmin = slider.val[0]
            self.cb_interact.norm.vmax = slider.val[1]

        if self.scatter is not None:
            self.scatter.set_norm(colors.Normalize(vmin=slider.val[0], vmax=slider.val[1]))

        # Redraw the figure to ensure it updates
        self.fig.canvas.draw_idle()

    def add_densityscatter(self, ax, x, y, density, perspective, **kwargs):
        kw = {
            "cb_format": '%.2f',
            "zorder": 1,
            "cfilter_coeff": 1,
            "vmin": None,
            "celest_colors": 'y'
        }
        kw.update(kwargs)

        ax = self.axs[ax]
        self.__setup_ax(ax, perspective=perspective, celest_colors=kw["celest_colors"])

        logdens = np.where(density > 0, np.log(density), 0)
        self.scatter = ax.scatter(x, y, c=logdens / np.log(10), cmap=self.cmap, marker='.', vmin=kw['vmin'], s=1.,
                                  zorder=kw['zorder'])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # cax.tick_params(axis='both', which='major', labelsize=8)
        self.cb_interact = plt.colorbar(self.scatter, cax=cax, orientation='vertical', format=kw['cb_format'])

    def add_triplot(self, ax, x, y, simplices, perspective, zorder=1):
        ax = self.axs[ax]
        self.__setup_ax(ax, perspective=perspective)
        ax.triplot(x, y, simplices, linewidth=0.1, c='w', zorder=zorder, alpha=1)

    def add_colormesh(self, ax, X, Y, dens, contour=True, fill_contour=False, **kwargs):
        kw = {
            "cb_format": '%.2f',
            "logmin": 0,
            "logmax": 0,
            "zorder": 1,
            "cfilter_coeff": 1,
            "numlvls": 10,
            "celest_colors": 'default'
        }
        kw.update(kwargs)

        ax_obj = self.axs[ax]
        self.__setup_ax(ax_obj, perspective="topdown", celest_colors=kw["celest_colors"])
        self.X = X
        self.Y = Y
        self.dens = dens

        np.seterr(divide='ignore')
        logdens = np.where(dens > 0, np.log(gaussian_filter(dens, kw['cfilter_coeff'], mode='constant')), 0)
        lvls = np.linspace(np.min(np.log(dens)[np.log(dens) > 0]) + kw['logmin'],
                           np.max(np.log(dens)[np.log(dens) > 0]) - kw['logmax'], kw['numlvls']) / np.log(10)
        np.seterr(divide='warn')

        if contour:
            self.c = ax_obj.contour(X, Y, logdens / np.log(10), cmap=self.cmap, levels=lvls,
                                    zorder=kw['zorder'])  # self.content[f"ax{ax}_c"]

        if fill_contour:
            self.cf = ax_obj.contourf(X, Y, logdens / np.log(10), cmap=self.cmap, levels=lvls, zorder=kw['zorder'] - 1)
        else:
            pm = ax_obj.pcolormesh(X, Y, logdens / np.log(10), cmap=self.cmap, shading='auto', zorder=kw['zorder'] - 1)

        divider = make_axes_locatable(ax_obj)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.tick_params(axis='both', which='major', labelsize=8)
        self.cb_interact = plt.colorbar(self.cf, cax=cax, orientation='vertical', format=kw['cb_format'])
